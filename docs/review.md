# Engineering review — 2026-08-07

A point-in-time audit of the codebase at commit `dc30321`, covering correctness, security and
cost. This is a **findings document**, not a roadmap: [PLAN.md §6](../PLAN.md) remains the single
roadmap, and the items below feed into its Phase 10 rather than competing with it.

Each finding states what was checked, what was found, and what it would take to fix. Severity is
about consequence, not effort.

**Scope note.** Everything here was verified against the running stack — the live API, the live
database, and the three real 200–535 page prospectuses now in `var/`. Nothing below is inferred
from reading code alone, which is the discipline [CLAUDE.md §7](../CLAUDE.md) exists to enforce.

---

## Summary

Status is kept current as findings are closed; the finding text itself stays as it was written,
so what was true at `dc30321` remains readable.

| # | Finding | Area | Severity | Status |
|---|---|---|---|---|
| 1 | Read-only role can read the audit trail and other users' questions | Security | **High** | Fixed |
| 2 | No rate limiting on login | Security | **High** | Fixed |
| 3 | No password strength policy | Security | Medium | Fixed |
| 4 | Per-document cost cap is too low for real documents | Cost | **High** | Fixed (Batch API deferred) |
| 5 | No security response headers | Security | Medium | Fixed |
| 6 | Agent answers unsupported questions confidently instead of refusing | Correctness | **High** | Fixed |
| 7 | No document upload in the UI | Gap | Medium | Fixed |
| 8 | Portfolio page runs N rule evaluations per request | Performance | Medium | Fixed |
| 9 | `rating_agency` extraction accuracy is 0.50 | Correctness | Medium | Fixed |
| 10 | Vision/OCR path has never run against a real provider | Coverage | Medium | Open |
| 11 | Retrieval runs on a placeholder embedder | Quality | Medium | Part-fixed |
| 12 | No dependency vulnerability scanning in CI | Security | Low | Fixed |
| 13 | Review-queue pages are unbounded | Performance | Low | Fixed |
| 14 | Agent answers a question about one instrument with all of them | Correctness | Medium | Fixed |
| 15 | A covenant question about one document is answered from every document | Correctness | **High** | Fixed |
| 16 | Ingesting a document does not make it searchable | Correctness | **High** | Fixed |

Findings 14, 15 and 16 were found after the original audit — 14 on 2026-08-10 while fixing 6, and
15 and 16 on 2026-08-13, both from one user asking an ordinary question about a document they had
uploaded. Their entries are at the end of the Correctness section.

---

## Security

### 1. The read-only role can read the audit trail — High

**Checked:** `information_schema.role_table_grants` for `opuscovintel_ro` against the allowlist in
[`app/agent/sql_guard.py`](../app/agent/sql_guard.py).

The SQL guardrail's allowlist deliberately excludes six operational tables, and its comment is
explicit about why:

> exposed each reviewer's identity and notes, every other user's questions and answers, raw cached
> model output, and the audit trail itself — an agent able to read the record of what it did.

But the Postgres grant does not match that intent. The read-only role still holds `SELECT` on all
six:

```
audit_logs · human_reviews · query_logs · llm_calls · llm_cache · extraction_jobs
```

This matters because [`docker/postgres/init`](../docker/postgres/init) calls the grant **"the
actual boundary"** and the guardrail "defence in depth". For these six tables the boundary was
never moved — only the defence in depth exists. Anything that bypasses the SQL parser (a parser
bug, a future code path using the read-only session directly) reaches them.

Precedent already exists: the Phase 9 migration revokes exactly this grant for `users` and
`user_sessions`. The same treatment should extend to the other six.

**Fix:** a migration revoking `SELECT` on those six tables from `opuscovintel_ro`, guarded on the
role existing, mirroring `20260807_0652_users_and_sessions.py`. Then a test asserting the read-only
role gets `permission denied` — which will also catch anyone re-widening the grant later.

**Fixed.** [`20260810_0733_revoke_operational_tables_from_readonly.py`](../migrations/versions/20260810_0733_revoke_operational_tables_from_readonly.py)
revokes all six; verified against the running database, which now grants the role eleven tables
rather than seventeen, and round-trips cleanly on `downgrade`.
[`tests/test_readonly_grants.py`](../tests/test_readonly_grants.py) proves the denial by
connecting *as the role*, after first asserting the grant was there to take away — otherwise
every denial would pass in a test database that never received it. Its last test pins the
readable set to the guardrail's allowlist, so the next table to arrive cannot inherit `SELECT`
from the init script's `ALTER DEFAULT PRIVILEGES` unnoticed.

**It broke two things, and the suite did not notice.** `make eval` and `make cost-report` both
read `llm_calls` through the read-only session and now exit with `permission denied`. Neither is
the agent — an operator asking what the pipeline spent is entitled to the ledger — so both were
moved to the app role. Found by running the commands, not by the 810 tests that were green, which
is [CLAUDE.md §7](../CLAUDE.md) making its point again: the suite runs everything read-write, so
no test in it can see a grant.

---

### 2. No rate limiting on login — High

**Checked:** `grep` for any rate-limit middleware across `app/`. There is none.

`POST /auth/login` and `POST /ui/login` accept unlimited attempts from a single client. The scrypt
cost (~170 ms per attempt, measured) slows an attacker to roughly 6 guesses/second per connection,
which is a side effect rather than a control — it does not stop a distributed or patient attack,
and it does not stop credential stuffing against a known username.

The one mitigation that *is* deliberate — identical responses and identical timing for every
failure mode — prevents username enumeration but does nothing about brute force.

**Fix:** per-IP and per-username attempt limiting with exponential backoff. Since sessions are
already Postgres rows, a `login_attempts` table fits the existing design without adding Redis.
Consider a lockout threshold with an audit-logged unlock, since this is an internal tool where an
operator can unlock an account.

**Fixed.** [`app/auth/rate_limit.py`](../app/auth/rate_limit.py) over a `login_attempts` table, no
Redis. Five failures per username and twenty per client address, then each further attempt waits
2s, 4s, 8s … capped at fifteen minutes; a success clears the count.

Two departures from the suggestion above, both deliberate. **No lockout** — a threshold that
disables an account hands anyone who knows a username a denial-of-service against that person, and
backoff bounds an attacker without needing an operator at 2am. **No off switch** — a flag that
disables rate limiting is the same shape as `AUTH_ENABLED=false`, and this one has no demo to
justify it.

Enforcement sits inside `AuthService.authenticate`, before the password is checked, so both login
paths inherit it and a future third caller cannot forget it; it raises rather than returning, so
forgetting to handle it is a 500 rather than an unlimited endpoint. Verified live against the
running API: attempts 1–5 answer 401, the sixth 429 with `Retry-After: 2`, the correct password is
refused while throttled, and after the wait the next attempt is 401 again with the delay doubled
to 4s.

One thing this cost: the attempt row has to be committed by the limiter itself. `get_session`
rolls back whenever a handler raises, and a failed login raises `HTTPException(401)` — so the row
counting the failure is destroyed by that failure, and the counter never passes one. The
regression test rolls the transaction back by hand, because the suite overrides `get_session` and
an HTTP-level test passes either way.

**Still open:** behind a proxy, `request.client.host` is the proxy, so the per-IP limit becomes
global. [docs/deploy.md §6](deploy.md) says what to configure.

---

### 3. No password strength policy — Medium

**Checked:** `AuthService.create_user` and `LoginRequest`.

`hash_password` refuses only an *empty* password. `create_user` accepts anything else, and
`LoginRequest` declares `min_length=1`. A one-character password is currently valid.

The CLI prompts with confirmation and never takes the password as an argument, which is right —
but nothing enforces a floor.

**Fix:** a minimum length (12+ for a passphrase-style policy) enforced in `create_user`, so both
the CLI and any future path inherit it. Avoid composition rules ("one symbol, one digit"); length
is the property that matters. Optionally check against a breached-password list, though for an
internal tool with operator-created accounts that is likely over-engineering.

**Fixed.** `validate_password` in [`app/auth/passwords.py`](../app/auth/passwords.py): twelve
characters, no composition rules, plus one context rule — the password may not contain the
username. The breached-password list stays declined, for the reason above. The floor is a module
constant rather than a setting: a per-environment minimum is one someone lowers for a demo and
never raises.

Called from `create_user` and `set_password` — where a password is *chosen* — and deliberately not
from `hash_password`, which the login path also calls to re-hash when the scrypt cost is raised.
Enforcing it there would turn every account created before the policy into a 500 at the moment its
owner types the right password; there is a test for that. `set_password` validates before it
mutates, so a rejected change leaves the account and its sessions alone.

`LoginRequest.min_length` stays at 1. The floor belongs at the point of choosing, not at the door.

Verified live through the CLI: `user-add` and `user-passwd` both print the message and exit
non-zero rather than raising. `user-passwd` had no `ValueError` handling — nothing it called could
raise one before — so it gained the same handler `user-add` already had.

---

### 5. No security response headers — Medium

**Checked:** `grep` for `Content-Security-Policy`, `X-Frame-Options`, `X-Content-Type-Options`,
`Strict-Transport-Security`. None are set.

This matters more here than in a typical app because the UI renders **clause text lifted verbatim
out of third-party PDFs**. Jinja autoescaping is on and tested, so the known path is covered — but
a CSP is the layer that holds when an escaping bug slips through, and there is no reason not to
have one on a page that loads no external resources.

**Fix:** a small middleware setting `Content-Security-Policy: default-src 'self'`,
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin`, and
`Strict-Transport-Security` when `SESSION_COOKIE_SECURE` is on. The UI uses no inline scripts or
external assets, so a strict policy should apply without exceptions.

**Not a finding:** CSRF. Both cookie paths set `SameSite=lax`, which blocks cross-site form POSTs.
Worth an explicit test so it cannot regress silently.

**Fixed.** [`SecurityHeadersMiddleware`](../app/core/middleware.py) sets all five, as middleware
rather than per-route so a page added later inherits the policy — the failure mode of the
decorator-based alternative is a new screen with no policy and nobody noticing. Verified on every
surface including a 404 and a static file. HSTS is gated on `SESSION_COOKIE_SECURE`, so it is
absent on the plain-HTTP local stack and present the moment a deployment is HTTPS.

**The last sentence of the fix above was wrong, and a browser is the only thing that could say
so.** "The UI uses no inline scripts or external assets" was true when it was written and stopped
being true when finding 7 landed the Angular client app. Two exceptions were needed, and they are
opposite in kind:

* **`/app` was fixed rather than exempted.** Angular's critical-CSS inliner emits an inline
  `<style>` block *and* an `onload="this.media='all'"` attribute on the stylesheet link. Under
  `default-src 'self'` the browser blocks both, the real stylesheet never leaves `media="print"`,
  and every screen renders **completely unstyled** — confirmed by turning the setting back on and
  loading the page. Rather than weaken the policy for the pages that render clause text,
  `inlineCritical` is off in [`frontend/angular.json`](../frontend/angular.json), which removes
  both constructs. A test asserts the built `index.html` still contains neither, and a second
  asserts the setting itself so a checkout with no build still catches a flip.
* **`/docs` was exempted, narrowly.** Swagger UI loads its bundle from a CDN, takes its favicon
  from the FastAPI site, and bootstraps from an inline `<script>` — under the application policy
  the page is blank. It gets its own policy carrying `'unsafe-inline'`. That is acceptable
  *there and only there*: `/docs` is disabled in production, and what it renders is our own
  OpenAPI schema rather than text out of a third-party PDF. A test pins that the exception cannot
  leak onto a page that renders clauses.

The property that matters is asserted directly rather than implied by the header value: the
application policy permits no inline execution of any kind. That is the layer standing behind
Jinja autoescaping, and `'unsafe-inline'` is the easiest thing in the world to add while chasing a
page that will not render — which is exactly what happened twice while writing this.

CSRF got its test, on the **form** login: `app/web/routes.py` sets its own cookie inline, separate
from the JSON path's helper, and only the latter was covered. Two `set_cookie` calls means one can
regress alone.

---

### 12. No dependency vulnerability scanning — Low

CI runs lint, types, tests, a migration round-trip and a container build — good coverage, and
notably it fails if a provider credential is ever added. It does not check dependencies for known
vulnerabilities.

**Fix:** `uv pip audit` (or `pip-audit`) as a CI job, plus Dependabot for the lockfile. Low
severity only because the dependency surface is small and pinned.

**Fixed.** A sixth CI job, `dependency audit`, plus `.github/dependabot.yml`.
Both trees audit **clean today**, which is the point of landing it now: the scan starts green, so
the first red tick is a real regression rather than a backlog somebody has to triage before the
job is worth anything.

**Amended 2026-08-16: `.github/dependabot.yml` is deleted, so Dependabot opens no more scheduled
version-update PRs.** Weekly grouped updates across three ecosystems still produced more pull
requests than anyone was reading, and a PR nobody reads is worse than no PR — it trains the habit
of merging dependency bumps unlooked-at. The detector is unchanged: `dependency audit` runs
`pip-audit` and `npm audit` on every CI run and `make audit` runs both locally, so a known
vulnerability still turns the build red. What is gone is the automatic upgrade PR, which makes
bumping a lockfile a deliberate act again. Dependabot **security** updates are a repository
setting rather than this file, and were not touched.

`uv pip audit` **does not exist** — a reasonable guess, and wrong, which one `--help` settled. The
tool is PyPA's `pip-audit`, run through `uvx` at a **pinned** `2.10.1`: a scanner that silently
upgrades itself is a supply-chain surface of its own.

Three decisions worth stating:

* **It fails the build.** A warning nobody is required to read is theatre. The escape hatch is
  [`.github/pip-audit-ignore.txt`](../.github/pip-audit-ignore.txt) — advisory ID, reason, date,
  author — so a suppression is a decision somebody signed rather than a flag somebody flipped, and
  `pip-audit` prints the ignored count so it stays visible in the log.
* **Dev dependencies are included** (`--all-groups`). They execute in CI and on developer machines,
  which is exactly where a compromised test dependency would run.
* **The client tree is audited too**, which this finding did not ask for. It ships to a browser and
  is the same surface; `npm audit` reads the lockfile, so the job needs no `npm ci`.

Verified by running the job's commands verbatim, then **proving the guard fires**: an audit of a
knowingly-vulnerable pin (`jinja2==2.11.3`) reports four advisories and exits 1, and the same run
with those four ignored exits 0 reporting "4 ignored". A scanner only trusted to pass has not been
tested. `make audit` runs the same thing locally, so a red tick is reproducible without pushing.

---

## Cost

### 4. The per-document cap is too low for real documents — High

**Checked:** `extract --all --dry-run` against the three real prospectuses.

| Document | Pages | Candidates | Prompt tokens | Worst case | vs `$2.00` cap |
|---|---|---|---|---|---|
| 2021 trust certificate | 535 | 93 | 468k | $20.94 | **10× over** |
| Dubai base prospectus | 201 | 51 | 257k | $11.48 | **6× over** |
| 2025 GMTN | 480 | 19 | 96k | $4.28 | **2× over** |

All three would abort mid-document. The guard is working exactly as designed — the *default* is
calibrated for the synthetic fixtures, which are one to five pages.

A mid-document abort is worse than it sounds: you pay for the calls made and get a partially
extracted document, which is harder to reason about than a clean refusal.

**Fix, in order:**

1. Raise `MAX_COST_PER_DOCUMENT_USD` to something realistic for 500-page documents (~$8), as a
   deliberate configuration change with the number written down.
2. Make the guard **refuse to start** a document whose dry-run ceiling already exceeds the cap,
   rather than discovering it partway. Failing before spending is strictly better than failing
   after.
3. Build the **Batch API** path that [PLAN.md §2](../PLAN.md) specifies and nothing implements —
   50% off, and backfilling a corpus is exactly the non-interactive workload it is for.

Worth noting the estimator is honest about being a ceiling: it prices every completion at the full
8,000-token budget. Real spend on the synthetic corpus came in at $0.60 against a much higher
ceiling. Expect real cost in the **$3–7** range for all three documents, but confirm it by running
the cheapest one first and reading `make cost-report`.

**Fixed (steps 1–2); step 3 deferred.** `MAX_COST_PER_DOCUMENT_USD` was raised to `8.00`
([`app/core/config.py`](../app/core/config.py), `.env.example`, [deploy.md](deploy.md)) — sized
for the $3–7 real spend with headroom, the number written down — and **tightened again to `5.00`
on 2026-08-16**, half the $10.00 global ceiling, so one document cannot exhaust the budget. That
second change refuses nothing the first admitted: no document in the corpus prices between the two
figures, checked before changing it. And the pipeline now **refuses
before spending**: after candidate detection and before the first billable call,
[`ExtractionPipeline`](../app/extract/pipeline.py) prices the whole document with the same
`estimate_candidate_cost` the `--dry-run` CLI uses, and if that ceiling exceeds the cap it marks
the document `budget_exceeded` and stops at $0 rather than paying its way to a partial extraction.
The estimator was refactored so the CLI and the guard share one implementation and cannot drift.
A preflight refusal carries a distinct job message and a `budget_preflight_refused` flag, so an
operator can tell a $0 clean refusal from a mid-document abort. On the numbers above, only the 2025
GMTN ($4.28) now starts; the two larger documents are refused up front until they run through the
Batch API path.

Step 3 — the **Batch API** — stays open as a separate item (PLAN.md §6 finding 4.3). It halves the
cost of exactly the two documents the raised cap does not yet admit, so it is the natural next step,
not part of this change.

---

## Correctness

### 6. The agent answers unsupported questions confidently — High

**Reproduced live:**

```bash
curl -X POST localhost:8000/query -d '{"question":"What is the CEO of the issuer paid?"}'
```

Returns `refused: false`, `confidence: 0.95`, `citations: []`, and a generic list of instruments.
Executive compensation appears nowhere in the corpus.

The cause is routing, not retrieval: the intent classifier sends the question to
`instrument_lookup`, which answers from structured rows and never needs retrieval — so the refusal
path is never reached. [CLAUDE.md §1.5](../CLAUDE.md) requires a refusal here.

The benchmark misses it because the one unanswerable question in the golden set classifies as
`unsupported` and takes the correct path. **10/10 passing is measuring the wrong thing for this
failure mode.**

A confident, uncited answer is the exact output this system is designed not to produce, which is
why this is High despite being pre-existing.

**Fix:** add unanswerable questions to the golden set that classify as `instrument_lookup`,
`portfolio_query` and `covenant_breach_check`; confirm they fail; then make the non-retrieval
intents refuse when nothing in the question maps to a known entity or field. Treat zero citations
plus high confidence as a contradiction the verify node should catch.

**Fixed.** G11–G13 were added first and failed on both read paths — the deterministic service has
the same defect, since `_instruments_for` falls back to every instrument when the question names
none. Then [`app/query/answerable.py`](../app/query/answerable.py): a question routed to a
structured intent is answerable only if **every salient word** in it is one the system has a
meaning for — a field it holds, a controlled vocabulary it knows, or the name of something in the
database. One unknown word and the answer is a refusal that names the word.

That is stricter than the rule suggested above, because the weaker one does not fix this bug.
"Issuer" *is* a field, so "nothing in the question maps to a known entity or field" is false for
the reported question, and it is doubly false for `What is the CEO of Synthetic Green Energy Sdn
Bhd paid?`, which names a real issuer. A whitelist also fails in the safe direction: an
unrecognised phrasing produces a visible refusal naming what it did not understand, where a
blocklist of out-of-scope topics is never finished and each gap is a confident wrong answer.

Enforced in the graph at three points, deliberately: `_retrieve` computes it (so an unanswerable
breach check never runs the rules engine or the portfolio SQL), `_synthesize` refuses on it, and
`_verify` refuses again — synthesis is a per-intent `match` and is where the next intent will be
added without its refusal branch.

Reproduced against the running API before and after. The three questions now return
`refused: true`, `confidence: 0.0`, no citations, and name the terms; twenty-two ordinary
questions across all three intents still answer, and `make eval` reports 13/13 on both paths with
refusal precision and recall 1.00.

The guard's real risk is over-refusal, so the test file is weighted that way — 22 questions that
must *not* be refused against 8 that must be.

**Also found while verifying:** the agent answers "Who is the issuer of the Green Ijarah Sukuk?"
by listing all three instruments. The graph's `_retrieve` calls `get_instrument` with no name
filter, where the deterministic path narrows to the instrument the question names. Over-broad
rather than unsupported, and a separate defect — recorded as finding 14 rather than folded in
here, and fixed there. (The deterministic path turned out not to narrow this one either; see
finding 14.)

### 9. `rating_agency` extraction accuracy is 0.50 — Medium

The one weak field in `make eval`: P 0.50 / R 0.50 on the LLM path, R 0.50 on rules, against ≥0.94
for every other field. Both extractors miss the same label, which points at normalisation rather
than at either model — likely the Malaysian national-scale suffixes (`(m)`, `id`) that
[CLAUDE.md §6](../CLAUDE.md) requires be normalised while preserving the raw string.

**Fix:** start in [`app/rules/ratings.py`](../app/rules/ratings.py). Write the regression test,
watch it fail, then fix.

**Fixed — both paths now P 1.00 / R 1.00 / F1 1.00** (LLM 0.50 → 1.00, rules 0.67 → 1.00), against
a target of ≥0.9.

**The module named above was the wrong one.** `app/rules/ratings.py` normalises the *notch* and
already strips `(m)` / `id`; `BBB-` parses there without complaint. The missing value was the
*agency*, which that module never touches. Two defects were behind the number, and only running the
extractors against the corpus separated them — with two labelled instances, "0.50" was one hit and
one miss.

* **The agency was looked up inside a single chunk.** In `rating-report.pdf` the trigger sentence
  is a 224-character chunk of its own naming no agency; MARC is in the chunk before it and the one
  after. `_agency_near` searches only the text it is handed, so no context window could reach it —
  `_CONTEXT_CHARS` was already 400, nearly twice the chunk. Both extractors missed the same label
  for the same structural reason, which is exactly why it read as a normalisation bug.
  [`resolve_document_agency`](../app/extract/rule_extractor.py) now resolves the one agency a
  document names and both paths use it as a *fallback* when a span names none. **Exactly one, or
  nothing**: two named agencies resolve to nothing rather than a guess, because MARC's `A-` and
  RAM's `AA3` are different scales and a wrong attribution is worse than an absent value — the same
  choice finding 14 made. A span that names its own agency always wins.
* **The LLM path wrote `"unknown"` as a value.** `rule_extractor` is explicit that "an absent
  agency is a fact, 'unknown' as a value is noise" and omits it; the pipeline guarded on
  truthiness, and `RatingAgency.UNKNOWN` is a non-empty `StrEnum` member. That string was the false
  positive holding LLM precision at 0.50 while the rule path sat at 1.00 — the gap between the two
  paths was this one line, not the models.

**Two version bumps, one of them nearly missed.** `EXTRACTOR_VERSION` → `rules-v3` and
`LLM_EXTRACTOR_VERSION` → `llm-pipeline-v3`. The extraction identity (CLAUDE.md 1.7) skips a
document whose identity already ran, so without the bumps every already-extracted document keeps
its agency-less and `"unknown"` rows for ever. The second bump was missed on the first attempt and
found only by re-running, where the pipeline printed "skipped; extraction identity already
satisfied" for the very document the fix targeted.

**One cross-document hazard, found by reading the caller.** The resolved agency lives on the
pipeline instance, and `extract --all` reuses one instance for the whole corpus — so a document
whose rule pass failed would have inherited the previous document's agency and stamped its
covenants with it. Cleared per run, with a test that drives the corpus through one instance and
asserts the agency-less trust deed claims none.

### 14. The agent answers about one instrument with all of them — Medium

*Found 2026-08-10 while verifying the fix for finding 6, not in the original audit.*

**Reproduced live:** `Who is the issuer of the Green Ijarah Sukuk?` returns `3 instrument(s)
matched` and lists the whole universe, at confidence 0.95.

The graph's `_retrieve` calls `get_instrument(session)` with no filter for `instrument_lookup`,
and the formatter prints whatever it is handed. The deterministic path does narrow — its
`_instruments_for` filters by the names the question mentions — so the two read paths disagree,
and the agent is the looser one.

Less dangerous than finding 6: the answer contains the right row and is not about something the
data cannot address. Still wrong — a question about one instrument is answered about three, and
at a portfolio's scale that is a page of noise around the fact somebody asked for.

**Fix:** narrow in `_retrieve` the way the deterministic path does, using `mentioned_entities`
against instrument and issuer names. Then decide whether the golden set should pin it — a
question naming one instrument whose answer must not name the others.

**Fixed, and it was worse than written above.** Two defects, not one.

*The claim that the deterministic path narrows was only half true.* It narrows when the question
quotes the stored name in full. The instrument is stored as `RM300m Green Ijarah Sukuk` and the
question said "the Green Ijarah Sukuk", so `mentioned_entities` — literal substring matching —
matched nothing and **both** paths answered about all three. `mentioned_entities` now also accepts
a contiguous run of at least two of a name's words, provided no other candidate shares that
phrase. Ambiguity still narrows to nothing and answers broadly: a phrase two instruments share
names neither, and over-answering is noise where guessing is a wrong attribution.

*The portfolio branch had the same defect, and there it produces a wrong number.*
`What is the total exposure of the Green Fixed Income Fund portfolio?` listed holdings from
`Income Growth Fund` too and totalled all of them — a figure presented as one fund's exposure that
was not. Narrowing covers instrument lookups, breach checks and portfolio queries alike, and it
runs after the answerability check from finding 6, which needs every name to decide whether the
question can be answered at all.

Both read paths now agree on all four questions checked live, `make eval` still reports 13/13 with
faithfulness 1.00, and a question that names nothing — `Which holdings would breach their rating
trigger?` — still evaluates all three instruments, which is the regression the narrowing could
most easily have caused.

**Noticed while fixing, not fixed:** `_format_portfolio_answer` reads the `get_portfolio_holdings`
result before the `run_read_only_sql` one, so the generated SQL is executed and its rows are then
ignored whenever holdings exist. Dead work rather than a wrong answer, but the SQL path is what
PLAN.md §5 says portfolio aggregation runs on.

### 15. A covenant question about one document is answered from every document — High

*Found 2026-08-13, in ordinary use, by a user who had uploaded a 201-page base prospectus and
asked about it.*

**Reproduced live:**

```
Q: What is the cross-default threshold in the Dubai prospectus?
A: cross_default · threshold RM30 million · (page 1)
   refused: false · confidence 0.85 · 5 citations
```

Every one of those five citations was from `scanned-sample.pdf`, `trust-deed.pdf` and
`sample-prospectus.pdf` — the *synthetic* fixtures. Not one was from the user's document. The
answer names a threshold that appears nowhere in the document the question asked about, and
carries citations that make it look checked.

`covenant_lookup` narrows by covenant *type* and by nothing else. It reads every covenant row in
the corpus, so a question that names a document is answered from whichever documents happen to
have been extracted. The deterministic path narrows by instrument when the question names one, and
by document never.

**Two things made it invisible until now.** The golden set asks about a corpus where every
document is extracted, so "all covenants" and "this document's covenants" are the same rows. And
the real prospectuses are ingested but *not* extracted — they cost more than the per-document cap
(finding 4) — so the only rows available to answer with belong to something else.

This is the third of a family: finding 6 answered a question the data could not address, finding
14 answered about one instrument with all of them, and this answers about one document with
another document's covenants. It is the worst of the three, because it is the case a user hits
without trying and the wrong figure is a covenant threshold.

**Fixed.** `covenant_lookup` now narrows to what the question names, on both read paths:

* a **document**, matched on a word that belongs to exactly one filename and is not a generic
  document word — "dubai" identifies one document, "prospectus" and "trust" identify none;
* an **instrument or issuer**, via the same `mentioned_entities` used by the other intents.

When the named document has no extracted covenants, the answer says exactly that, names the
document, and refuses — rather than reaching for rows from elsewhere. That refusal is the right
answer for every real document in the corpus today, and it says why, which "no supporting evidence"
did not.

A question that names nothing still returns the whole corpus, which is what makes "what
cross-default thresholds do we have?" answerable.

### 16. Ingesting a document does not make it searchable — High

*Found 2026-08-13, alongside finding 15 and by the same user. It is why finding 15 was reachable
at all.*

**Checked:** the corpus, against the columns retrieval actually reads.

```
Dubai_12B_Project_Drive_-_Base_Prospectus_1.pdf   870 chunks   0 embedded   0 fts
2021-trust-certificate-prospectus.pdf            2677 chunks   0 embedded   0 fts
2025-gmtn-prospectus.pdf                         2620 chunks   0 embedded   0 fts
```

Every real document in the corpus was invisible to every question asked about it. Not
under-ranked — **absent**: the vector leg reads `embedding` and the keyword leg reads `fts`, and
both were null for 6,167 of 6,200 chunks.

Ingestion stopped at `chunked`. Indexing was `opuscovintel index`, a separate command, and the
Documents screen reported those documents with a healthy status and no hint that nothing could
find them. So the user asked about a document that was, as far as retrieval was concerned, not
there — and finding 15 answered from the synthetic fixtures instead.

Two failures compounding: a pipeline with a manual step in the middle, and an interface that
called the step before it "done".

**Fixed.** `ingest_and_index` parses, chunks and indexes as one operation, used by the worker and
by `POST /documents/{id}/process` alike — the endpoint matters because a document ingested through
it leaves no queued parse job for the worker to claim, so it would never have been indexed at all.

The status endpoint gained `searchable`, and `terminal` now also requires that no job is still
queued or running: a document whose `embed` job is pending is not finished, and the client polls
until the server says it is. The Documents screen says which of the two a document is, and the
corpus list marks the ones that are ingested and unfindable — the three above stay marked until
someone runs `opuscovintel index`, which is the honest thing for it to say.

Asserted on the columns rather than the status: a test that checked `status == "embedded"` would
pass against a pipeline that set the status and indexed nothing.

---

## Performance

### 8. The portfolio page runs N rule evaluations per request — Medium

[`app/web/routes.py`](../app/web/routes.py) `portfolio_detail` calls `evaluate_covenant_rule` once
per holding, and each call issues several queries. Two holdings is fine; a realistic 200-bond
portfolio would issue hundreds of queries per page load.

Reusing the agent's own tool was the right call — a second rules implementation would eventually
disagree with the first, and the one on screen is the one someone acts on. The fix is batching, not
duplicating.

**Fix:** a batch entry point that loads covenants and triggers for many instruments in one query
and evaluates in memory. The rules engine itself is pure functions over loaded data, so this is a
data-loading change, not a logic change. Add pagination while you're there.

**Fixed.** `evaluate_covenant_rules` in [`app/agent/tools.py`](../app/agent/tools.py) loads
instruments, rating triggers and covenants in **three queries total** regardless of how many
instruments are asked for. A 200-bond portfolio goes from ~600 round trips to 3.

The warning above shaped the implementation more than the batching did. Rather than a second
evaluator, the per-instrument logic was extracted into `_evaluate_loaded`, and **both** the
single-instrument tool the agent calls and the new batch entry point evaluate through it. The
loaders differ; the evaluation cannot. A test asserts the two paths return identical data
field-for-field, so a future edit to one that does not reach the other fails rather than drifts.

**The query count is asserted, not assumed.** A batch entry point that loops internally satisfies
every behavioural test — it returns the right answers, just slowly, which is precisely the defect.
So the tests count SQL statements through a SQLAlchemy event listener and pin the total at three.
Verified the way this codebase requires: with the batching replaced by a loop, the twelve
behavioural tests still pass and only the two counting tests go red.

### 13. Review-queue and instrument pages are unbounded — Low

The review queue renders up to 100 items with no pagination; `/ui/instruments` requests 200. Both
are fine at demo scale and will not be at production scale.

**Fixed.** Both take a `page` parameter and render 50 rows at a time through one `_page_window`
helper. Verified live against the development database, which holds 124 pending items: page 1
reports *"Showing 1–50 of 124 · page 1 of 3"* with **50** rows and a disabled *Previous*; page 3
reports *"Showing 101–124 of 124"* with 24 rows and a disabled *Next*.

Three small decisions. A page past the end **clamps rather than 404s** — a queue drains while
somebody is reading it, so a stale page number is ordinary and the last page beats an error.
`page=0` is a 422, because `ge=1` on the query parameter is cheaper than defending against a
negative offset. And the control is **hidden when there is only one page**: the instrument list has
three rows and renders no pagination at all, because a control that can never do anything is
furniture.

Also folded in: the review page called `count_pending()` twice per request, for the nav badge and
the heading, which always showed the same number.

---

## Gaps

### 7. No document upload in the UI — Medium

The UI has four screens; upload is not one of them. Documents load via CLI or
`POST /documents/upload`. For 500-page files the CLI is arguably the better path anyway, but a
user cannot get a document into the system from the interface they were given.

**Fix:** an upload screen posting to the existing endpoint, with the ingestion job's progress
visible — the worker already tracks status in `extraction_jobs`, so there is state to show.

**Fixed**, and it needed a server change first: `extraction_jobs` held the progress and nothing
exposed it, so the only way to know whether an upload had been ingested was to query the database.
`GET /documents/{id}/status` now assembles the document's status, its page and chunk counts, the
per-job timings and the failure message, plus a `terminal` flag. The flag is the contract: the
client polls until the *server* says nothing further will happen, so a status added to the
pipeline later cannot make a client report a half-parsed document as finished.

The screen itself is Angular ([`frontend/`](../frontend)), served from `/app` by the same process,
and shows the two phases separately because they fail differently — bytes leaving the browser
(`HttpClient` progress events) and the worker parsing (polling). A duplicate is reported as a
duplicate rather than an error, and a failure shows the reason the service already wrote down.

Verified in a browser against the running stack, twice: upload → `uploaded`/queued → *Parse now* →
`chunked`, with page and chunk counts, the VLM page count, and both job rows with timings. Ten
client-side unit tests cover the polling contract, the duplicate path and the error text; six
Python tests cover the endpoint, and six more cover the mount (deep links fall back to
`index.html`; a missing bundle still 404s).

**Why Angular rather than more Jinja.** The read screens have no client-side state worth managing
and stay server-rendered; upload has two kinds of progress and an error path, which is where a
client app earns its build step. Both are served from one origin, so the session cookie keeps
`HttpOnly` and `SameSite=lax` — see [docs/deploy.md §6](deploy.md) before splitting them.

### 10. The vision/OCR path has never run live — Medium

`VlmService` and `opuscovintel ocr` are wired but have never made a real call; the OpenAI account
had no credit. [CLAUDE.md §7](../CLAUDE.md) is explicit that wiring is not finishing, and this
exact path already produced one defect (a wrong storage key) found only by running it.

The real corpus makes this cheap to close: `ocr --all --dry-run` reports **12 pages** across the
three prospectuses, about **$0.26** total. Well under the 40-page cap.

### 11. Retrieval runs on a placeholder embedder — Medium

Without `QWEN_API_KEY`, `get_embedder()` returns `HashingEmbedder`. Search still functions, but the
vector half of hybrid retrieval carries no semantic signal — so the Phase 4 claim that hybrid beats
either leg alone is untested with real vectors, and the FTS/kNN candidate legs default off.

This now matters more: the real corpus is ~6,000 chunks rather than ~20.

**Fix:** close [PLAN.md §9 Q2](../PLAN.md) (endpoint region and dimensionality) *before* indexing.
1024 dimensions is baked into the schema; changing it later means re-embedding everything and
rebuilding the HNSW index.

**Half fixed — the decision, not the key.** §9 Q2 is closed: international endpoint, 1024
dimensions, keep both. The schema, the HNSW index and `VECTOR_DIMENSION` already agree on 1024, and
nothing measured argues for a wider vector at ~6,000 chunks. What remains is a funded
`QWEN_API_KEY`, a re-index, and a re-baseline — which is Phase 10.4 and is not something a code
change can close.

**What did change is that running on the placeholder is no longer silent**, which is what made this
finding cost a user an hour. Asking about a real prospectus returned a page of legal advisers and
two director biographies above the negative-pledge clause, and nothing anywhere said that only the
keyword leg was answering. Now:

* the fallback logs once per process, naming the reason and the consequence — *"the vector leg of
  hybrid retrieval carries no semantic signal; only keyword matching is answering"*;
* a query whose vector leg matched **nothing** because the corpus was indexed by a different model
  says so, and names both models. That is the trap waiting for the day the key is added:
  `search_by_vector` filters to chunks embedded by the same model, correctly, and the failure mode
  of that filter is silence.

Neither is retrieval quality. They are the difference between a known limitation and a mystery.

---

## What was checked and found healthy

Worth recording, so a later reader knows these were examined rather than skipped.

- **Container hardening** — multi-stage build, non-root user, no build tooling in the runtime
  image, `uv sync --locked` so the image is reproducible, working healthcheck.
- **CI** — separate jobs for code, schema and image; a migration round-trip with a drift check;
  `REQUIRE_POSTGRES=1` so database tests cannot silently skip; and a step that fails the build if a
  provider credential is ever added.
- **Secrets** — `.env` is git-ignored, `.env.example` is committed, keys are `SecretStr`, and a
  blank key is correctly treated as absent.
- **Cookies** — `HttpOnly`, `SameSite=lax`, `Secure` gated on config, and settings refuse to start
  a production with an insecure cookie.
- **Session handling** — a fresh token per login (no fixation), revocation on logout and on
  password change, expiry checked in SQL rather than in Python, and deactivating a user
  immediately invalidates live sessions.
- **SQL injection** — generated SQL is parsed with `sqlglot` rather than pattern-matched, with a
  table *and* column allowlist, forced `LIMIT`, and a statement timeout.
- **Money handling** — `Decimal` end to end, including across the wire, where Pydantic serialises
  it as a JSON string rather than a float.
- **Test isolation** — a dedicated database, transaction rollback, and no cross-test leakage
  (verified: zero residual rows after a full run).

---

## Suggested order

Grouped by what they buy, hardest-hitting first.

**Now — small, high value**

1. ~~Revoke the read-only role's grant on the six operational tables~~ *(finding 1 — done)*
2. ~~Raise the per-document cost cap and refuse-before-spending~~ *(finding 4 — done; Batch API still open, finding 4.3)*
3. ~~Password minimum length~~ *(finding 3 — done)*
4. ~~Security headers middleware~~ *(finding 5 — done)*

**Next — needs design**

5. ~~Login rate limiting~~ *(finding 2 — done)*
6. ~~Make unsupported questions refuse, and extend the golden set to catch them~~ *(finding 6 — done)*
7. ~~Fix `rating_agency` normalisation~~ *(finding 9 — done; it was scoping, not normalisation)*

**Then — the accuracy question that matters most**

8. Extract one real prospectus and re-baseline every accuracy figure *(PLAN.md Phase 10.1)*
9. Live-verify the OCR path — 12 pages, ~$0.26 *(finding 10)*
10. Real embeddings, after settling dimensionality *(finding 11)*

**Later — scale, not correctness**

11. ~~Batch the portfolio page's rule evaluation; add pagination~~ *(findings 8, 13 — done)*
12. Upload screen *(finding 7)*
13. ~~Dependency scanning in CI~~ *(finding 12 — done)*
14. Batch API for bulk re-extraction *(finding 4.3)*
