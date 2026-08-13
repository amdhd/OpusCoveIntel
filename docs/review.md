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
| 4 | Per-document cost cap is too low for real documents | Cost | **High** | Open |
| 5 | No security response headers | Security | Medium | Open |
| 6 | Agent answers unsupported questions confidently instead of refusing | Correctness | **High** | Fixed |
| 7 | No document upload in the UI | Gap | Medium | Fixed |
| 8 | Portfolio page runs N rule evaluations per request | Performance | Medium | Open |
| 9 | `rating_agency` extraction accuracy is 0.50 | Correctness | Medium | Open |
| 10 | Vision/OCR path has never run against a real provider | Coverage | Medium | Open |
| 11 | Retrieval runs on a placeholder embedder | Quality | Medium | Open |
| 12 | No dependency vulnerability scanning in CI | Security | Low | Open |
| 13 | Review-queue pages are unbounded | Performance | Low | Open |
| 14 | Agent answers a question about one instrument with all of them | Correctness | Medium | Fixed |
| 15 | A covenant question about one document is answered from every document | Correctness | **High** | Fixed |

Findings 14 and 15 were found after the original audit — 14 on 2026-08-10 while fixing 6, and 15 on
2026-08-13 by a user asking an ordinary question about a document they had uploaded. Their entries
are at the end of the Correctness section.

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

---

### 12. No dependency vulnerability scanning — Low

CI runs lint, types, tests, a migration round-trip and a container build — good coverage, and
notably it fails if a provider credential is ever added. It does not check dependencies for known
vulnerabilities.

**Fix:** `uv pip audit` (or `pip-audit`) as a CI job, plus Dependabot for the lockfile. Low
severity only because the dependency surface is small and pinned.

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

### 13. Review-queue and instrument pages are unbounded — Low

The review queue renders up to 100 items with no pagination; `/ui/instruments` requests 200. Both
are fine at demo scale and will not be at production scale.

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
2. Raise the per-document cost cap and refuse-before-spending *(finding 4)*
3. ~~Password minimum length~~ *(finding 3 — done)*
4. Security headers middleware *(finding 5)*

**Next — needs design**

5. ~~Login rate limiting~~ *(finding 2 — done)*
6. ~~Make unsupported questions refuse, and extend the golden set to catch them~~ *(finding 6 — done)*
7. Fix `rating_agency` normalisation *(finding 9)*

**Then — the accuracy question that matters most**

8. Extract one real prospectus and re-baseline every accuracy figure *(PLAN.md Phase 10.1)*
9. Live-verify the OCR path — 12 pages, ~$0.26 *(finding 10)*
10. Real embeddings, after settling dimensionality *(finding 11)*

**Later — scale, not correctness**

11. Batch the portfolio page's rule evaluation; add pagination *(findings 8, 13)*
12. Upload screen *(finding 7)*
13. Dependency scanning in CI *(finding 12)*
14. Batch API for bulk re-extraction *(finding 4.3)*
