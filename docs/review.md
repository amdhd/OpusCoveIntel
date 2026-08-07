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

| # | Finding | Area | Severity |
|---|---|---|---|
| 1 | Read-only role can read the audit trail and other users' questions | Security | **High** |
| 2 | No rate limiting on login | Security | **High** |
| 3 | No password strength policy | Security | Medium |
| 4 | Per-document cost cap is too low for real documents | Cost | **High** |
| 5 | No security response headers | Security | Medium |
| 6 | Agent answers unsupported questions confidently instead of refusing | Correctness | **High** |
| 7 | No document upload in the UI | Gap | Medium |
| 8 | Portfolio page runs N rule evaluations per request | Performance | Medium |
| 9 | `rating_agency` extraction accuracy is 0.50 | Correctness | Medium |
| 10 | Vision/OCR path has never run against a real provider | Coverage | Medium |
| 11 | Retrieval runs on a placeholder embedder | Quality | Medium |
| 12 | No dependency vulnerability scanning in CI | Security | Low |
| 13 | Review-queue pages are unbounded | Performance | Low |

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

### 9. `rating_agency` extraction accuracy is 0.50 — Medium

The one weak field in `make eval`: P 0.50 / R 0.50 on the LLM path, R 0.50 on rules, against ≥0.94
for every other field. Both extractors miss the same label, which points at normalisation rather
than at either model — likely the Malaysian national-scale suffixes (`(m)`, `id`) that
[CLAUDE.md §6](../CLAUDE.md) requires be normalised while preserving the raw string.

**Fix:** start in [`app/rules/ratings.py`](../app/rules/ratings.py). Write the regression test,
watch it fail, then fix.

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

1. Revoke the read-only role's grant on the six operational tables *(finding 1)*
2. Raise the per-document cost cap and refuse-before-spending *(finding 4)*
3. Password minimum length *(finding 3)*
4. Security headers middleware *(finding 5)*

**Next — needs design**

5. Login rate limiting *(finding 2)*
6. Make unsupported questions refuse, and extend the golden set to catch them *(finding 6)*
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
