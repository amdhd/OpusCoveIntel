# PLAN.md — OpusCovIntel Implementation Plan

**Status:** Phases 1–9 built. Phase 10 (accuracy and coverage) is the remaining work — see §6.
**Written:** 2026-08-04 · **Last revised:** 2026-08-07

This plan reconciles two prior drafts — a production-grade spec (`100usd`) and a minimal offline MVP
(`10usd`) — against the actual target: a LangGraph-orchestrated, multi-provider covenant intelligence
MVP on Postgres/pgvector.

---

## 1. What was taken from each draft

| Concern | 100usd draft | 10usd draft | Decision |
|---|---|---|---|
| Datastore | Postgres 16 + pgvector | SQLite + FTS5 | **Postgres + pgvector**, hybrid with `tsvector` |
| Retrieval | vector + keyword + metadata | FTS5 / LIKE | **Hybrid, RRF-fused** |
| Orchestration | LangGraph | forbidden | **LangGraph** for the query agent |
| Extraction | frontier LLM structured | regex only | **Both, in parallel** — see §3 |
| PDF handling | PyMuPDF + pdfplumber + VLM | none | **Full stack**, VLM gated by page confidence |
| Validation | Pydantic + retry | optional | **Pydantic v2 + one feedback retry** |
| **Budget control** | token *logging* | **hard caps + cache + breaker** | **Adopted from the cheap plan — see §2** |
| Human review | full queue + history | none | **Full queue with value history** |
| Audit | audit tables | queries table | **Full audit + query log** |
| Evaluation | multi-metric harness | 10 golden Q, 8/10 pass | **Both:** field-F1 *and* golden questions |
| Infra | Docker, Redis, Celery, MinIO, S3, RBAC, OIDC, OTel | none | **Deferred, then mostly declined** — see Phase 10.6. Compose = Postgres + api + worker; local FS storage |
| Portfolio module | full | holdings CSV | **Minimal** — 2 tables; the killer queries are portfolio-level |

### Where each draft was wrong

**The expensive plan has no cost ceiling.** It logs `token_usage` and `estimated_cost` but never
stops. With Qwen that is survivable; with Opus and GPT-vision it is not. It also assumes whole
documents flow to the frontier model — a 300-page prospectus across 7 clause-type passes is
roughly **$10/document** at `claude-opus-5` rates ($5/MTok in, $25/MTok out). Its nine phases front-load
enterprise scaffolding (Celery, MinIO, OIDC, Prometheus) that serves none of the MVP features.

**The cheap plan cannot do the actual job.** Its "documents" are hand-written Markdown with
`Cross Default Threshold: RM30 million` on its own line — regexes that work there will not survive
one real trust deed. No vector search, no VLM, no PDF. But its *governance* is exactly right, and
its instinct to build the deterministic path first is the correct sequencing.

---

## 2. Cost governance (adopted from the cheap plan, hardened)

This is the backbone. Everything routes through `app/llm/router.py`.

**Four guards, checked before every call:**

| Guard | Env var | Default | Behavior on breach |
|---|---|---|---|
| Per-document ceiling | `MAX_COST_PER_DOCUMENT_USD` | `2.00` | Abort doc, mark `budget_exceeded`, queue for review |
| Global ceiling | `MAX_TOTAL_COST_USD` | `200.00` | Circuit-breaker opens; all calls refused |
| Per-call ceiling | `MAX_COST_PER_CALL_USD` | `0.50` | Reject before dispatch |
| VLM pages per doc | `MAX_VLM_PAGES_PER_DOC` | `40` | Fail loudly; do not silently truncate |

**Three cost reducers:**

1. **Candidate narrowing (largest lever).** Regex + Postgres FTS + pgvector kNN against clause-type
   exemplars reduce a 300-page document to ~30 candidate spans (~40k tokens) before Opus sees anything.
   Estimated: **$10/doc → ~$0.35/doc.**
2. **Prompt caching.** `cache_control: {"type": "ephemeral"}` on the system + JSON schema + few-shot
   prefix. Cache reads bill at 0.1×; writes at 1.25×, so break-even is two calls. The minimum
   cacheable prefix on `claude-opus-5` is **512 tokens** — lower than most models, so our ~3k-token
   extraction prefix caches comfortably. Prefix must be byte-stable (no timestamps, no UUIDs,
   `sort_keys=True` on any serialized JSON).
3. **Response cache.** `llm_cache` keyed on `sha256(prompt_version | model_id | content_hash)`.
   Re-running an unchanged pipeline costs $0.

Plus: **Batch API for backfills** (50% discount, non-interactive), and a `--dry-run` estimator that
prices a document via `count_tokens` before spending anything.

**Verify spend, don't assume it.** `usage.cache_read_input_tokens` must be non-zero across repeated
extractions. If it is zero, a silent cache invalidator is in the prefix — that is a bug, not a
tuning issue.

---

## 3. Two extractors, deliberately

We run the rule-based extractor **and** the Opus extractor on every candidate span. This looks
redundant; it is not:

- The rules give a **free quality signal** — where they disagree with the LLM, that field goes to
  human review with no extra model cost.
- They give a **fallback** when the budget guard trips mid-document.
- They give an **A/B baseline** so "did the LLM actually help?" is measurable rather than assumed.
- They keep the system **partially functional at $0**, which matters for CI and demos.

Disagreement rate is a tracked metric in the eval harness.

---

## 4. Data model (17 tables)

```
documents            sha256, filename, source_type, document_type, issuer_guess,
                     language, page_count, status, parse_confidence, uploaded_by, ts
document_pages       document_id, page_no, char_count, image_area_ratio, has_text_layer,
                     parse_method, vlm_used, vlm_reason, confidence          ← new vs both drafts
document_chunks      document_id, page_no, section_title, chunk_text, chunk_type, language,
                     fts_config, embedding vector(1024), char_start, char_end, hash
instruments          issuer_name, instrument_name, instrument_type, currency, isin,
                     sukuk_structure, issue_size NUMERIC, maturity_date, rating, rating_agency
clauses              document_id, instrument_id, clause_type, clause_text, page_no,
                     source_quote, char_start, char_end, normalized_json, confidence,
                     extraction_status, review_status
covenants            clause_id, instrument_id, covenant_type, summary, conditions_json,
                     thresholds_json, trigger_event, severity, confidence, review_status
call_schedules       instrument_id, call_date, call_price NUMERIC, call_type, source_clause_id
rating_triggers      instrument_id, rating_agency, trigger_rating, trigger_rank INT,
                     trigger_direction, consequence, source_clause_id       ← rank enables ordinal SQL
sukuk_structures     instrument_id, structure_type, spv_name, originator, underlying_asset,
                     profit_rate, purchase_undertaking, dissolution_events_json,
                     shariah_compliance_events_json
portfolios           name, owner, mandate_type, base_currency
portfolio_holdings   portfolio_id, instrument_id, quantity, market_value, nav_weight, as_of_date
extraction_jobs      document_id, job_type, status, provider, model_id, prompt_version,
                     extractor_version, token_usage, estimated_cost, error, started/finished
llm_calls            job_id, stage, provider, model_id, prompt_tokens, completion_tokens,
                     cache_read_tokens, cache_write_tokens, estimated_cost_usd, latency_ms
llm_cache            cache_key, prompt_hash, model_id, response_json, estimated_cost_usd
human_reviews        entity_type, entity_id, field_name, old_value, new_value, source_quote,
                     page_no, confidence, trigger_reason, status, reviewer_id, notes, reviewed_at
audit_logs           actor_type, actor_id, action, entity_type, entity_id, payload_json, request_id
query_logs           user_id, question, intent, retrieved_chunk_ids, tools_called, sql_generated,
                     answer, citations_json, confidence, token_usage, estimated_cost
```

**Two additions neither draft had:** `document_pages` (page-level confidence is what routes VLM
spend, so it needs to be queryable, not a log line) and `rating_triggers.trigger_rank` (an integer
rank so "downgraded below A" is a plain SQL `<=` instead of string comparison).

Indexes: `document_id`, `instrument_id`, `covenant_type`, `clause_type`, `review_status`,
`as_of_date`, `portfolio_id`; HNSW on `document_chunks.embedding`; GIN on the `tsvector` column.

---

## 5. Query agent (LangGraph)

```
classify_intent ─→ plan ─→ retrieve ─→ tools ─→ rules_eval ─→ synthesize ─→ verify ─→ log
                              │                                                │
                              └──── insufficient evidence ──→ refuse ──────────┘
```

**Intents:** `document_search` · `covenant_lookup` · `instrument_lookup` · `portfolio_query` ·
`covenant_breach_check` · `unsupported`

**Tools (all deterministic):** `search_clauses` · `get_instrument` · `get_covenants` ·
`get_call_schedules` · `get_rating_triggers` · `get_portfolio_holdings` · `run_read_only_sql` ·
`evaluate_covenant_rule` · `cite_sources`

**`verify` node** is the part most systems skip: every factual claim in the drafted answer must map
to a `clause_id` that was actually retrieved this turn. Unsupported claims are stripped and the
confidence is reduced. This is the structural guard against a fluent, wrong answer.

**SQL guardrail:** read-only role · `SELECT` only, parsed via `sqlglot` (not regex) · table+column
allowlist · `statement_timeout=5s` · forced `LIMIT 1000` · every generated statement logged.
Prefer parameterized templates over free-form SQL; free-form is the fallback, not the default.

---

## 6. Phases

Sequencing principle taken from the cheap plan: **build the deterministic path end-to-end first.**
Phases 1–4 spend **$0 on LLM APIs** and still produce a working, queryable, demoable system. That
gives a baseline to measure LLM lift against, and means a budget bug in Phase 5 can't take down a
system that already works.

Phases 0–9 are **built**. Phase 10 is not. Each heading below carries its state so a
reader can stop guessing from commit history.

### Phase 0 — Plan *(this document)* ✅
Deliverables: `CLAUDE.md`, `PLAN.md`.

### Phase 1 — Scaffold ✅ *($0)*
`pyproject.toml`, Dockerfile, compose (postgres+pgvector, api, worker), Makefile, `.env.example`,
ruff/mypy/pytest config, settings loader, structured logging + `request_id` middleware, FastAPI
skeleton, `/health` + `/ready`.
**Accept:** `make install lint type test` pass · `make up` boots · `GET /health` → 200.

### Phase 2 — Database & domain ✅ *($0)*
SQLAlchemy models, Pydantic schemas, enums, Alembic migration, repositories, pgvector + GIN indexes,
seed script (synthetic portfolios + instruments).
**Accept:** migrations apply clean from scratch · repository CRUD tests pass · seed produces a demo
portfolio.

### Phase 3 — Ingestion ✅ *($0)*
Upload API, SHA256 dedup, local FS storage adapter (S3-shaped interface), PyMuPDF text + pdfplumber
tables, **page-confidence scoring**, chunking with span offsets, job tracking.
**Accept:** duplicate detected by hash · text+tables extracted from synthetic PDF · every chunk
carries `(page, char_start, char_end)` · low-confidence pages correctly flagged (VLM not yet wired).

### Phase 4 — Search & rules ✅ *($0 — first demoable milestone)*
Embedding adapter interface + **fake deterministic embedder**, hybrid retrieval (pgvector + tsvector,
RRF fusion), rule-based regex extractor, **rules engine** (ordinal rating comparison, threshold
evaluation, date windows), CLI query over the deterministic path.
**Accept:** hybrid search beats either leg alone on the golden set · rules engine unit tests cover
every covenant type · ≥6/10 golden questions answerable with **zero LLM calls**.

### Phase 5 — LLM layer ✅ *(first spend — guards land before adapters)*
`app/llm/`: budget guard, response cache, cost tracker, mock provider — **written and tested first**.
Then adapters: Anthropic (`claude-opus-5`), OpenAI vision, Qwen (chat + embeddings). Real Qwen
embeddings replace the fake embedder; VLM fallback wired to page confidence.
**Accept:** budget guard provably blocks an over-budget call (unit test) · cache hit costs $0 ·
mock provider drives the whole pipeline in CI · **`make test` makes zero paid API calls.**

### Phase 6 — Extraction pipeline ✅
Versioned Jinja2 prompts, candidate detection, Opus structured extraction with prompt-cached prefix,
Pydantic validation + one feedback retry, **citation verification**, rules/LLM disagreement
detection, review-queue routing, per-document cost attribution.
**Accept:** golden synthetic covenant extracted correctly · invalid JSON triggers exactly one retry
then review · unverifiable citation is never persisted · measured cost/document logged and under cap.

### Phase 7 — Query agent, review & audit ✅
LangGraph graph, tools, SQL guardrail, citation formatting, verify node, query logging.
Review queue API (approve/reject/edit with value history), audit log, audit read endpoint.
**Accept:** ≥8/10 golden questions answered with correct citations · agent refuses when evidence is
absent · non-`SELECT` SQL rejected (test) · a correction preserves prior value + reviewer + reason ·
every mutation appears in `audit_logs`.

### Phase 8 — Evaluation, hardening & deferred infra ✅
Eval harness (field-level F1, enum exact match, numeric tolerance, date tolerance, citation
precision/recall, answer faithfulness, refusal correctness, **rules-vs-LLM agreement**, cost/doc).
GitHub Actions (lint, type, test — no paid calls). Then, only now: Celery/Redis, S3/MinIO, RBAC/OIDC,
OTel/Prometheus, deployment docs, runbook.
**Accept:** `make eval` emits metrics to `evals/results/` · CI green · docs cover deploy + operate.

**Delivered except the deferred infra**, which Phase 10 resolves by decision rather than
by building most of it.

### Phase 9 — HTTP surface, authentication, UI ✅
Not in the original plan; added once the eval harness made it obvious the system had no
human-facing surface at all. `POST /query` and the catalogue read endpoints, session
auth with two roles, and the four-screen server-rendered UI.

Also fixed here: `get_session` never committed, so every review decision returned 200 and
persisted nothing. Found by running the endpoint against the real database — the rolled-back
test session had hidden it from the whole suite.
**Accept (met):** a reviewer clears a queue item end to end without the CLI · every covenant
on screen links to its highlighted source span · anonymous requests are refused.

### Phase 10 — Hardening and accuracy *(in progress)*
The remaining work. Phases 1–9 built the machine; this is about whether it is *right* and
whether it is *safe*, neither of which the current numbers answer.

Findings and their evidence are in **[docs/review.md](docs/review.md)** — an audit taken at
`dc30321` against the running stack and three real 200–535 page prospectuses. This section is
the plan; that document is the reasoning behind it. Items 7–10 below came out of it.

1. **A real prospectus, and a re-baseline.** §9 Q1, still open, and the largest source of
   schedule risk. `make eval` reports F1 0.95 (LLM) / 0.99 (rules) against documents we
   generated ourselves — that measures the harness, not the extractor. Regex patterns,
   chunking and candidate detection are all tuned to invented layouts. Nothing licensed
   may be committed (CLAUDE.md 7); keep it under `var/`.
2. **`rating_agency` extraction.** The one weak field: P 0.50 / R 0.50 on the LLM path,
   R 0.50 on rules, against ≥0.94 everywhere else. Both extractors miss the same label,
   which points at normalisation rather than at either model — start with the `(m)` / `id`
   national-scale suffixes in `app/rules/ratings.py`.
3. **Live-verify the VLM.** Wired (`opuscovintel ocr`) and never once run against a real
   provider; blocked on OpenAI credit. Closes §9 Q3.
4. **Real embeddings and the semantic candidate legs.** Everything runs on
   `HashingEmbedder`, so the vector leg of hybrid retrieval is noise and the FTS/kNN
   candidate legs default off. Close §9 Q2 **before** indexing: 1024 dims is baked into the
   schema, and changing it means re-embedding the corpus and rebuilding the HNSW index.
5. **Refuse more.** The agent answers some unsupported questions at 0.95 confidence instead
   of refusing — the intent classifier routes them to `instrument_lookup`, which answers
   from structured rows without needing retrieval, so the refusal path is never reached.
   The golden set misses this because its one unanswerable question takes the other path.
6. **Decide the deferred infra, mostly by declining it.** Celery/Redis and MinIO buy nothing
   at this volume — the worker's `FOR UPDATE ... SKIP LOCKED` poll is correct and simpler,
   and the local store already implements an S3-shaped interface. OIDC was superseded by
   Phase 9's session auth. Keep OTel/Prometheus, cheaply. Record the decision here rather
   than leaving four unbuilt items looking like debt.
7. ~~**Move the boundary for the six operational tables.**~~ ✅ `opuscovintel_ro` held `SELECT`
   on `audit_logs`, `human_reviews`, `query_logs`, `llm_calls`, `llm_cache` and
   `extraction_jobs`; `sql_guard.py` excluded all six from the allowlist and said plainly why,
   and the grant never followed. Revoked in
   `20260810_0733_revoke_operational_tables_from_readonly.py`, with a test that connects as the
   role, gets `permission denied`, and pins the role's readable set to the allowlist so a future
   table cannot inherit `SELECT` from the init script's default privileges.
8. **Raise the cost cap, and fail before spending rather than during.** All three real
   prospectuses exceed `MAX_COST_PER_DOCUMENT_USD=2.00` — worst case $20.94, $11.48 and
   $4.28 — so each would abort mid-document, paying for the calls made and leaving a
   partial extraction. Raise the default to something calibrated for 500-page documents,
   and make the guard refuse a document whose dry-run ceiling already exceeds the cap.
   Then build the **Batch API** path §2 specifies and nothing implements: 50% off, and
   backfilling a corpus is exactly its workload.
9. **Close the auth gaps.** ✅ *Rate limiting* — `login_attempts` plus exponential backoff per
   username and per client address (`app/auth/rate_limit.py`), enforced inside
   `AuthService.authenticate` so both login paths inherit it; backoff rather than lockout, so
   nobody needs an operator to get back in. Still open: no password minimum length, and no
   security response headers. A CSP matters more here than usual because the UI renders clause
   text lifted verbatim out of third-party PDFs — autoescaping is on and tested, and CSP is the
   layer that holds when an escaping bug slips through.
10. **Batch the portfolio page's rule evaluation.** It calls `evaluate_covenant_rule` once per
    holding, each issuing several queries — fine for two positions, hundreds of queries for a
    realistic 200-bond portfolio. Reusing the agent's tool was right; a second rules
    implementation would eventually disagree with the first. Batch the loading, not the logic.

**Accept:** one real document ingests, extracts, and has its numbers written down next to the
synthetic baseline, however bad they are · `rating_agency` F1 ≥0.9 on both methods · one page
OCR'd live with its cost in the ledger · retrieval measured against the hashing baseline ·
the read-only role is denied on all six operational tables · a document over the cost cap is
refused before the first call, not during.

---

## 7. Explicitly out of scope for MVP

Autonomous trading or compliance decisions · real-time market data · generic chatbot ·
multi-tenancy · full OIDC/SSO · Snowflake/BigQuery sync · mobile · non-Malaysian regulatory logic ·
production HA/DR.

---

## 8. Acceptance criteria for "MVP done"

1. A synthetic PDF prospectus ingests, chunks, and embeds.
2. A scanned page routes to the VLM and only that page does.
3. Covenants, call schedules, rating triggers, and sukuk structures extract into structured rows,
   each with a verified page + verbatim quote.
4. Low-confidence and disagreeing extractions land in the review queue; a correction preserves history.
5. The LangGraph agent answers ≥8/10 golden questions with citations and refuses the unanswerable one.
6. No answer contains a claim not traceable to a retrieved clause.
7. `make eval` reports extraction F1, citation recall, and faithfulness.
8. Total LLM spend per document is logged, attributed by stage, and under `MAX_COST_PER_DOCUMENT_USD`.
9. `make test` completes with zero paid API calls.

---

## 9. Open questions

1. **Real documents.** Everything below assumes synthetic fixtures. When do licensed prospectuses
   become available? Regex and chunking heuristics will need retuning against real layouts, and
   that is the single biggest source of schedule risk.
2. **Qwen embedding endpoint.** DashScope international vs. mainland endpoint — which region, and is
   `text-embedding-v4` at 1024 dims the right dimensionality? (Changing dims later means re-embedding
   the whole corpus and an index rebuild.)
3. **GPT vision model ID.** Env-configured; needs a current, verified ID and its per-image token cost
   before the VLM cap can be tuned sensibly.
4. **Portfolio holdings source.** CSV upload for MVP, or a read-only feed from the existing PMS?
5. ~~**Reviewer identity.**~~ **Closed 2026-08-07.** Placeholder reviewer IDs were not acceptable:
   the value came from the request body, so the audit trail recorded whatever the caller typed.
   Phase 9 replaced it with session-derived identity and two roles. OIDC stays deferred.
6. **Global budget.** Is `MAX_TOTAL_COST_USD=200` the right ceiling for the build phase?
7. **Bahasa Malaysia volume.** What share of the real corpus is BM? If material, the extraction
   prompts need BM few-shot examples and the golden set needs BM questions.

---

## 10. Assumptions

- Single tenant, single asset manager, internal users only.
- Documents are PDFs, ≤600 pages, ≤100MB.
- Postgres 16+ with the `vector` extension available.
- API keys for Anthropic, OpenAI, and Qwen/DashScope are provisioned and funded.
- Answers are decision support, not investment advice; a human reviews before any action.
