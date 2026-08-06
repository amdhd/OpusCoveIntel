# Handover — after Phase 8, before the UI

Written 2026-08-07 at commit `4f27e0d` (`main`, clean). Supersedes the Phase 8
handover retired in `5a153b5`. **Delete this file when items 1–4 have landed** —
same convention as `5929c39` and `5a153b5`. Do not let it rot into a second
roadmap: [PLAN.md](PLAN.md) is the roadmap, this is only what a fresh session
cannot infer from the code.

Read [CLAUDE.md](CLAUDE.md) first. Its §2 routing table has a `Status` column
that says what is actually wired versus merely designed; trust that column over
any prose, including this file's.

---

## 1. Where the project is

Phases 1–8 are built and the deterministic path is verified end to end.

| | State |
|---|---|
| Tests | **673** collected; lint and `mypy --strict` clean |
| CI | green on `main`, `REQUIRE_POSTGRES=1`, zero paid API calls |
| Golden questions | **10/10 on both paths**, faithfulness 1.00, refusal F1 1.00 |
| Extraction F1 (micro) | LLM **0.95** · rules **0.99**, agreement 0.88 |
| Cost | $0.60 over 39 calls, $0.20/document, prompt cache confirmed (221k cached read tokens) |
| VLM | wired (`opuscovintel ocr`), still **never run live** — no OpenAI credit |
| Embeddings | **`HashingEmbedder`** — no `QWEN_API_KEY`, so no real vectors anywhere |
| HTTP surface | health, documents, review, audit. **No query endpoint. No UI.** |

Two commands spend money, both require an explicit target and both have
`--dry-run`: `opuscovintel extract` and `opuscovintel ocr`. Everything else,
including `make check` and `make eval`, is $0 by construction.

**Resuming takes about two minutes:**

```bash
open -a Docker && make up && make migrate && make seed && make check
```

## 2. What is genuinely unfinished

Two things sat inside Phase 8 in PLAN.md §6 and were never built:

- **The deferred infra** — Celery/Redis, S3/MinIO, RBAC/OIDC, OTel/Prometheus.
  `grep` finds none of it in `pyproject.toml`; [docker-compose.yml:3](docker-compose.yml)
  still calls it deferred. Most of it should stay declined (item 10). The
  exception is auth, which is now load-bearing (item 3).
- **Four rows of the routing table** that read *not built*: cheap-model
  classification, answer synthesis, the eval judge, and real Qwen embeddings.

Neither is a defect. Both are decisions that were postponed and are now due.

---

## 3. The ten things to do, in priority order

Items 1–4 are one chain: each unblocks the next, and together they are the
missing human-facing half of the product. Items 5–8 are quality and coverage
gaps that can proceed in parallel. Items 9–10 are bookkeeping.

### 1. `POST /query` — give the agent an HTTP surface

The agent is reachable only from the CLI. [app/main.py:51](app/main.py) mounts
four routers and none of them can ask a question. Wrap
[app/agent/service.py](app/agent/service.py)'s `open_agent_query_service()` and
return `AgentAnswer` — answer, citations, confidence, `refused`, `tools_used`,
`chunk_ids`. The refusal case is part of the contract, not an error: a refusal
is HTTP 200 with `refused: true` and `confidence: 0.0` (CLAUDE.md §1.5).

Keep the two-session split intact — the service already handles it, so the route
must not open its own session.

**Accept:** a refusal and an answered question both round-trip with citations ·
`query_logs` and `audit_logs` gain a row per request · no business logic in the
router (CLAUDE.md §9).

### 2. Read endpoints for covenants, instruments and portfolio

A UI cannot render a single covenant today. The repositories already exist in
`app/db/repositories/` — these are thin routers over them, not new SQL.

Minimum set: list/get covenants by instrument and by type · get instrument with
its call schedules, rating triggers and sukuk structure · portfolio holdings with
exposure · a clause endpoint returning `(page, char_start, char_end)` and the
chunk text, which is what makes item 4's provenance viewer possible.

**Accept:** every covenant response carries its source page and verbatim quote —
a row that cannot is invalid by CLAUDE.md §1.2 and must not be serialised.

### 3. Auth and real reviewer identity — before any of this leaves localhost

Today the review and audit endpoints are unauthenticated; that was survivable
while the only client was a local CLI. Items 1–2 change that. Separately,
PLAN.md §9 Q5 asked whether placeholder reviewer IDs were acceptable and the
question was never closed — every `human_reviews.reviewer_id` is a placeholder,
which weakens exactly the audit trail the project exists to produce.

Do **not** reach for full OIDC. Start with session auth plus two roles (analyst,
reviewer) and a real user table; OIDC can come later behind the same interface.

**Accept:** an unauthenticated request to review or audit gets 401 · an approve
records a real reviewer · a correction still preserves prior value, reviewer and
reason.

### 4. The UI — four screens

Nothing exists: no HTML, no `package.json`, no templates anywhere in the tree.

**Server-rendered Jinja + HTMX over the existing FastAPI app, not a React SPA.**
Single-tenant internal tool, no build step, no second language, no second
deployment, and it keeps CLAUDE.md §3's "routers only, no business logic" intact.
An SPA buys nothing here and costs a whole toolchain.

Four screens carry the product:

1. **Ask** — question in, answer with inline citations out, refusal rendered as a
   first-class result rather than an error state.
2. **Document and clause viewer** — jump from a citation to `(page, char_start,
   char_end)` and highlight the span. This is the screen worth real effort; it is
   what makes the provenance chain visible rather than merely true.
3. **Review queue** — the one that turns CLAUDE.md §5's triggers into something a
   human can actually work. Approve, correct, reject, with value history.
4. **Portfolio breach board** — rules-engine output over holdings.

**Accept:** a reviewer can clear a queue item end to end without touching the
CLI · every displayed covenant links to its highlighted source span.

### 5. Fix `rating_agency` extraction — the one red cell in the eval

Latest report: `rating_agency` scores **P 0.50 / R 0.50 / F1 0.50** on the LLM
path and **R 0.50** on the rules path. Every other field is ≥0.94. Both extractors
miss the same label, which points at the fixture or the normalisation rather than
at either model.

Start at [app/rules/ratings.py](app/rules/ratings.py) and the `rating_agency`
handling added in `67480ea`. Malaysian national-scale suffixes (`(m)`, `id`) are
the usual culprit — CLAUDE.md §6 requires normalising them while preserving the
raw string.

**Accept:** `rating_agency` F1 ≥0.9 on both methods · a regression test that you
have watched fail with the fix disabled (CLAUDE.md §7).

### 6. Get one real prospectus and re-baseline

The single biggest source of schedule risk, flagged as PLAN.md §9 Q1 and never
resolved. The eval report says it plainly: synthetic fixtures are a regression
baseline, **not** a production accuracy estimate. An F1 of 0.99 on documents we
generated ourselves says the harness works, not that the extractor does.

Regex patterns, chunking heuristics and candidate detection are all tuned against
layouts we invented. Expect them to need retuning, and expect that to be the
longest pole in the project. Nothing may be committed to the repo (CLAUDE.md §7)
— keep it under `var/`, which is gitignored.

**Accept:** one real document ingests, extracts, and has its numbers written down
next to the synthetic baseline, however bad they are.

### 7. Live-verify the VLM path

`VlmService` and `opuscovintel ocr` are wired but have never made a real call —
the OpenAI account had no credit. CLAUDE.md §7 is explicit that wiring something
up is not finishing it, and this exact path already produced one defect (a wrong
storage key) that only surfaced by running it.

`make ocr-dry-run` first, then one scanned page, then check `document_pages.vlm_reason`
and the chunks it produced. Confirm `MAX_VLM_PAGES_PER_DOC` fails loudly rather
than truncating.

**Accept:** one page OCR'd live, its chunks queryable, cost in the ledger, and
PLAN.md §9 Q3 (the vision model ID and its per-image cost) answered.

### 8. Turn on real embeddings and the semantic candidate legs

Without `QWEN_API_KEY` everything falls back to `HashingEmbedder`, so hybrid
retrieval currently has a vector leg made of noise, and the FTS and kNN candidate
legs added in `73198e1` default off. The whole "hybrid beats either leg alone"
claim from Phase 4 is untested with real vectors.

Close PLAN.md §9 Q2 first — endpoint region and dimensionality. **1024 dims is
baked into the schema; changing it later means re-embedding the corpus and
rebuilding the HNSW index.** Decide once.

**Accept:** real embeddings indexed · candidate legs enabled and measured ·
retrieval quality compared against the hashing baseline on the golden set.

### 9. Refresh PLAN.md and README

[PLAN.md:3](PLAN.md) still reads *"Status: Phase 0 (planning). No code written
yet."* — four phases stale, and it is the second file anyone opens. Update the
status line, mark Phases 1–8 done, and add a Phase 9 section covering items 1–4.
About thirty minutes, and it can be done at any point.

### 10. Triage the deferred infra — mostly by declining it

PLAN.md §6 lists Celery/Redis, S3/MinIO, RBAC/OIDC and OTel/Prometheus as Phase 8
work. Do not build it on autopilot. My reading:

- **Celery + Redis — decline.** [app/worker/main.py](app/worker/main.py) polls
  `extraction_jobs.status` with `FOR UPDATE ... SKIP LOCKED`. That is correct,
  simple, and loses nothing at this volume. Revisit when job volume justifies a
  broker, not before.
- **MinIO — decline.** The local-FS store already implements an S3-shaped
  interface; swap the adapter when there is a real deployment.
- **OIDC — folded into item 3**, scoped down to session auth and roles.
- **OTel/Prometheus — keep, cheaply.** Structured logs with `request_id` exist;
  the missing piece is cost and latency per stage, which `make cost-report`
  half-answers already.

Record the decision in PLAN.md rather than leaving four unbuilt items looking
like debt.

---

## 4. Two rules that have paid for themselves repeatedly

Both are in CLAUDE.md §7 now, so this is a pointer, not a copy — but they are the
reason the numbers in §1 are trustworthy, and every item above is subject to them.

**Run it, do not only read it.** Six defects in this codebase were found by
executing the path and by nothing else — not review, not `mypy --strict`, not a
green suite. Items 1, 4, 7 and 8 are all "wired but never run" shaped, which is
precisely the shape that has failed before.

**Prove a regression test fails before trusting that it passes.** Disable the
fix, watch it go red, restore. Verify by reading the diff of the disabled state,
not by assuming the edit landed — a formatter once silently defeated exactly this
check and three tests passed against the bug they were written to catch. This
applies directly to item 5.
