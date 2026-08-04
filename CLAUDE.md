# CLAUDE.md — OpusCovIntel

Sukuk & bond **covenant intelligence** platform for a Malaysian fixed-income asset manager.
Ingests prospectuses, trust deeds, rating reports and announcements; extracts covenants into
structured, cited records; answers portfolio-level questions through a governed LangGraph agent.

**Read this file before writing code. It encodes decisions that are expensive to reverse.**

---

## 1. Non-negotiable invariants

These are the rules that make the system audit-defensible. Violating any one of them is a bug,
even if tests pass.

1. **The LLM never computes a breach.** LLMs extract values and explain. A deterministic Python
   rules engine (`app/rules/`) evaluates trigger conditions; SQL aggregates portfolio exposure.
   If you find yourself asking a model "which holdings breach X", stop — that is a rules-engine
   call with an LLM-authored *summary* on top.
2. **Every fact is traceable to a span.** `Covenant → Clause → DocumentChunk → (page, char_start, char_end)`.
   A covenant row that cannot name its source page and verbatim quote is invalid and must not be persisted.
3. **Citations are verified, not trusted.** The model returns `source_quote`; we assert that quote
   actually occurs in the cited chunk (normalized exact match, then ≥0.92 rapidfuzz ratio).
   Failure → `extraction_status='citation_failed'` → human review. Never persist an unverified quote.
4. **No silent LLM spend.** Every model call goes through `app/llm/router.py`, which enforces the
   budget guard and cache. Direct `anthropic.Anthropic()` / `openai.OpenAI()` calls outside
   `app/llm/adapters/` are forbidden.
5. **Refusal to answer is a correct outcome.** If retrieval returns no supporting clause, the agent
   answers "No supporting evidence in the corpus" with `confidence: 0.0`. Never let the model fill gaps.
6. **The query agent uses a read-only DB role.** `DATABASE_URL_RO` with `SELECT`-only grants.
   Generated SQL is allowlist-validated, `LIMIT`-capped, and statement-timeout-bounded.
7. **Idempotent by extraction identity.** `(document_sha256, prompt_version, model_id, extractor_version)`
   is the uniqueness key. Re-running an unchanged pipeline is a no-op and costs $0.

---

## 2. Model routing

Configured in `configs/models.yaml`, never hardcoded. Model IDs live in env vars — **verify current
IDs against provider docs before changing defaults**.

| Stage | Provider / default | Rationale |
|---|---|---|
| Document classification, section detection | Qwen (`qwen-plus`) | High volume, low stakes, cheap |
| Candidate clause detection | rules + pgvector + FTS (no LLM) | Free; narrows 300pp → ~30 spans |
| **Covenant / legal structured extraction** | **`claude-opus-5`**, `effort: high` | Highest stakes; long-context legal reasoning |
| Scanned / low-confidence page OCR | GPT vision model (`OPENAI_VLM_MODEL`) | Only pages that fail the text-layer check |
| Embeddings | Qwen `text-embedding-v4`, **1024 dims** | Strong multilingual (EN + Bahasa Malaysia) |
| Answer synthesis | `claude-opus-5`, `effort: medium` | Citation discipline, refusal calibration |
| Eval judge (faithfulness) | `claude-opus-5`, separate prompt version | Must not share prompt with generator |

### Anthropic API rules (verified against current API)

- **Model ID is `claude-opus-5`** — no date suffix. $5/MTok in, $25/MTok out. 1M context, 128K max output.
- **`temperature`, `top_p`, `top_k` are rejected with a 400.** Do not set them. Steer via prompt.
- **`thinking: {"type": "enabled", "budget_tokens": N}` is rejected with a 400.** Use
  `thinking: {"type": "adaptive"}` (on by default) and control depth with `output_config: {"effort": ...}`.
- **Assistant-turn prefills return a 400.** Use structured outputs instead.
- **`max_tokens` caps thinking + response together.** Extraction calls must budget ≥8000.
- ⚠️ **`citations: {enabled: true}` is incompatible with `output_config.format` (400).**
  This is why we carry citations as schema fields and verify them ourselves. Do not "fix" this by
  dropping structured outputs.

### Cost levers, in order of impact

1. **Candidate narrowing** — never send a whole document to Opus. ~20× saving.
2. **Prompt caching** — `cache_control: {"type": "ephemeral"}` on the system+schema+few-shot prefix.
   Reads cost 0.1×. Minimum cacheable prefix on `claude-opus-5` is **512 tokens** (lower than most
   models — short prompts *do* cache here). Keep the prefix byte-stable: no timestamps, no UUIDs,
   `json.dumps(..., sort_keys=True)`.
3. **Batch API** — 50% off for non-interactive bulk re-extraction. Use for backfills.
4. **Response cache** — keyed on `sha256(prompt_version | model_id | content)`.

---

## 3. Architecture

```
app/
  api/        FastAPI routers only. No business logic, no SQL.
  core/       settings (pydantic-settings), structured logging, request_id middleware
  domain/     Pydantic v2 schemas + enums. Pure — imports nothing from db/ or llm/
  db/         SQLAlchemy 2.x models, Alembic, repositories
  ingest/     PyMuPDF/pdfplumber, page-confidence scoring, VLM fallback, chunking
  llm/        adapters/ (anthropic, openai, qwen) + router.py + budget.py + cache.py
  extract/    prompts/*.jinja2, candidate detection, extractors, validation loop
  rules/      deterministic covenant evaluation. Pure functions. Heavily unit-tested.
  agent/      LangGraph query graph + tools + SQL guardrail
  review/     human review queue service
  evals/      golden set + metrics harness
```

**Dependency direction is one-way:** `api → services → repositories → models`.
`domain/` and `rules/` are leaves and import nothing from the layers above them.

---

## 4. Ingestion & VLM fallback

Page-level confidence decides routing. A page goes to the VLM **only** if it fails these checks:

```python
needs_vlm = (
    page.text_char_count < 120
    or page.image_area_ratio > 0.70
    or page.text_layer is None
    or (page.has_table_hint and pdfplumber_extraction_failed)
    or page.garbled_unicode_ratio > 0.25   # cid: mojibake, common in old scans
)
```

Log the *reason* on `document_pages.vlm_reason`. Cap VLM pages per document via
`MAX_VLM_PAGES_PER_DOC` — a 400-page scan that trips every check must fail loudly, not quietly
spend $80.

## 5. Extraction loop

```
chunk → candidate detection (regex + FTS + kNN vs clause-type exemplars)
      → Opus structured extraction (Pydantic schema, prompt-cached prefix)
      → Pydantic validation
      → on failure: ONE retry with the validation error appended
      → still failing → human review queue (never a silent drop)
      → citation verification against chunk text
      → rule-based extractor runs in parallel; DISAGREEMENT is a free review trigger
```

Every extracted field carries `confidence`, `method` (`rule` | `llm` | `vlm`), and `source_chunk_id`.

**Review queue triggers:** `confidence < 0.85` · rules/LLM disagree · validation retried ·
citation verification failed · value sourced from a VLM page · any monetary threshold > RM100m.

---

## 6. Malaysian domain notes

- Currency is **MYR**; parse `RM30 million` / `RM30m` / `RM30,000,000` → `Decimal("30000000")`.
  Use `Decimal` for money — never `float`.
- Rating agencies: **MARC**, **RAM** (plus S&P/Moody's/Fitch for cross-border). Malaysian national
  scale ratings carry `(m)` / `id` suffixes — normalize but preserve the raw string.
- Rating comparison is **ordinal, not lexical**: `AA-` > `A+`. Use the rank table in
  `app/rules/ratings.py`. Never string-compare ratings.
- Sukuk structures: Ijarah, Wakalah, Musharakah, Mudharabah, Murabahah, Istisna'.
- Shariah non-compliance is typically a **dissolution event** triggering a **purchase undertaking**
  — model these as distinct linked entities, not one free-text field.
- Documents mix English and **Bahasa Malaysia**. Postgres FTS has no Malay stemmer: use the
  `english` config for EN chunks and `simple` for BM chunks, stored per-chunk in `chunks.fts_config`.

---

## 7. Conventions

- Python 3.12, full type hints, `ruff`, `mypy --strict` on `domain/` and `rules/`.
- UUIDv7 primary keys. All timestamps `TIMESTAMPTZ`, UTC.
- Enums for every controlled vocabulary — no bare strings for `clause_type`, `covenant_type`, etc.
- `logging` only, JSON-structured, with `request_id`. No `print`.
- Tests use the mock LLM provider. **CI must never hit a paid API.** A real-provider test requires
  `RUN_LIVE_LLM_TESTS=1` and is excluded from the default `make test`.
- Synthetic documents only in fixtures. Real prospectuses are copyrighted — do not commit them.
- Secrets from env only; `.env` is gitignored, `.env.example` is committed.

## 8. Commands

```bash
make up          # docker compose: postgres+pgvector, api, worker
make migrate     # alembic upgrade head
make lint type test
make ingest-sample && make extract-sample && make query-sample
make eval        # golden-set metrics → evals/results/
make cost-report # spend by document, stage, provider
```

---

## 9. Do not

- Do not call an LLM provider SDK outside `app/llm/adapters/`.
- Do not put business logic in a route handler.
- Do not let the agent execute non-`SELECT` SQL, or SQL touching a non-allowlisted table.
- Do not persist an extraction whose citation failed verification.
- Do not add Celery, Redis, MinIO, RBAC, or OIDC before Phase 8 — they are explicitly deferred.
- Do not send a full document to Opus "just to see if it works". That is a $10 keystroke.
