# OpusCovIntel

Sukuk & bond **covenant intelligence** for a Malaysian fixed-income asset manager.

Ingests prospectuses, trust deeds, rating reports and announcements; extracts covenants into
structured, **cited** records; answers portfolio-level questions through a governed LangGraph agent
that refuses to answer without evidence.

> **Status: Phase 2 (database & domain) complete.** 17 tables, migrations, repositories and
> synthetic seed data are in. Ingestion lands in Phase 3; the API still serves only
> health/readiness. See [PLAN.md](PLAN.md) for the phase plan and [CLAUDE.md](CLAUDE.md) for
> architectural invariants.

## Quick start

```bash
make install     # creates .venv (Python 3.12) and .env
make up          # postgres+pgvector, api, worker
make migrate     # apply schema
make seed        # synthetic instruments, portfolios, holdings (idempotent)
make check       # lint + type + test
```

Tests run against a dedicated `opuscovintel_test` database, created on first use, so they never
depend on — or disturb — your development data. Override with `TEST_DATABASE_URL`.

Without Docker, run against your own Postgres:

```bash
make run         # uvicorn with autoreload on :8000
```

## Layout

| Path | Purpose |
|---|---|
| `app/api/` | FastAPI routers. No business logic, no SQL. |
| `app/core/` | Settings, structured logging, request-id middleware. |
| `app/domain/` | Pydantic schemas + enums. Pure leaf — imports nothing from `db/` or `llm/`. |
| `app/db/` | SQLAlchemy models, repositories, dual (RW / RO) engines. |
| `app/ingest/` | PDF parsing, page confidence, VLM fallback, chunking. *(Phase 3)* |
| `app/llm/` | Provider adapters behind a budget-guarded router. *(Phase 5)* |
| `app/extract/` | Prompts, candidate detection, validation loop. *(Phase 6)* |
| `app/rules/` | Deterministic covenant evaluation. Pure functions. *(Phase 4)* |
| `app/agent/` | LangGraph query graph + SQL guardrail. *(Phase 7)* |
| `app/evals/` | Golden set and metrics harness. *(Phase 8)* |

## The four things that make this auditable

1. **The LLM never computes a breach.** Models extract and explain; a deterministic rules engine
   evaluates triggers and SQL aggregates exposure.
2. **Every fact traces to a span** — covenant → clause → chunk → (page, char offsets).
3. **Citations are verified, not trusted.** A quote that isn't found in its cited chunk goes to
   human review instead of into the database.
4. **The query agent holds a read-only database role.** Enforced by Postgres grants, not by
   convention.

## Cost control

LLM spend is capped, not merely logged. Every model call passes through a budget guard with
per-call, per-document, and global ceilings; responses are cached by content hash, and frontier
models only ever see pre-narrowed candidate spans rather than whole documents. See
[PLAN.md §2](PLAN.md).

`make test` makes **zero** paid API calls. Billable tests are marked `live_llm` and skipped unless
`RUN_LIVE_LLM_TESTS=1`.

## Data disclaimer

Synthetic fixtures only — no real prospectuses are committed. Output is **decision support, not
investment or legal advice**, and is intended for human review before any action.
