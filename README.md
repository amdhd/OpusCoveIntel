# OpusCovIntel

[![CI](https://github.com/amdhd/OpusCoveIntel/actions/workflows/ci.yml/badge.svg)](https://github.com/amdhd/OpusCoveIntel/actions/workflows/ci.yml)

Sukuk & bond **covenant intelligence** for a Malaysian fixed-income asset manager.

Ingests prospectuses, trust deeds, rating reports and announcements; extracts covenants into
structured, **cited** records; answers portfolio-level questions through a governed LangGraph agent
that refuses to answer without evidence.

> **Status: Phase 3 (ingestion) complete.** PDFs upload, deduplicate by content hash, parse to
> per-page confidence telemetry, and chunk into spans that can be cited. Retrieval and the rules
> engine land in Phase 4; the first LLM spend is Phase 5. See [PLAN.md](PLAN.md) for the phase
> plan and [CLAUDE.md](CLAUDE.md) for architectural invariants.

## Quick start

```bash
make install     # creates .venv (Python 3.12) and .env
make up          # postgres+pgvector, api, worker
make migrate     # apply schema
make seed        # synthetic instruments, portfolios, holdings (idempotent)
make check       # lint + type + test
```

## Ingesting a document

```bash
make ingest-sample   # generate a synthetic prospectus, then parse and chunk it
```

Or over HTTP — `POST` returns 201 for new bytes and 200 with `duplicate: true` for bytes already
known, because a duplicate is a correct outcome rather than an error:

```bash
curl -F "file=@var/sample-prospectus.pdf" localhost:8000/documents/upload
```

Uploading queues the document; the worker claims it (`extraction_jobs.status = 'queued'` is the
queue — no broker until Phase 8) and writes pages and chunks. `GET /documents/{id}/pages` shows
per-page parse confidence and, for a page that failed the text-layer checks, which check tripped;
`GET /documents/{id}/chunks` shows every chunk with the character span it was cut from. Phase 3
*detects* pages that will need the vision model but never calls one — ingestion costs $0.

Tests run against a dedicated `opuscovintel_test` database, created on first use, so they never
depend on — or disturb — your development data. Override with `TEST_DATABASE_URL`. Database-backed
tests skip when Postgres is unreachable, which is convenient locally and dangerous in CI, so CI
sets `REQUIRE_POSTGRES=1` to turn that skip into a failure.

CI runs `make check` plus a migration round-trip (`upgrade` → `downgrade base` → `upgrade`, then a
drift check) and a container build that asserts `/health` answers 200 while `/ready` answers 503
with no database reachable. No provider credentials are available to any job, and the workflow
fails if one is ever added.

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
| `app/ingest/` | Object storage, PDF parsing, page confidence, chunking. |
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
