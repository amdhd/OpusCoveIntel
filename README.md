# OpusCovIntel

[![CI](https://github.com/amdhd/OpusCoveIntel/actions/workflows/ci.yml/badge.svg)](https://github.com/amdhd/OpusCoveIntel/actions/workflows/ci.yml)

Sukuk & bond **covenant intelligence** for a Malaysian fixed-income asset manager.

Ingests prospectuses, trust deeds, rating reports and announcements; extracts covenants into
structured, **cited** records; answers portfolio-level questions through a governed LangGraph agent
that refuses to answer without evidence.

> **Status: Phases 1–8 built; LLM extraction verified against the live API.** Documents ingest,
> index, extract into cited covenants, and answer questions through hybrid retrieval, a
> deterministic rules engine and the LangGraph agent. The golden set passes **10/10 with zero LLM
> calls**, and `make eval` scores extraction against a labelled corpus for $0.
>
> **One stage spends money: covenant extraction** (`opuscovintel extract`), confirmed live against
> `claude-opus-5` — 7 calls, $0.069, prompt caching engaged. Everything else is $0, and
> `--dry-run` prices a document before anything is dispatched.
>
> **The query agent makes no model calls.** `app/agent/` is fully deterministic and imports nothing
> from `app/llm/`; CLAUDE.md's routing table reserves a model for answer synthesis, and that stage
> is not built. The table's `Status` column says which stages are live.
>
> Deferred, deliberately: Celery/Redis, S3/MinIO, RBAC/OIDC and OTel/Prometheus — see
> [docs/deploy.md §6](docs/deploy.md) for the order to add them in. See [PLAN.md](PLAN.md) for the
> phase plan and [CLAUDE.md](CLAUDE.md) for architectural invariants.

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

## Asking it something

```bash
make demo            # migrate, seed, ingest, index, extract, then run the golden set
```

That runs the whole pipeline end to end for **$0**, and finishes by answering ten golden
questions with no model involved. Individually:

```bash
make index           # embed + full-text index every chunk
make extract-sample  # deterministic extractor -> cited clauses and covenants
make golden          # the golden question set
make ask-sample      # the same question through the LangGraph agent, logged and audited
```

The same question can be put to either path. `query` is the Phase 4 service; `ask` routes through
the agent, which adds the plan, the verify node and a row in `query_logs` — and, today, no model:

```bash
uv run opuscovintel ask "What is the cross-default threshold?"
```

### The one command that spends money

```bash
make extract-llm-dry-run   # price it first — free
uv run opuscovintel extract <document-id>
```

`extract` runs the LLM extractor over candidate spans. It refuses to run without an explicit
target (`--all` if you mean the whole corpus), prints the ceilings, and asks before dispatching.
`--dry-run` prices the work from the candidate spans without calling anything:

```
7 candidate span(s), ~22,565 prompt tokens, worst case $1.5128 (cap $2.00)
```

Worst case assumes every completion uses its full token budget, which is what the budget guard
assumes. The run that produced that estimate actually cost **$0.069**.

```bash
uv run opuscovintel query "Which holdings would breach their rating trigger at the current rating?"
```

```
intent:     covenant_breach_check
confidence: 0.95

1 covenant breach(es) found across 3 instrument(s).
  - RM300m Green Ijarah Sukuk [BREACH] rating_trigger: A- is below the trigger rating A by 1 notch(es)
  ...
Not evaluated for lack of reported facts:
  - RM300m Green Ijarah Sukuk: cross_default (no accelerated indebtedness reported)
```

Two things in that output are load-bearing. The breach is **computed** by
[`app/rules/covenants.py`](app/rules/covenants.py) against an ordinal rating rank, not inferred by
a model. And covenants that could not be evaluated are **named**, because "could not evaluate" and
"compliant" are different answers — silently dropping the first is the most dangerous output this
system could produce.

Ask it something the corpus cannot evidence and it refuses:

```bash
uv run opuscovintel query "Should we buy more Malaysian sukuk next quarter?"
```

## Measuring it

```bash
make eval-demo       # corpus + eval from nothing: migrate, seed, ingest, index, extract, score
make eval            # score what is already there -> evals/results/
make cost-report     # LLM spend by stage and by document
```

`make eval` is $0 — no metric calls a model — and writes a JSON record and a Markdown summary. It
scores each extractor **separately**, because the difference between them is the answer to "did the
LLM actually help?", which is the question two extractors running in parallel exist to make
answerable:

| Field | rule P | rule R | llm P | llm R |
|---|---|---|---|---|
| covenant_type | 1.00 | 1.00 | 0.83 | 0.71 |
| threshold_ratio | 1.00 | 1.00 | 0.67 | 0.67 |
| operator | 1.00 | 1.00 | 1.00 | 0.67 |

On documents written to be extractable, the regexes win — which is exactly what a $0 baseline is
for, and exactly the claim that needs re-testing the day a real trust deed arrives. Every report
says so on its first line.

A metric with no data reports nothing rather than zero: a corpus nobody has run the LLM over has no
rules-vs-LLM agreement rate, not a perfect one.

Operating it day to day, including what each review-queue trigger means and what to do when a
document is stuck, is in [docs/operate.md](docs/operate.md). Deployment is in
[docs/deploy.md](docs/deploy.md).

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
| `app/llm/` | Budget-guarded provider adapters, response cache, cost ledger, VLM service. |
| `app/extract/` | Regex + LLM extractors, prompts, citation verification, dry-run pricing. |
| `app/rules/` | Deterministic covenant evaluation, money and dates. Pure functions. |
| `app/retrieval/` | Hybrid retrieval: pgvector + tsvector, fused by reciprocal rank. |
| `app/query/` | The deterministic query path — intent, evidence, refusal. |
| `app/evals/` | Golden questions, labelled ground truth, and the metrics harness behind `make eval`. |
| `app/agent/` | LangGraph query graph + SQL guardrail. Deterministic — no model calls. |

## The four things that make this auditable

1. **The LLM never computes a breach.** Models extract and explain; a deterministic rules engine
   evaluates triggers and SQL aggregates exposure.
2. **Every fact traces to a span** — covenant → clause → chunk → (page, char offsets).
3. **Citations are verified, not trusted.** A quote that isn't found in its cited chunk goes to
   human review instead of into the database.
4. **The query agent holds a read-only database role.** Enforced by Postgres grants, not by
   convention. It runs on two sessions: reads on the read-only role, and its own query log on the
   read-write one, because a single session cannot both be denied writes and write an audit trail.

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
