# Deploying OpusCovIntel

What it takes to stand this up somewhere other than a laptop, and what is
deliberately not here yet.

The MVP is three processes and a database. That is a decision, not an omission:
PLAN.md defers Celery, Redis, MinIO, OIDC and Prometheus to Phase 8 and nothing
in Phases 1–7 needs them. §6 below says what to add first when it does.

---

## 1. What runs

| Process | Command | What it does |
|---|---|---|
| `api` | `uvicorn app.main:app` | Upload, review queue, audit reads, `/health`, `/ready` |
| `worker` | `python -m app.worker.main` | Claims queued ingestion jobs, parses and chunks |
| `db` | `pgvector/pgvector:pg16` | Everything. Also the job queue |

There is no broker. `extraction_jobs.status = 'queued'` is the queue, claimed with
`SELECT ... FOR UPDATE SKIP LOCKED`, so several workers can poll the same table
without contending. That is the whole reason Redis is not here yet.

Extraction and OCR are **not** background work: both spend money, so both are
operator-invoked from the CLI and never triggered by an upload.

## 2. Prerequisites

- **Postgres 16+ with `vector`, `pg_trgm` and `unaccent`.** The schema will not
  create without `vector` — `document_chunks.embedding` is `vector(1024)`.
- **Two database roles.** `docker/postgres/init/01-init.sql` creates them, and
  a managed Postgres needs it run by hand:
  - the application role, read-write;
  - `opuscovintel_ro`, `SELECT`-only, with `default_transaction_read_only = on`
    and `statement_timeout = '5s'`.

  CLAUDE.md 1.6 is enforced by that second role, not by the SQL guardrail. The
  guardrail is defence in depth; the grant is the boundary. Deploying with
  `DATABASE_URL_RO` pointed at the read-write role silently removes it.
- **Python 3.12** (`>=3.12,<3.13`), or the runtime image, which pins it.
- **Persistent storage for `STORAGE_DIR`.** Document bytes live on the
  filesystem behind an S3-shaped interface (`app/ingest/storage.py`). Two
  containers sharing it need a shared volume; that is what the compose `storage`
  volume is. Swapping in S3 is an adapter, not a refactor.

## 3. Configuration

Every setting is an environment variable, loaded by `app/core/config.py`.
`.env.example` is the complete list; `uv run opuscovintel config` prints what a
process actually resolved, with secrets redacted.

The ones that decide behaviour rather than plumbing:

| Variable | Default | Why it matters |
|---|---|---|
| `DATABASE_URL` | compose-local | Read-write role |
| `DATABASE_URL_RO` | compose-local | **Must be the read-only role.** See §2 |
| `STORAGE_DIR` | `./var/storage` | Must survive a restart |
| `MAX_TOTAL_COST_USD` | `200.00` | Circuit breaker. All calls refused past it |
| `MAX_COST_PER_DOCUMENT_USD` | `2.00` | Aborts a document, marks `budget_exceeded` |
| `MAX_COST_PER_CALL_USD` | `0.50` | Rejects before dispatch |
| `MAX_VLM_PAGES_PER_DOC` | `40` | A document over it is refused, not truncated |
| `EXTRACTION_MODEL` | `claude-opus-5` | Part of the extraction identity — changing it re-runs everything |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.85` | Below it, a field goes to human review |
| `ANTHROPIC_API_KEY` | unset | **Absent means extraction cannot run.** Which is the safe default |

A blank value (`QWEN_API_KEY=`) is read as *absent*, not as an empty key. That
distinction cost a debugging session once: an empty string is not None, so every
`if key is None` fallback was skipped and indexing died rather than using the
offline embedder that was sitting right there.

## 4. Standing it up

```bash
make up          # build and start db, api, worker
make migrate     # alembic upgrade head
make seed        # synthetic instruments and portfolios; idempotent
```

Then verify, in this order — each check rules out a different failure:

```bash
curl -fsS localhost:8000/health          # process is alive
curl -sS -o /dev/null -w '%{http_code}' localhost:8000/ready   # 200; 503 means no database
uv run opuscovintel check                # the CLI can reach the database
uv run opuscovintel check-schema         # enum CHECK constraints match the models
uv run opuscovintel eval --skip-agent    # the pipeline still extracts what it used to
```

`/health` answers even with no database and `/ready` returns 503 in the same
situation. That difference is the point: a database blip should take an instance
out of the load balancer without an orchestrator killing a healthy container.

`check-schema` exists because `alembic check` cannot see enum constraints —
Alembic excludes type-bound CHECKs from autogenerate, so a value added to a
`StrEnum` without a migration passes every other check and then fails on an
`INSERT` in production.

## 5. Migrations

```bash
uv run alembic upgrade head
```

CI applies from scratch, downgrades to base, and re-applies on every pull
request, so a downgrade that does not work is caught before it is needed.

Order of operations on a deploy: **migrate first, then roll the application.**
Nothing in the schema is destructive today, and a new column that no running
code writes is harmless. A migration that drops or renames must be split across
two releases; there is no online-schema-change tooling here to hide behind.

## 6. What is deferred, and what to add first

CLAUDE.md 9 deferred Celery/Redis, S3/MinIO, RBAC/OIDC and OTel/Prometheus.
Phase 9 built authentication; PLAN.md Phase 10.6 declines most of the rest with
reasons. In the order the remaining constraints actually bite:

1. **Object storage (S3).** The first thing to break when the API and worker
   stop sharing a filesystem — which is the first thing that happens on more
   than one host. `app/ingest/storage.py` is already the S3-shaped seam.
2. ~~**Reviewer identity (OIDC).**~~ **Built in Phase 9**, and scoped down:
   session auth with two roles rather than OIDC. `human_reviews.reviewer_id` is
   no longer a client-supplied placeholder — it comes from the session, so the
   audit trail records who was actually signed in. Full OIDC stays deferred and
   is not currently needed. Security response headers now ship on every response
   (`app/core/middleware.py`); set `SESSION_COOKIE_SECURE=true` on an HTTPS
   deployment, which is also what turns HSTS on.

   **If you put a proxy in front of this, pass the client address through.**
   Login rate limiting counts failures per username *and* per client IP, and
   the IP it sees is `request.client.host`. Behind a load balancer that is the
   balancer, so every user shares one counter and `LOGIN_MAX_FAILURES_PER_IP`
   becomes a global limit rather than a per-attacker one. Run uvicorn with
   `--proxy-headers --forwarded-allow-ips=<proxy>` so `X-Forwarded-For` is
   honoured — and only from the proxy, since a client that can set the header
   itself can also reset its own counter.
3. **Metrics (OTel/Prometheus).** Spend and job state are already queryable —
   `opuscovintel cost-report`, `extraction_jobs`, `llm_calls` — so this buys
   alerting, not visibility. Alert on the budget ceilings and on the pending
   review count first.
4. **Celery/Redis.** Last. `SKIP LOCKED` polling handles this volume, and a
   broker is a second thing that can be down.

### The client app

`/app` is the Angular build in `frontend/dist`, served by the API process from its own origin.
The image builds it in a Node stage and copies only the static output, so no Node reaches the
runtime container. A deployment that has not built it loses `/app` and nothing else — the API and
`/ui` do not depend on it, and the process says so at startup.

Keep it on the API's origin. Splitting them means CORS plus `SameSite=none`, which trades the CSRF
property described above for a deployment convenience.

## 7. Storage, and who can see it

**One directory, shared by everything that touches documents.** `STORAGE_DIR`
holds the PDF bytes every citation ultimately points at. Compose bind-mounts
the repository's `./var/storage` into both `api` and `worker`, rather than
using a named volume, so a document ingested from a terminal
(`opuscovintel ingest`, or `uv run uvicorn` during development) and a document
uploaded to the containerised API land in the same place.

That was not always true, and the failure is worth recognising. With a named
volume the containers and the host had *separate* corpora: a PDF uploaded to a
locally-run API was invisible to the containerised worker, which claimed the
job, found no object at the storage key, and marked the document `failed` with

```
ObjectNotFoundError: no object at 'documents/53/97/5397afe…pdf'
```

Both halves were behaving correctly and the database rows were identical either
way, which is what made it take a while to understand.

**A document's bytes and its rows must be backed up and restored together.**
The database holds the citation offsets into the chunks; the filesystem holds
the document bytes those offsets point at. A restore of one without the other
leaves covenant rows citing spans in documents nobody can open, which fails
CLAUDE.md 1.2 in the least visible way possible: the rows still look fine.

The same applies to `docker compose down -v`, which drops the database volume
while `./var/storage` survives on disk — leaving files with nothing pointing at
them, and a corpus that has to be re-ingested rather than restored.

**Production is a different question.** A shared filesystem stops working the
moment `api` and `worker` run on different hosts; `app/ingest/storage.py` is
already an S3-shaped interface for exactly that day (PLAN.md defers S3/MinIO to
Phase 8).

Nothing else is precious. Chunks, embeddings, clauses and covenants can all be
rebuilt from the document bytes for $0 (`ingest`, `index`, `extract-rules`) —
but **human review decisions cannot**, and they live only in Postgres.
