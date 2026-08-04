# Handover — starting Phase 3 (Ingestion)

Written 2026-08-04, at commit `a7d8011`. Delete this file when Phase 3 lands.

Read [CLAUDE.md](CLAUDE.md) first — it holds the architectural invariants. This document
covers only what a fresh session cannot infer from the code.

---

## 1. Where the project is

Phases 0–2 are complete, committed, and pushed to `main` on
[amdhd/OpusCoveIntel](https://github.com/amdhd/OpusCoveIntel) (private).
Note the repo spells it `OpusCove**I**ntel`; the local directory is `OpusCovIntel`.

| | State |
|---|---|
| Docs | `CLAUDE.md` (invariants), `PLAN.md` (8 phases, cost model, data model) |
| Schema | 17 tables, Alembic migration `20260804_0803_initial_schema` |
| Code | domain enums + UUIDv7, rating rank engine, SQLAlchemy models, repositories, seed |
| Tests | **120 passing**; lint and `mypy` clean |
| API | `/health` + `/ready` only. No upload endpoint yet — that is Phase 3. |

**Verified, not assumed:** the migration applies from scratch, downgrades to zero and
re-applies; `alembic revision --autogenerate` reports no drift; the seed converges over
repeated runs; the read-only Postgres role exists and is restricted.

Nothing has called a paid LLM API yet. Phases 1–4 are designed to cost **$0** — the first
spend is Phase 5.

## 2. Resuming (about two minutes)

```bash
cd /Users/hadi/OpusCovIntel
open -a Docker            # daemon is often stopped; nothing below works without it
make up                   # db + api + worker
make migrate              # apply schema
make seed                 # synthetic instruments/portfolios (idempotent)
make check                # lint + type + test -> expect 120 passed
```

If `make check` is green you are in a known-good state and can start Phase 3.

## 3. Phase 3 scope

From `PLAN.md`. Still **$0** — no LLM calls. VLM fallback is *detected* here but not
*invoked* until Phase 5.

Build:

1. **Upload API** — `POST /documents/upload`, plus `GET /documents`, `GET /documents/{id}`.
   Needs `python-multipart` for form uploads.
2. **SHA256 dedup** — hash the bytes; `DocumentRepository.get_by_sha256` already exists and
   the unique constraint is already in the schema.
3. **Storage adapter** — local filesystem behind an S3-shaped interface, writing under
   `settings.STORAGE_DIR`. Keep the interface narrow (`put`, `get`, `uri_for`) so Phase 8 can
   swap in S3/MinIO without touching callers.
4. **PDF parsing** — PyMuPDF for text, pdfplumber for tables.
5. **Page-confidence scoring** — populate `document_pages` (the heuristic is written out in
   CLAUDE.md §4). Set `vlm_reason` where a page trips a check; a CHECK constraint enforces
   that a `vlm_used` page names its reason.
6. **Chunking** — every chunk must carry `(page_number, char_start, char_end)`; the whole
   citation chain depends on those offsets being real.
7. **Job tracking** — write `extraction_jobs` rows; the identity key is already unique in the
   schema, so re-processing an unchanged document should be a no-op.

Acceptance: duplicate detected by hash · text and tables extracted from a synthetic PDF ·
every chunk carries a real span · low-confidence pages correctly flagged.

Likely new dependencies: `pymupdf`, `pdfplumber`, `python-multipart`.

**You need synthetic PDF fixtures.** None exist yet — `tests/fixtures/` is empty, and real
prospectuses must never be committed (copyright, CLAUDE.md §7). Generate them in code so
they are reproducible: PyMuPDF can write PDFs directly, which conveniently also lets you
build a deliberately text-layer-less page to exercise the VLM detection path. The three
synthetic issuers in `app/db/seed.py` are the natural subjects.

## 4. Decisions already made that constrain Phase 3

- **Repositories never commit.** Transaction scope belongs to the caller. Add a service layer
  for ingestion rather than committing inside a repository.
- **No business logic in route handlers** (CLAUDE.md §3, §9). The upload endpoint should
  delegate to an ingestion service.
- **`Decimal` for money, never `float`.** Already enforced by column types.
- **Ratings go through `app/rules/ratings.py`.** Never string-compare a rating.
- **Celery/Redis stay deferred to Phase 8.** The worker polls; do not add a broker. Phase 3
  can enqueue by writing a `queued` row and letting `app/worker/main.py` claim it — that
  placeholder loop is waiting for exactly this.
- **Storage is local-FS now, S3 later.** Design the interface accordingly, but do not add
  MinIO.

## 5. Environment gotchas that cost time in this session

These are non-obvious and will otherwise be rediscovered the hard way.

- **Python 3.12 comes from `uv`.** The system Python is 3.14 and the project pins
  `>=3.12,<3.13`. Always `uv run ...`; never invoke system `python3`.
- **`pytest` addopts already contains `-q`.** Passing `-q` again yields `-qq`, which silently
  suppresses the summary line — this broke a scripted commit-splitter that grepped for
  `N passed`. Just run `uv run pytest`.
- **SQLAlchemy's `URL.__str__` masks the password as `***`.** Any URL that must actually
  connect needs `url.render_as_string(hide_password=False)`. This produces a confusing
  `InvalidPasswordError` rather than an obvious formatting error.
- **Tests use a dedicated `opuscovintel_test` database**, created on first use by
  `tests/conftest.py`, with rollback isolation per test. They will not see `make seed` data,
  by design. Override with `TEST_DATABASE_URL`.
- **`greenlet` is required** by SQLAlchemy's async bridge and fails at *runtime*, not import.
- **The Postgres init script only runs on first volume creation.** After editing
  `docker/postgres/init/*.sql` you need `make down-volumes && make up`. The migration creates
  the extensions itself so managed Postgres works regardless.
- **Ruff enforces PEP 695 generics** (`class Foo[T: Base]`), not `Generic[T]`.
- **`SIM300` is disabled project-wide** — settings attributes are `SCREAMING_CASE`, so ruff
  misreads `settings.MAX_COST == Decimal(...)` as a Yoda condition.
- **A `StrEnum` renders as its value** under `str()`; prefer explicit `.value` when it becomes
  a dict key.

## 6. Working conventions

- **Conventional Commits**, split by functionality into small commits. Each commit must
  import cleanly and keep the suite green. Verify with `git stash push --keep-index
  --include-untracked` so a commit is tested with only its own files present — otherwise the
  working tree still holds later work and every run is a false green.
- **Create a backup branch before any history rewrite**, and confirm
  `git diff backup/x main` is empty before deleting it.
- `make check` (lint + type + test) before every commit.
- Tests must make **zero paid API calls**. Billable tests are marked `live_llm` and skipped
  unless `RUN_LIVE_LLM_TESTS=1`.
- Synthetic fixtures only. No real prospectuses in the repo.

## 7. Open questions (unchanged from PLAN.md §9)

Nothing blocks Phase 3, but two will bite later:

1. **Qwen embedding dimensionality** is assumed 1024 and baked into the schema. Changing it
   means re-embedding the corpus and rebuilding the HNSW index — settle it before Phase 5.
2. **GPT vision model ID** is a placeholder (`gpt-4o` in `.env.example`). It needs a current,
   verified ID and per-image token cost before the VLM page cap can be tuned sensibly.

Also outstanding: real licensed documents (all fixtures are synthetic, so parsing heuristics
are untested against real layouts — the biggest schedule risk in the plan), and the share of
the corpus that is Bahasa Malaysia.

## 8. Known minor issues

- `starlette.testclient` emits a `StarletteDeprecationWarning` about `httpx` vs `httpx2`.
  Third-party, harmless; revisit if CI runs with `-W error`.
- The `worker` compose service previously reported `unhealthy` because it inherited the
  image's HTTP healthcheck while serving no HTTP. Fixed by disabling the healthcheck for
  that service.
