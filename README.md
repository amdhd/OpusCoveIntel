# OpusCovIntel

[![CI](https://github.com/amdhd/OpusCoveIntel/actions/workflows/ci.yml/badge.svg)](https://github.com/amdhd/OpusCoveIntel/actions/workflows/ci.yml)

**Read a bond prospectus so an analyst doesn't have to — and show your working.**

OpusCovIntel reads sukuk and bond documents (prospectuses, trust deeds, rating reports), pulls
out the promises the issuer made, and lets you ask questions about them across a whole portfolio.
Every answer links back to the exact sentence on the exact page it came from.

---

## The problem

A Malaysian fixed-income fund might hold 200 bonds. Each one comes with a 300–600 page legal
document containing **covenants** — binding promises the issuer made, such as:

> *"An event of default shall occur if any indebtedness of the Issuer exceeding **RM30,000,000**
> becomes due and payable prior to its stated maturity."*

When a credit rating gets downgraded, someone has to answer *"which of our holdings just tripped a
covenant?"* within hours. Today that means analysts reading PDFs by hand. The answers exist, but
they're buried across thousands of pages, and the cost of missing one is a portfolio that breached
a limit nobody noticed.

## What this does

1. **Ingests** the PDFs — parses text, detects scanned pages, splits into chunks that remember
   which page and character range they came from.
2. **Extracts** covenants into structured database rows — thresholds, ratios, rating triggers, call
   schedules — each one carrying the verbatim quote it came from.
3. **Answers questions** across the portfolio, like *"which holdings breach their rating trigger at
   the current rating?"*, with citations you can click through to the source text.

## What makes it different from "chat with your PDF"

This is built for an audit, not a demo. Four rules are enforced in code, not in a prompt:

| Rule | Why it matters |
|---|---|
| **The AI never decides a breach** | Models read text and pull out numbers. Whether `1.9 > 1.75` is a breach is decided by plain Python in [`app/rules/`](app/rules/covenants.py), which is unit-tested and reproducible. An AI that computes compliance is an AI that can hallucinate compliance. |
| **Every fact traces to a source** | `covenant → clause → chunk → (page, character offsets)`. A covenant that cannot name its page and quote is rejected before it reaches the database. |
| **Quotes are checked, not trusted** | The model returns a quote; we verify that quote actually appears in the chunk we gave it. If it doesn't, the extraction goes to a human instead of into the database. |
| **"I don't know" is a valid answer** | If there's no supporting evidence, the system says so with confidence `0.0` rather than guessing. A confident wrong answer about a covenant is worse than no answer. |

There's a fifth, quieter one: **"could not evaluate" and "compliant" are never merged**. If a
covenant couldn't be checked because a figure wasn't reported, the output says that explicitly.
Silently treating unknown as fine is the most dangerous thing this system could do.

---

## How it works

```
PDF
 │
 ▼
┌─────────────┐   PyMuPDF + pdfplumber. Scores every page for text quality;
│  Ingest     │   pages that fail (scanned, image-heavy, garbled) are flagged
└─────────────┘   for the vision model. Costs $0.
 │
 ▼  chunks, each remembering (page, char_start, char_end)
┌─────────────┐   Regex + full-text search + vector similarity narrow
│  Narrow     │   ~500 pages down to ~50 candidate passages.
└─────────────┘   This is the main cost lever — ~20x saving.
 │
 ▼
┌─────────────┐   Two extractors run in parallel on every candidate:
│  Extract    │   • regex rules  ($0, deterministic)
└─────────────┘   • Claude Opus  (handles language regex can't)
 │                Where they disagree → human review, free signal.
 ▼  verified quotes only
┌─────────────┐   PostgreSQL: instruments, covenants, rating triggers,
│  Store      │   call schedules, portfolio holdings.
└─────────────┘
 │
 ▼
┌─────────────┐   LangGraph agent: classify → retrieve → evaluate with the
│  Ask        │   rules engine → answer → verify every claim → log.
└─────────────┘   Refuses when evidence is missing.
```

## Technology choices

| Layer | Choice | Why this one |
|---|---|---|
| **Language** | Python 3.12 | Full type hints; `mypy --strict` on the pure-logic modules. |
| **API** | FastAPI + Pydantic v2 | Request/response validation from the same types the domain uses. |
| **Database** | PostgreSQL 16 + `pgvector` | One database for relational data, full-text search *and* vectors. A separate vector store would mean two systems to keep consistent, for a corpus this size. |
| **ORM** | SQLAlchemy 2.x (async) + Alembic | Async throughout; migrations are reviewed code, not auto-applied drift. |
| **Retrieval** | Hybrid: `pgvector` HNSW + Postgres `tsvector`, fused by Reciprocal Rank Fusion | Legal text needs exact keyword matching (`"RM30,000,000"`) *and* semantic search (`"gearing"` ≈ `"leverage ratio"`). Either alone misses. |
| **Agent** | LangGraph | An explicit state graph, so the reasoning path is inspectable and each node testable — not a loop of opaque tool calls. |
| **LLM** | Claude Opus (`claude-opus-5`) | Long-context legal reasoning with structured output. Routed through one module with hard budget caps. |
| **UI** | Jinja templates + ~40 lines of JS | Four screens for internal users. A React SPA would add a build step, a second language and a second deployment for no gain. |
| **Auth** | Server-side sessions, `hashlib.scrypt` | No dependency; sessions are revocable rows, which a signed cookie isn't. |
| **Background work** | A Postgres table polled with `FOR UPDATE SKIP LOCKED` | At this volume a broker is operational surface with no benefit. Revisit when job volume justifies it. |
| **Packaging** | `uv`, multi-stage Docker, non-root | Locked dependencies; the runtime image contains no build tooling. |

**Deliberately not used:** Redis, Celery, MinIO, Kubernetes, a vector database, a frontend
framework. Each was considered and declined — the reasoning is in [PLAN.md](PLAN.md). They can be
added when something actually needs them.

---

## Quick start

```bash
make install     # create .venv (Python 3.12) and .env
make up          # start postgres+pgvector, api, worker
make migrate     # apply the schema
make seed        # load synthetic demo data
```

Create a login, then open <http://localhost:8000>:

```bash
make user-add u=yourname role=reviewer
```

To see the whole pipeline run end to end for **$0** — no API keys needed:

```bash
make demo
```

That ingests a synthetic prospectus, indexes it, extracts covenants with the regex extractor, and
answers thirteen benchmark questions without calling any model.

## Loading your own documents

Parsing is free and needs no API key.

```bash
uv run opuscovintel ingest path/to/prospectus.pdf
uv run opuscovintel index          # build the search indexes
```

> **There is no upload button in the UI yet.** Documents are loaded via the CLI above or
> `POST /documents/upload`. A browser upload screen is the next thing to build — see
> [PLAN.md Phase 10](PLAN.md).

Uploading the same file twice is not an error: documents are identified by SHA-256, so a repeat
returns the existing record.

### Running the AI extractor (this one costs money)

Always price it first — this is free and dispatches nothing:

```bash
uv run opuscovintel extract --all --dry-run
```

```
93 candidate span(s), ~467,917 prompt tokens, worst case $20.94 (cap $2.00)
  ** exceeds the per-document cap; the guard will stop mid-document **
```

Then run it against **one** document to check real cost before doing the rest:

```bash
uv run opuscovintel extract <document-id>
make cost-report
```

`extract` refuses to run without an explicit target, prints the ceilings, and asks before spending.

---

## Cost control

The design assumption is that an ungoverned pipeline over 600-page documents will quietly spend
hundreds of dollars. So spend is **capped**, not merely logged.

| Guard | Default | What happens |
|---|---|---|
| Per document | `$2.00` | Stops that document, queues it for review |
| Per call | `$0.50` | Rejected before dispatch |
| Global | `$200.00` | Circuit breaker — all calls refused |
| Vision-model pages per document | `40` | Fails loudly rather than silently truncating |

Three things keep the bill down:

1. **Candidate narrowing** — the model sees ~50 passages, never 600 pages. Roughly a 20× saving,
   and the single biggest lever.
2. **Prompt caching** — the system prompt and schema are byte-stable, so repeat calls bill the
   prefix at 0.1×. Measured: 221,627 cached tokens across one run.
3. **Response caching** — keyed on content hash, so re-running an unchanged pipeline costs $0.

**Real measurements from this repo**, three actual prospectuses (1,216 pages):

| Document | Pages | Candidates | Worst case |
|---|---|---|---|
| 2021 trust certificate | 535 | 93 | $20.94 |
| Dubai base prospectus | 201 | 51 | $11.48 |
| 2025 GMTN | 480 | 19 | $4.28 |

"Worst case" prices every reply at its full token budget, which is what the guard assumes. Actual
spend runs several times lower. **The `$2.00` default is too low for documents this size** and
needs raising deliberately — see [docs/review.md](docs/review.md).

### Which API keys you actually need

| Provider | Required? | Used for |
|---|---|---|
| **Anthropic** | **Yes** | Covenant extraction. The only stage that needs a key to work. |
| **OpenAI** | Only for scans | Vision fallback for pages with no text layer. Typically a handful of pages (~$0.02 each). |
| **Qwen / DashScope** | Optional | Embeddings. Without it the system falls back to a free offline embedder — search still works, but semantic matching is much weaker. |

Everything except extraction and vision OCR runs at $0, including the entire test suite.

---

## Security

- **Everything requires a login** except `/health`, `/ready` and the login page.
- **Two roles.** `analyst` reads and asks; `reviewer` can also decide review-queue items — the one
  action where a human overrides the machine.
- **Who approved what is not self-reported.** The reviewer recorded on a decision comes from the
  session, not from the request body, so the audit trail can't be forged by a client.
- **Passwords** use `hashlib.scrypt` (memory-hard) with per-password parameters stored alongside
  the hash, so cost can be raised later without invalidating anyone. New passwords must be at
  least twelve characters — length only, no "one symbol, one digit" rules.
- **Repeated failed logins back off**, per username and per client address, doubling up to a cap.
  Backoff rather than lockout, so knowing someone's username is not a way to lock them out.
- **Sessions** are revocable database rows. Only a SHA-256 of the token is stored, so a database
  dump doesn't hand over live sessions.
- **The query agent connects as a read-only Postgres role.** Generated SQL is parsed with
  `sqlglot`, checked against a table *and column* allowlist, capped with `LIMIT`, and bounded by a
  statement timeout. The grant is the real boundary; the parser is defence in depth. The role is
  denied the operational tables outright — the audit trail, the review queue, other people's
  questions and the cached model output are not readable by the thing being audited.
- **Login is deliberately uninformative** — wrong password, unknown user and disabled account
  return the same message in the same time, so the endpoint can't be used to enumerate staff.

Known gaps are listed honestly in [docs/review.md](docs/review.md): there are no security response
headers yet, and behind a proxy the per-address half of the login limit needs the client address
forwarded ([docs/deploy.md](docs/deploy.md) §6).

---

## Testing

```bash
make check    # lint + type check + 849 tests
make eval     # score extraction accuracy -> evals/results/
```

- **`make test` makes zero paid API calls.** Structurally, not by convention: billable tests are
  marked and skipped unless `RUN_LIVE_LLM_TESTS=1`, and CI is given no provider credentials at all.
- Tests run against a **dedicated** `opuscovintel_test` database, isolated by transaction rollback,
  so they never touch your development data.
- CI runs the suite, a **migration round-trip** (`upgrade` → `downgrade base` → `upgrade` plus a
  drift check), and a container build asserting `/health` returns 200 while `/ready` returns 503
  with no database.

`make eval` scores both extractors separately, because the whole point of running two is being able
to answer *"did the AI actually help?"* Current figures on the synthetic corpus:

| Extractor | Precision | Recall | F1 |
|---|---|---|---|
| Regex rules | 1.00 | 0.98 | **0.99** |
| Claude Opus | 0.91 | 0.98 | **0.95** |

On documents *written* to be extractable, regex wins — which is exactly what a free baseline is
for, and exactly the claim that needs re-testing against real prospectuses. Every report says so on
its first line.

---

## Project layout

| Path | Purpose |
|---|---|
| `app/api/` | HTTP routes. No business logic, no SQL. |
| `app/web/` | Server-rendered UI — Jinja templates, four screens. |
| `app/core/` | Settings, structured logging, request-id middleware. |
| `app/domain/` | Pydantic schemas and enums. Pure — imports nothing below it. |
| `app/db/` | SQLAlchemy models, repositories, dual (read-write / read-only) engines. |
| `app/ingest/` | PDF parsing, page-quality scoring, chunking, object storage. |
| `app/llm/` | Provider adapters behind a budget guard, response cache, cost ledger. |
| `app/extract/` | Regex + AI extractors, prompts, quote verification, dry-run pricing. |
| `app/rules/` | Deterministic covenant evaluation. Pure functions, heavily tested. |
| `app/retrieval/` | Hybrid search: vectors + full text, fused by reciprocal rank. |
| `app/agent/` | LangGraph query graph and the SQL guardrail. |
| `app/catalog/` | Read-side assembly of instruments, covenants and their provenance. |
| `app/auth/` | Password hashing, sessions, login. |
| `app/evals/` | Benchmark questions, labelled ground truth, metrics harness. |

Dependencies point one way: `api → services → repositories → models`. `domain/` and `rules/` are
leaves that import nothing above them.

## The UI

Four screens, server-rendered from the same services the JSON API uses — so the two can never
disagree about what a covenant is.

- **Ask** — question in, answer with clickable citations out. A refusal is shown as an answer, not
  an error, because it is one.
- **Source** — click a citation to see the quote highlighted inside the original text, with the
  page, how it was extracted, and whether the quote was verified. This is the screen that makes the
  audit trail checkable rather than merely recorded.
- **Review queue** — approve, correct or reject flagged extractions.
- **Portfolio** — holdings with covenant status; breaches sort to the top, and anything that
  couldn't be evaluated reads *not reported*, never *ok*.

---

## Status and what's next

**Phases 1–9 are built and running.** Documents ingest, extract into cited covenants, and answer
portfolio questions through a UI, an API and a CLI. The benchmark set passes 10/10.

Two things to be clear about:

- **All accuracy figures are measured against synthetic documents we wrote ourselves.** They show
  the harness works. They do *not* show the extractor is accurate on real prospectuses. Re-baselining
  against a real document is the top priority.
- **The query agent currently makes no model calls** — it's fully deterministic. Answer synthesis
  by a model is designed but not built. [CLAUDE.md](CLAUDE.md)'s routing table marks which stages
  are live.

The prioritised plan, including known bugs, security gaps and cost issues, is in
**[docs/review.md](docs/review.md)**. The phase plan is in [PLAN.md](PLAN.md); the architectural
rules that shouldn't be broken are in [CLAUDE.md](CLAUDE.md). Day-to-day operation is in
[docs/operate.md](docs/operate.md) and deployment in [docs/deploy.md](docs/deploy.md).

## Disclaimer

Output is **decision support, not investment or legal advice**, and is intended for human review
before any action is taken. No real prospectuses are committed to this repository — the fixtures
are synthetic, and licensed documents belong in `var/`, which is git-ignored.
