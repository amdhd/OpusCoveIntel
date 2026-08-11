# OpusCovIntel — client app

Angular 20, served by the FastAPI process at `/app`.

## What is here, and what is not

This app owns the screens that need client-side state:

| Screen | Why it is here rather than in Jinja |
|---|---|
| `documents` | Upload with two kinds of progress — bytes leaving the browser, then the worker ingesting — plus a failure path. This is [finding 7](../docs/review.md) |
| `ask` | Answer, citations and refusal rendered without a round trip per question |
| `instruments` | Selecting an instrument loads its covenants in place |
| `review` | Approve / correct / reject with optimistic-ish refresh and a 409 path |

**Portfolios and the provenance viewer stay server-rendered** (`/ui/...`), and the nav links out
to them. Rebuilding a working read-only page in a framework buys nothing; the two are one product
because they share one stylesheet — the build references
[`app/web/static/app.css`](../app/web/static/app.css) rather than copying it.

## Running it

```bash
make frontend          # npm ci && ng build  -> frontend/dist, served at /app
make frontend-serve    # ng serve on :4200, proxying the API on :8000
make frontend-test     # unit tests, headless Chrome
make frontend-types    # regenerate src/app/api/schema.d.ts from the API's OpenAPI
```

The build is optional: the API and `/ui` run without it, and `app/main.py` logs that `/app` is
unavailable rather than failing to start.

## Two rules worth keeping

**Types come from the server.** `src/app/api/schema.d.ts` is generated from the API's OpenAPI
document, which comes from the Pydantic models. Hand-written interfaces drift, and the first
symptom is a field that is `undefined` in production and fine in every test. CI regenerates and
fails if the committed copy has moved.

**The server decides when ingestion is done.** The upload screen polls until `terminal` comes back
true; it never infers completion from a status string it recognises. A client that decided for
itself would report a half-parsed document as finished the first time the pipeline gained a stage.

## Why Angular 20 rather than 21

The Angular 21 CLI requires Node ≥24.15; the machine this was built on runs 24.14. Nothing here
depends on a v20 API, so bumping is a version change and a `npm ci`.
