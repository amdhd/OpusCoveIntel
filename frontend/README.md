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
make frontend-test     # unit tests, headless Chromium
make frontend-types    # regenerate src/app/api/schema.d.ts from the API's OpenAPI
```

Tests use Vitest in a real Playwright Chromium browser. After `npm ci`, run
`npx playwright install chromium` once in `frontend/` before `make frontend-test`.
CI installs Chromium and its Linux system dependencies automatically. The browser
is a development dependency and is not included in the application image.

Angular 20's `unit-test` builder is experimental. We use it to remove Karma's
unpatched dependency tree, while retaining real browser DOM tests and all existing
assertions. Polling tests use Vitest's fake timers instead of Angular `fakeAsync`.
The scoped `vitest → tinypool` override pins 2.1.2 to remove worker-option RCE
advisories; the suite exercises browser mode rather than the Node worker pool.
Remove the override when a supported Vitest version depends on a patched release.

`npm audit` still reports the moderate
[Vitest redirect-mock advisory](https://github.com/advisories/GHSA-82fw-gwwq-j7x9)
through four package entries. Its fix is in Vitest 4.1.11+, while Angular 20's
builder supports Vitest 3. Browser-mode mock RPC is token protected; this setup
does not use standalone mocker plugins or expose the test server to the network.
A supported Angular/Vitest upgrade is the follow-up. CI retains its existing
high-severity threshold; no advisories are ignored.

The build is optional: the API and `/ui` run without it, and `app/main.py` logs that `/app` is
unavailable rather than failing to start.

## Three rules worth keeping

**Types come from the server.** `src/app/api/schema.d.ts` is generated from the API's OpenAPI
document, which comes from the Pydantic models. Hand-written interfaces drift, and the first
symptom is a field that is `undefined` in production and fine in every test. CI regenerates and
fails if the committed copy has moved.

**The server decides when ingestion is done.** The upload screen polls until `terminal` comes back
true; it never infers completion from a status string it recognises. A client that decided for
itself would report a half-parsed document as finished the first time the pipeline gained a stage.

**`inlineCritical` stays off.** `optimization.styles.inlineCritical: false` in `angular.json` looks
like a performance setting being left on the table, and it is deliberate. The inliner emits an
inline `<style>` block and an `onload="this.media='all'"` attribute on the stylesheet link, and the
API serves a `Content-Security-Policy` with no `'unsafe-inline'` (docs/review.md finding 5). The
browser blocks both, the real stylesheet never leaves `media="print"`, and **every screen renders
unstyled** — it fails completely, not subtly. The policy is strict because the server-rendered
pages show clause text lifted verbatim out of third-party PDFs, so the fix is to keep the build
CSP-clean rather than to weaken the header. Two tests pin this: one on the built `index.html`, one
on the setting.

## Why Angular 20 rather than 21

The Angular 21 CLI requires Node ≥24.15; the machine this was built on runs 24.14. Nothing here
depends on a v20 API, so bumping is a version change and a `npm ci`.
