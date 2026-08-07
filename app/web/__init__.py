"""Server-rendered UI.

Jinja templates over the same services the JSON API uses -- no SPA, no build
step, no second language. A single-tenant internal tool with four screens does
not earn a frontend toolchain, and keeping the UI inside the app means a
covenant renders through the same `CatalogService` that answers the API, so the
two cannot disagree about what a covenant is.

The pages are progressive: every action is a plain form POST that works without
JavaScript. `static/app.js` adds a small amount of polish on top and nothing
the UI depends on.
"""
