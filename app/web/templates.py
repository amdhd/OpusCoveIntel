"""The Jinja environment.

Autoescaping is the reason this is a module rather than an inline constructor
call. These pages render clause text lifted verbatim out of third-party PDFs
and free-text reviewer notes; escaping is a security control here, so it is set
explicitly and asserted in a test rather than left to a framework default that
could change.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.web.format import confidence, label, money

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
# Registered as filters rather than called in each template, so that a screen
# cannot quietly invent its own way of printing a ringgit figure.
templates.env.filters["money"] = money
templates.env.filters["confidence"] = confidence
templates.env.filters["label"] = label
# Explicit, not inherited. `select_autoescape` would also cover .html, but
# stating it leaves no doubt for a reader deciding whether `|safe` is needed.
templates.env.autoescape = True
templates.env.trim_blocks = True
templates.env.lstrip_blocks = True
