"""Security response headers — docs/review.md finding 5.

The header values themselves are easy to assert and easy to make meaningless.
What these tests actually pin is the *property* the finding is about: the pages
that render clause text lifted verbatim out of third-party PDFs execute no
inline script and no inline style, so a Jinja autoescaping bug has a second
layer to get through.

Two of these were written after the browser proved the policy wrong. A strict
`default-src 'self'` blanked `/docs` (Swagger bootstraps from an inline
`<script>`) and, before `inlineCritical` was turned off, stripped every style
from `/app`. Neither is visible from Python -- both needed a real browser and a
real console, which is [CLAUDE.md 7] making its point again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

import app.main as main_module
from app.auth.service import AuthService
from app.core.middleware import APP_CSP, DOCS_CSP, HSTS_VALUE
from app.domain.enums import UserRole

REPO_ROOT = Path(__file__).resolve().parent.parent


# -- the headers are present, everywhere -------------------------------------


@pytest.mark.parametrize("path", ["/health", "/ui/login", "/static/app.css", "/no-such-path"])
def test_every_response_carries_the_security_headers(client: TestClient, path: str) -> None:
    """Including a 404 and a static file.

    Applied as middleware rather than per-route precisely so a path nobody
    thought about still gets a policy -- an error page renders content too.
    """
    response = client.get(path)

    assert response.headers["content-security-policy"] == APP_CSP
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "same-origin"


# -- the property worth protecting -------------------------------------------


def test_the_application_policy_permits_no_inline_execution() -> None:
    """The whole point of the finding.

    The UI renders third-party clause text. `'unsafe-inline'` in either
    `script-src` or `style-src` would give back exactly the capability CSP is
    here to remove, and it is the easiest thing in the world to add while
    chasing a broken page.
    """
    assert "'unsafe-inline'" not in APP_CSP
    assert "'unsafe-eval'" not in APP_CSP
    assert "default-src 'self'" in APP_CSP
    # Clickjacking, base-tag hijacking and form exfiltration.
    assert "frame-ancestors 'none'" in APP_CSP
    assert "base-uri 'self'" in APP_CSP
    assert "form-action 'self'" in APP_CSP
    assert "object-src 'none'" in APP_CSP


def test_the_docs_exception_is_scoped_to_the_docs_path(client: TestClient) -> None:
    """Swagger's inline bootstrap is allowed there and nowhere else.

    `/docs` renders our own OpenAPI schema, not document text, and is disabled
    in production. The exception is acceptable there for exactly that reason,
    so this test pins that it cannot leak onto a page that renders clauses.
    """
    docs_csp = client.get("/docs").headers["content-security-policy"]
    page_csp = client.get("/ui/login").headers["content-security-policy"]

    assert docs_csp == DOCS_CSP
    assert "'unsafe-inline'" in docs_csp
    assert "'unsafe-inline'" not in page_csp


# -- HSTS is gated on the deployment actually being HTTPS ---------------------


def test_hsts_is_absent_over_plain_http(client: TestClient) -> None:
    """The local and test stacks are HTTP; a max-age there is a future outage."""
    assert "strict-transport-security" not in client.get("/health").headers


def test_hsts_is_sent_when_the_deployment_is_https(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import Settings, get_settings

    secure = Settings(ENVIRONMENT="local", SESSION_COOKIE_SECURE=True)
    monkeypatch.setattr(main_module, "get_settings", lambda: secure)
    get_settings.cache_clear()

    with TestClient(main_module.create_app()) as secure_client:
        assert secure_client.get("/health").headers["strict-transport-security"] == HSTS_VALUE

    get_settings.cache_clear()


# -- the client build must stay CSP-clean ------------------------------------


def test_the_built_client_app_has_no_inline_style_or_handler() -> None:
    """`inlineCritical: false` in frontend/angular.json, pinned.

    Angular's critical-CSS inliner emits a `<style>` block *and* an
    `onload="this.media='all'"` attribute on the stylesheet link. Under the
    application policy the browser blocks both, the real stylesheet never
    leaves `media="print"`, and every screen renders unstyled -- verified by
    turning it back on and loading the page.

    Nothing in Python notices, which is why this asserts on the build output.
    """
    index = REPO_ROOT / "frontend" / "dist" / "index.html"
    if not index.is_file():
        pytest.skip("client app not built; `make frontend` produces frontend/dist")

    html = index.read_text()

    assert "<style" not in html, "inlineCritical is back on; the CSP will strip every style"
    assert "onload=" not in html, "an inline event handler will be blocked by the CSP"


def test_the_build_config_keeps_critical_css_out_of_the_html() -> None:
    """The setting itself, so a checkout with no build still catches a flip."""
    config = json.loads((REPO_ROOT / "frontend" / "angular.json").read_text())
    production = config["projects"]["opuscovintel-frontend"]["architect"]["build"][
        "configurations"
    ]["production"]

    assert production["optimization"]["styles"]["inlineCritical"] is False


# -- CSRF, which the finding explicitly called out as worth pinning ----------


async def test_the_ui_login_cookie_stays_samesite_lax(
    anonymous_client: AsyncClient, db_session: AsyncSession
) -> None:
    """review.md finding 5: "Not a finding: CSRF ... worth an explicit test".

    `SameSite=lax` is what blocks a cross-site form POST, and it is the reason
    there is no CSRF token anywhere in this codebase. If it ever regresses to
    `none` -- which is what splitting the client app onto its own origin would
    require -- that must be a deliberate decision, not a discovery.

    This pins the **form login** in `app/web/routes.py`, which sets its own
    cookie inline. The JSON path has its own helper and its own test
    (`test_auth.py::test_login_sets_an_httponly_session_cookie`); two
    independent `set_cookie` calls means one of them can regress alone, and
    this was the one nothing covered.
    """
    service = AuthService(db_session)
    await service.create_user(
        username="csrf-probe", password="a-long-enough-passphrase", role=UserRole.REVIEWER
    )
    await db_session.flush()

    response = await anonymous_client.post(
        "/ui/login",
        data={"username": "csrf-probe", "password": "a-long-enough-passphrase"},
        follow_redirects=False,
    )

    assert response.status_code == 303, "the form login should redirect on success"
    cookie = response.headers["set-cookie"]
    assert "samesite=lax" in cookie.lower()
    assert "httponly" in cookie.lower()
