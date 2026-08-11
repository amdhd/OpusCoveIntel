"""Serving the Angular build from the API's own origin.

Two things are worth pinning and neither is about Angular.

**A client-side route is not a file.** `/app/documents` exists only in the
browser's router, so a plain static mount answers 404 to a reload or a shared
link. The mount falls back to `index.html` -- but only for paths that look like
routes. A missing bundle must still 404, because a JavaScript request answered
with HTML fails later, somewhere else, and much less legibly.

**The build is optional.** A Python-only checkout has never run `make frontend`,
and neither has CI's test job. The API and the server-rendered UI must not care.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module

INDEX = "<!doctype html><html><body><app-root></app-root></body></html>"


@pytest.fixture
def built_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory shaped like `ng build` output."""
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "main-ABC123.js").write_text("console.log('bundle');")
    monkeypatch.setattr(main_module, "CLIENT_APP_DIR", tmp_path)
    return tmp_path


def client_for() -> TestClient:
    return TestClient(main_module.create_app())


def test_the_api_starts_without_a_client_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal state of a checkout that has never run `make frontend`."""
    monkeypatch.setattr(main_module, "CLIENT_APP_DIR", tmp_path / "never-built")

    with client_for() as client:
        assert client.get("/health").status_code == 200
        # No mount, so nothing answers here -- a missing screen, not a broken app.
        assert client.get("/app/").status_code == 404


def test_the_index_is_served_at_the_mount_root(built_client: Path) -> None:
    with client_for() as client:
        response = client.get("/app/")

    assert response.status_code == 200
    assert "<app-root>" in response.text


def test_a_client_route_falls_back_to_the_index(built_client: Path) -> None:
    """A deep link and a reload both land here, and both must work."""
    with client_for() as client:
        response = client.get("/app/documents")

    assert response.status_code == 200
    assert "<app-root>" in response.text


def test_a_bundle_is_served_as_itself(built_client: Path) -> None:
    with client_for() as client:
        response = client.get("/app/main-ABC123.js")

    assert response.status_code == 200
    assert "bundle" in response.text


def test_a_missing_asset_is_a_404_not_the_index(built_client: Path) -> None:
    """The failure that would otherwise be silent.

    Answering a missing `.js` with `index.html` gives the browser HTML where it
    expected a script: a syntax error in the console, far from the cause.
    """
    with client_for() as client:
        response = client.get("/app/main-DELETED.js")

    assert response.status_code == 404
    assert "<app-root>" not in response.text


def test_the_server_rendered_ui_is_untouched(built_client: Path) -> None:
    """Both UIs are served by one process; mounting one must not shadow the other."""
    with client_for() as client:
        response = client.get("/ui/login")

    assert response.status_code == 200
    assert "Sign in" in response.text
