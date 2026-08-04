"""Health and readiness endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.middleware import REQUEST_ID_HEADER


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "opuscovintel"
    assert body["environment"] == "test"


def test_health_does_not_touch_the_database(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Liveness must stay green when the database is down (see health.py docstring)."""

    async def _explode() -> bool:
        raise AssertionError("/health must not check the database")

    monkeypatch.setattr("app.api.routes.health.check_database", _explode)
    assert client.get("/health").status_code == 200


def test_ready_returns_200_when_database_is_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _ok() -> bool:
        return True

    monkeypatch.setattr("app.api.routes.health.check_database", _ok)

    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"database": True}}


def test_ready_returns_503_when_database_is_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _down() -> bool:
        return False

    monkeypatch.setattr("app.api.routes.health.check_database", _down)

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers.get(REQUEST_ID_HEADER)


def test_upstream_request_id_is_preserved(client: TestClient) -> None:
    """A proxy-supplied id must survive so traces join up across hops."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"
