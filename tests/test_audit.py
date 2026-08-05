"""Audit log endpoint tests.

Phase 7 acceptance: "every mutation appears in audit_logs."

The audit trail is append-only and returned in reverse chronological order.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ops import AuditLog
from app.domain.enums import ActorType


@pytest.mark.usefixtures("storage_root")
class TestAuditTrailEndpoint:
    async def test_empty_trail_returns_nothing(
        self, api_client: AsyncClient
    ) -> None:
        response = await api_client.get(
            f"/audit/clause/{uuid.uuid4()}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["entries"] == []

    async def test_audit_trail_includes_all_actions(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        entity_type = "covenant"
        entity_id = uuid.uuid4()

        actions = ["extracted", "review_corrected", "breach_evaluated"]
        for action in actions:
            audit = AuditLog(
                actor_type=ActorType.SYSTEM,
                actor_id="test-suite",
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                payload_json={"test": True, "action": action},
            )
            db_session.add(audit)
        await db_session.flush()

        response = await api_client.get(
            f"/audit/{entity_type}/{entity_id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert data["entity_type"] == entity_type

    async def test_audit_trail_is_reverse_chronological(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        import datetime as dt
        entity_type = "clause"
        entity_id = uuid.uuid4()
        base = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.UTC)

        for i, action in enumerate(["created", "updated", "deleted"]):
            audit = AuditLog(
                actor_type=ActorType.USER,
                actor_id="tester",
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                payload_json={"step": i},
                created_at=base + dt.timedelta(seconds=i),
            )
            db_session.add(audit)
        await db_session.flush()

        response = await api_client.get(
            f"/audit/{entity_type}/{entity_id}"
        )
        assert response.status_code == 200
        data = response.json()
        entries = data["entries"]
        # Most recent first: "deleted", "updated", "created"
        assert entries[0]["action"] == "deleted"
        assert entries[1]["action"] == "updated"
        assert entries[2]["action"] == "created"

    async def test_audit_trail_respects_limit(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        entity_type = "clause"
        entity_id = uuid.uuid4()

        for i in range(20):
            audit = AuditLog(
                actor_type=ActorType.SYSTEM,
                action=f"action_{i}",
                entity_type=entity_type,
                entity_id=entity_id,
                payload_json={"i": i},
            )
            db_session.add(audit)
        await db_session.flush()

        response = await api_client.get(
            f"/audit/{entity_type}/{entity_id}?limit=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
        assert len(data["entries"]) == 5

    async def test_audit_trail_entity_isolation(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        entity_a = uuid.uuid4()
        entity_b = uuid.uuid4()

        for eid in [entity_a, entity_b]:
            audit = AuditLog(
                actor_type=ActorType.SYSTEM,
                action="test_action",
                entity_type="clause",
                entity_id=eid,
                payload_json={},
            )
            db_session.add(audit)
        await db_session.flush()

        response_a = await api_client.get(f"/audit/clause/{entity_a}")
        response_b = await api_client.get(f"/audit/clause/{entity_b}")
        assert response_a.json()["count"] == 1
        assert response_b.json()["count"] == 1

    async def test_payload_is_returned(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        entity_id = uuid.uuid4()
        audit = AuditLog(
            actor_type=ActorType.USER,
            actor_id="reviewer-1",
            action="review_corrected",
            entity_type="clause",
            entity_id=entity_id,
            payload_json={
                "old_value": "RM30m",
                "new_value": "RM50m",
                "reviewer": "reviewer-1",
            },
        )
        db_session.add(audit)
        await db_session.flush()

        response = await api_client.get(f"/audit/clause/{entity_id}")
        assert response.status_code == 200
        entry = response.json()["entries"][0]
        assert entry["payload"]["old_value"] == "RM30m"
        assert entry["payload"]["new_value"] == "RM50m"
