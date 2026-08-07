"""Review queue API tests.

Phase 7 acceptance: "a correction preserves prior value + reviewer + reason ·
every mutation appears in audit_logs."

Tests run against the test DB inside a rolled-back transaction. The review
queue API is exercised through httpx against the FastAPI app.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ops import HumanReview
from app.db.repositories.ops import AuditLogRepository
from app.domain.enums import ReviewStatus, ReviewTrigger


async def _create_pending_review(
    session: AsyncSession,
    entity_type: str = "clause",
    field_name: str = "threshold_amount",
    old_value: str = "RM30,000,000",
) -> HumanReview:
    review = HumanReview(
        entity_type=entity_type,
        entity_id=uuid.uuid4(),
        field_name=field_name,
        old_value=old_value,
        trigger_reason=ReviewTrigger.RULE_LLM_DISAGREEMENT,
        status=ReviewStatus.PENDING,
    )
    session.add(review)
    await session.flush()
    return review


@pytest.mark.usefixtures("storage_root")
class TestListPending:
    async def test_empty_queue_returns_nothing(self, api_client: AsyncClient) -> None:
        response = await api_client.get("/review/pending")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["total_pending"] == 0

    async def test_pending_items_are_listed(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _create_pending_review(db_session)
        await _create_pending_review(db_session, field_name="gearing_ratio")

        response = await api_client.get("/review/pending")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["total_pending"] == 2
        assert len(data["items"]) == 2


@pytest.mark.usefixtures("storage_root")
class TestApproveReview:
    async def test_approve_marks_review_approved(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session)

        response = await api_client.post(
            f"/review/{review.id}/approve",
            json={"notes": "threshold confirmed"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["action"] == "approved"

        # Verify in DB
        await db_session.refresh(review)
        assert review.status == ReviewStatus.APPROVED
        assert review.reviewer_id == "test-reviewer"

    async def test_approve_writes_audit_log(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session)

        await api_client.post(
            f"/review/{review.id}/approve",
            json={},
        )

        # Audit log must exist
        entries = await AuditLogRepository(db_session).list_for_entity(
            review.entity_type, review.entity_id
        )
        assert len(entries) >= 1
        approved_entry = next((e for e in entries if e.action == "review_approved"), None)
        assert approved_entry is not None
        assert approved_entry.actor_id == "test-reviewer"

    async def test_approve_fails_for_already_resolved_review(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        import datetime as dt

        review = await _create_pending_review(db_session)
        review.status = ReviewStatus.APPROVED
        review.reviewer_id = "previous-reviewer"
        review.reviewed_at = dt.datetime.now(dt.UTC)
        await db_session.flush()

        response = await api_client.post(
            f"/review/{review.id}/approve",
            json={},
        )
        assert response.status_code == 409

    async def test_approve_fails_for_non_existent_review(self, api_client: AsyncClient) -> None:
        fake_id = uuid.uuid4()
        response = await api_client.post(
            f"/review/{fake_id}/approve",
            json={},
        )
        assert response.status_code == 404


@pytest.mark.usefixtures("storage_root")
class TestCorrectReview:
    async def test_correct_preserves_old_value(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session, old_value="RM30,000,000")

        response = await api_client.post(
            f"/review/{review.id}/correct",
            json={
                "new_value": "RM50,000,000",
                "notes": "misread — trust deed states RM50m",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "corrected"

        await db_session.refresh(review)
        assert review.status == ReviewStatus.CORRECTED
        # Old value preserved
        assert review.old_value == "RM30,000,000"
        assert review.new_value == "RM50,000,000"
        assert review.reviewer_id == "test-reviewer"

    async def test_correct_writes_audit_log_with_full_history(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session, old_value="RM30,000,000")

        await api_client.post(
            f"/review/{review.id}/correct",
            json={
                "new_value": "RM50,000,000",
                "notes": "corrected per trust deed",
            },
        )

        entries = await AuditLogRepository(db_session).list_for_entity(
            review.entity_type, review.entity_id
        )
        corrected = next((e for e in entries if e.action == "review_corrected"), None)
        assert corrected is not None
        payload = corrected.payload_json
        assert payload["old_value"] == "RM30,000,000"
        assert payload["new_value"] == "RM50,000,000"

    async def test_correct_requires_new_value(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session)
        response = await api_client.post(
            f"/review/{review.id}/correct",
            json={"new_value": ""},
        )
        assert response.status_code == 422


@pytest.mark.usefixtures("storage_root")
class TestRejectReview:
    async def test_reject_marks_review_rejected(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session)

        response = await api_client.post(
            f"/review/{review.id}/reject",
            json={
                "reason": "false positive — pattern matched a definition section",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"

        await db_session.refresh(review)
        assert review.status == ReviewStatus.REJECTED

    async def test_reject_writes_audit_log_with_reason(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session)

        await api_client.post(
            f"/review/{review.id}/reject",
            json={
                "reason": "not a real covenant — Boilerplate text",
            },
        )

        entries = await AuditLogRepository(db_session).list_for_entity(
            review.entity_type, review.entity_id
        )
        rejected = next((e for e in entries if e.action == "review_rejected"), None)
        assert rejected is not None
        reason = str(rejected.payload_json.get("rejection_reason", ""))
        assert "Boilerplate" in reason

    async def test_reject_requires_reason(
        self, api_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        review = await _create_pending_review(db_session)
        response = await api_client.post(
            f"/review/{review.id}/reject",
            json={"reason": ""},
        )
        assert response.status_code == 422
