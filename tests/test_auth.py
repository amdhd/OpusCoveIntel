"""Authentication: passwords, sessions, and the boundary around the API.

These use `anonymous_client` and drive the real login endpoint, because the
thing under test *is* the identity dependency that every other test file
overrides away.

Three properties carry the most weight:

* protected endpoints refuse an anonymous caller,
* a review decision is attributed to the session, not to the request body, and
* login tells an attacker nothing about which usernames exist.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from httpx import AsyncClient, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.passwords import (
    MIN_PASSWORD_LENGTH,
    WeakPasswordError,
    hash_password,
    needs_rehash,
    validate_password,
    verify_password,
)
from app.auth.service import AuthService, normalize_username
from app.db.models.auth import User, UserSession
from app.db.models.ops import AuditLog, HumanReview
from app.domain.enums import ReviewStatus, ReviewTrigger, UserRole

pytestmark = pytest.mark.usefixtures("storage_root")

PASSWORD = "an-adequately-long-test-passphrase"


async def _make_user(
    session: AsyncSession,
    *,
    username: str = "aminah",
    role: UserRole = UserRole.REVIEWER,
    password: str = PASSWORD,
) -> object:
    service = AuthService(session)
    user = await service.create_user(username=username, password=password, role=role)
    await session.flush()
    return user


async def _login(
    client: AsyncClient, username: str = "aminah", password: str = PASSWORD
) -> Response:
    return await client.post("/auth/login", json={"username": username, "password": password})


async def _pending_review(session: AsyncSession) -> HumanReview:
    review = HumanReview(
        entity_type="covenant",
        entity_id=uuid.uuid4(),
        field_name="threshold_amount",
        old_value="RM30,000,000",
        trigger_reason=ReviewTrigger.RULE_LLM_DISAGREEMENT,
        status=ReviewStatus.PENDING,
    )
    session.add(review)
    await session.flush()
    return review


# -- password primitives -----------------------------------------------------


class TestPasswordHashing:
    def test_a_password_verifies_against_its_own_hash(self) -> None:
        encoded = hash_password(PASSWORD)
        assert verify_password(PASSWORD, encoded)
        assert not verify_password(PASSWORD + "x", encoded)

    def test_two_hashes_of_one_password_differ(self) -> None:
        """Salting, observably. Equal hashes would mean a rainbow table works."""
        assert hash_password(PASSWORD) != hash_password(PASSWORD)

    @pytest.mark.parametrize(
        "corrupt",
        ["", "not-a-hash", "scrypt$1$2", "scrypt$x$8$1$AAAA$AAAA", "bcrypt$1$2$3$4$5"],
    )
    def test_a_malformed_hash_is_a_failed_login_not_a_crash(self, corrupt: str) -> None:
        """A corrupt row must not 500 -- that tells an attacker they found one."""
        assert verify_password(PASSWORD, corrupt) is False

    def test_an_empty_password_is_refused_at_hash_time(self) -> None:
        with pytest.raises(ValueError, match="empty password"):
            hash_password("")

    def test_a_weaker_hash_is_flagged_for_rehash(self) -> None:
        assert needs_rehash(hash_password(PASSWORD, n=1 << 14))
        assert not needs_rehash(hash_password(PASSWORD))


# -- the strength policy -----------------------------------------------------


class TestPasswordPolicy:
    """Length, and one context rule. No composition rules -- see passwords.py."""

    @pytest.mark.parametrize("password", ["", "x", "short", "elevenchars"])
    def test_anything_under_twelve_characters_is_refused(self, password: str) -> None:
        assert len(password) < MIN_PASSWORD_LENGTH
        with pytest.raises(WeakPasswordError, match="at least 12"):
            validate_password(password)

    def test_twelve_ordinary_characters_are_enough(self) -> None:
        """No symbol, no digit, no capital. Length is the property that matters."""
        validate_password("correcthorse")

    def test_a_password_containing_the_username_is_refused(self) -> None:
        with pytest.raises(WeakPasswordError, match="username"):
            validate_password("aminah-aminah-aminah", username="Aminah")

    def test_an_absurdly_long_password_is_refused(self) -> None:
        """Not a strength rule -- an unbounded password is an unbounded scrypt input."""
        with pytest.raises(WeakPasswordError, match="at most"):
            validate_password("x" * 5000)

    def test_hashing_itself_stays_permissive(self) -> None:
        """The login path re-hashes with whatever the account already has.

        If `hash_password` applied the policy, raising the scrypt cost would
        turn every legacy short password into a 500 at the moment its owner
        types it correctly.
        """
        assert hash_password("short")

    async def test_a_weak_password_cannot_create_an_account(self, db_session: AsyncSession) -> None:
        with pytest.raises(WeakPasswordError):
            await AuthService(db_session).create_user(username="aminah", password="short")

        assert await AuthService(db_session).users.get_by_username("aminah") is None

    async def test_a_rejected_password_change_leaves_the_account_untouched(
        self, db_session: AsyncSession
    ) -> None:
        """Validate before mutating.

        A change that raises after assigning the hash would leave the user
        unable to log in with either password, and their sessions revoked.
        """
        user = await _make_user(db_session)
        service = AuthService(db_session)
        issued = await service.start_session(user, ttl=dt.timedelta(hours=1))  # type: ignore[arg-type]

        with pytest.raises(WeakPasswordError):
            await service.set_password(user, "short")  # type: ignore[arg-type]

        assert await service.authenticate("aminah", PASSWORD) is not None
        assert await service.resolve(issued.token) is not None

    async def test_an_account_created_before_the_policy_can_still_log_in(
        self, db_session: AsyncSession
    ) -> None:
        """The floor is at the point of choosing, not at the door.

        Enforcing it at login would lock out every account that predates the
        policy -- an outage dressed as a security improvement.
        """
        legacy = User(
            username="legacy",
            display_name="Legacy",
            password_hash=hash_password("short"),
            role=UserRole.ANALYST,
        )
        db_session.add(legacy)
        await db_session.flush()

        assert await AuthService(db_session).authenticate("legacy", "short") is not None


# -- the service -------------------------------------------------------------


class TestAuthService:
    async def test_usernames_are_case_and_whitespace_insensitive(
        self, db_session: AsyncSession
    ) -> None:
        await _make_user(db_session, username="Aminah")
        service = AuthService(db_session)
        assert await service.authenticate("  AMINAH  ", PASSWORD) is not None

    async def test_a_duplicate_username_is_refused(self, db_session: AsyncSession) -> None:
        await _make_user(db_session, username="aminah")
        with pytest.raises(ValueError, match="already exists"):
            await _make_user(db_session, username="AMINAH")

    async def test_a_deactivated_user_cannot_authenticate(self, db_session: AsyncSession) -> None:
        user = await _make_user(db_session)
        user.is_active = False  # type: ignore[attr-defined]
        await db_session.flush()

        service = AuthService(db_session)
        assert await service.authenticate("aminah", PASSWORD) is None

    async def test_the_session_row_never_holds_the_token(self, db_session: AsyncSession) -> None:
        """A database dump must not yield live sessions."""
        user = await _make_user(db_session)
        service = AuthService(db_session)
        issued = await service.start_session(user, ttl=dt.timedelta(hours=1))  # type: ignore[arg-type]

        rows = (await db_session.scalars(select(UserSession))).all()
        assert len(rows) == 1
        assert issued.token not in (rows[0].token_fingerprint or "")
        assert len(rows[0].token_fingerprint) == 64

    async def test_an_expired_session_does_not_resolve(self, db_session: AsyncSession) -> None:
        user = await _make_user(db_session)
        service = AuthService(db_session)
        issued = await service.start_session(user, ttl=dt.timedelta(seconds=-1))  # type: ignore[arg-type]
        assert await service.resolve(issued.token) is None

    async def test_a_revoked_session_does_not_resolve(self, db_session: AsyncSession) -> None:
        user = await _make_user(db_session)
        service = AuthService(db_session)
        issued = await service.start_session(user, ttl=dt.timedelta(hours=1))  # type: ignore[arg-type]

        assert await service.resolve(issued.token) is not None
        assert await service.end_session(issued.token) is True
        assert await service.resolve(issued.token) is None

    async def test_deactivating_a_user_kills_a_live_session(self, db_session: AsyncSession) -> None:
        """The account authorises, not the session row."""
        user = await _make_user(db_session)
        service = AuthService(db_session)
        issued = await service.start_session(user, ttl=dt.timedelta(hours=1))  # type: ignore[arg-type]

        user.is_active = False  # type: ignore[attr-defined]
        await db_session.flush()
        assert await service.resolve(issued.token) is None

    async def test_changing_a_password_revokes_every_session(
        self, db_session: AsyncSession
    ) -> None:
        """A password change usually follows a suspected compromise."""
        user = await _make_user(db_session)
        service = AuthService(db_session)
        first = await service.start_session(user, ttl=dt.timedelta(hours=1))  # type: ignore[arg-type]
        second = await service.start_session(user, ttl=dt.timedelta(hours=1))  # type: ignore[arg-type]

        await service.set_password(user, "a-different-passphrase-entirely")  # type: ignore[arg-type]

        assert await service.resolve(first.token) is None
        assert await service.resolve(second.token) is None

    def test_normalize_username_is_what_the_model_stores(self) -> None:
        assert normalize_username("  Ahmad.Rahman ") == "ahmad.rahman"


# -- the HTTP boundary -------------------------------------------------------


class TestTheBoundary:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/review/pending"),
            ("get", "/instruments"),
            ("get", "/portfolios"),
            ("get", "/documents"),
            ("get", f"/audit/covenant/{uuid.uuid4()}"),
            ("post", "/query"),
        ],
    )
    async def test_protected_endpoints_refuse_an_anonymous_caller(
        self, anonymous_client: AsyncClient, method: str, path: str
    ) -> None:
        call = getattr(anonymous_client, method)
        response = (
            await call(path, json={"question": "anything"})
            if method == "post"
            else await call(path)
        )
        assert response.status_code == 401, f"{method.upper()} {path} was reachable anonymously"

    async def test_health_stays_open(self, anonymous_client: AsyncClient) -> None:
        """A liveness probe carries no credentials and must not need any."""
        assert (await anonymous_client.get("/health")).status_code == 200

    async def test_login_sets_an_httponly_session_cookie(
        self, anonymous_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _make_user(db_session)
        response = await _login(anonymous_client)

        assert response.status_code == 200
        assert response.json()["user"]["username"] == "aminah"
        assert "password_hash" not in response.text

        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie.replace("samesite", "SameSite")

    async def test_a_session_cookie_unlocks_protected_endpoints(
        self, anonymous_client: AsyncClient, db_session: AsyncSession, seeded_universe: None
    ) -> None:
        await _make_user(db_session)
        assert (await anonymous_client.get("/instruments")).status_code == 401

        await _login(anonymous_client)
        assert (await anonymous_client.get("/instruments")).status_code == 200

    async def test_logout_revokes_the_session(
        self, anonymous_client: AsyncClient, db_session: AsyncSession, seeded_universe: None
    ) -> None:
        await _make_user(db_session)
        await _login(anonymous_client)
        assert (await anonymous_client.get("/auth/me")).status_code == 200

        assert (await anonymous_client.post("/auth/logout")).status_code == 204
        assert (await anonymous_client.get("/auth/me")).status_code == 401

    @pytest.mark.parametrize(
        ("username", "password"),
        [("aminah", "wrong-password"), ("nobody-at-all", PASSWORD)],
    )
    async def test_every_login_failure_looks_identical(
        self,
        anonymous_client: AsyncClient,
        db_session: AsyncSession,
        username: str,
        password: str,
    ) -> None:
        """A wrong password and an unknown user must be indistinguishable.

        Otherwise the endpoint enumerates who works here.
        """
        await _make_user(db_session)
        response = await anonymous_client.post(
            "/auth/login", json={"username": username, "password": password}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid username or password"


# -- attribution -------------------------------------------------------------


class TestReviewAttribution:
    async def test_an_approval_is_attributed_to_the_session(
        self, anonymous_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """The audit trail records who was logged in, not who the body claims."""
        await _make_user(db_session, username="aminah", role=UserRole.REVIEWER)
        review = await _pending_review(db_session)
        await _login(anonymous_client)

        response = await anonymous_client.post(
            f"/review/{review.id}/approve",
            json={"notes": "checked against page 2"},
        )
        assert response.status_code == 200

        await db_session.refresh(review)
        assert review.reviewer_id == "aminah"

        audit = (
            await db_session.scalars(select(AuditLog).where(AuditLog.entity_id == review.entity_id))
        ).all()
        assert [row.actor_id for row in audit] == ["aminah"]

    async def test_a_forged_reviewer_id_in_the_body_is_ignored(
        self, anonymous_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """The old contract took `reviewer_id` from the request. It must not.

        Sending it now is simply an unknown field; the recorded reviewer stays
        the authenticated one.
        """
        await _make_user(db_session, username="aminah", role=UserRole.REVIEWER)
        review = await _pending_review(db_session)
        await _login(anonymous_client)

        response = await anonymous_client.post(
            f"/review/{review.id}/approve",
            json={"reviewer_id": "the-head-of-compliance", "notes": "nice try"},
        )
        assert response.status_code == 200

        await db_session.refresh(review)
        assert review.reviewer_id == "aminah"

    async def test_an_analyst_may_read_the_queue_but_not_decide_it(
        self, anonymous_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _make_user(db_session, username="junior", role=UserRole.ANALYST)
        review = await _pending_review(db_session)
        await _login(anonymous_client, username="junior")

        assert (await anonymous_client.get("/review/pending")).status_code == 200

        response = await anonymous_client.post(f"/review/{review.id}/approve", json={})
        assert response.status_code == 403

        await db_session.refresh(review)
        assert review.status is ReviewStatus.PENDING, "a 403 must not have changed anything"
