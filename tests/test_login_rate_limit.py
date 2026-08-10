"""Login rate limiting.

The property under test is not "a 429 appears somewhere". It is that repeated
failures make the *next* attempt refuse to check a password at all, that the
counter survives the failed request that produced it, and that none of it tells
an attacker which usernames exist.

The counter surviving is the one worth stating twice. `get_session` rolls back
whenever a handler raises, and a failed login raises `HTTPException(401)` -- so
an attempt row written in the request's own transaction is erased by the very
failure it exists to count, and the limiter would sit at one failure forever
while reporting success.

`test_the_attempt_outlives_a_rolled_back_transaction` is the test for that, and
it rolls the session back itself rather than going through the endpoint: the
suite overrides `get_session` with the test's own session, so an HTTP test
never runs the dependency whose rollback is the hazard. Driving the endpoint
would have looked like the stronger test and proved nothing -- the row survives
there whatever `record()` does.
"""

from __future__ import annotations

import datetime as dt

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.rate_limit import LoginRateLimiter, LoginThrottledError, backoff_delay
from app.auth.service import AuthService
from app.db.models.auth import LoginAttempt
from app.domain.enums import UserRole

pytestmark = pytest.mark.usefixtures("storage_root")

PASSWORD = "an-adequately-long-test-passphrase"
USERNAME = "aminah"


@pytest.fixture(autouse=True)
def _tight_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small thresholds, so a test costs three scrypt hashes rather than six.

    The policy is identical at any threshold -- `test_backoff_*` below pins the
    shape of the curve -- and every hash here is ~170 ms of deliberate work.
    """
    monkeypatch.setenv("LOGIN_MAX_FAILURES_PER_USERNAME", "2")
    monkeypatch.setenv("LOGIN_MAX_FAILURES_PER_IP", "4")
    monkeypatch.setenv("LOGIN_BACKOFF_BASE_SECONDS", "30")
    monkeypatch.setenv("LOGIN_FAILURE_WINDOW_MINUTES", "15")


async def _make_user(session: AsyncSession, *, username: str = USERNAME) -> None:
    await AuthService(session).create_user(
        username=username, password=PASSWORD, role=UserRole.REVIEWER
    )
    await session.flush()


async def _attempts(session: AsyncSession, *, username: str = USERNAME) -> int:
    return (
        await session.scalar(
            select(func.count()).select_from(LoginAttempt).where(LoginAttempt.username == username)
        )
    ) or 0


async def _backdate(session: AsyncSession, *, by: dt.timedelta) -> None:
    """Move every recorded attempt into the past.

    Cheaper and more honest than sleeping: the limiter compares stored
    timestamps against the clock, so shifting the timestamps tests exactly what
    waiting would.
    """
    await session.execute(update(LoginAttempt).values(created_at=LoginAttempt.created_at - by))
    await session.flush()


# -- the policy, without a database ------------------------------------------


class TestBackoffCurve:
    def test_nothing_is_owed_below_the_threshold(self) -> None:
        for failures in range(5):
            assert backoff_delay(failures, threshold=5, base_seconds=2, max_seconds=900) == (
                dt.timedelta(0)
            )

    def test_the_delay_doubles_from_the_threshold_on(self) -> None:
        delays = [
            backoff_delay(failures, threshold=5, base_seconds=2, max_seconds=900).total_seconds()
            for failures in (5, 6, 7, 8)
        ]

        assert delays == [2, 4, 8, 16]

    def test_the_delay_is_capped(self) -> None:
        """Unbounded doubling is a lockout with extra steps."""
        assert backoff_delay(50, threshold=5, base_seconds=2, max_seconds=900) == dt.timedelta(
            seconds=900
        )
        # And does not overflow into an unrepresentable timedelta on the way.
        assert backoff_delay(10_000, threshold=5, base_seconds=2, max_seconds=900) == dt.timedelta(
            seconds=900
        )


# -- the limiter, against the database ---------------------------------------


async def test_failures_are_counted_and_then_refused(db_session: AsyncSession) -> None:
    await _make_user(db_session)
    service = AuthService(db_session)

    for _ in range(2):
        assert await service.authenticate(USERNAME, "wrong") is None

    with pytest.raises(LoginThrottledError) as caught:
        await service.authenticate(USERNAME, "wrong")

    assert caught.value.scope == "username"
    assert caught.value.retry_after_seconds > 0


async def test_the_right_password_is_refused_while_throttled(db_session: AsyncSession) -> None:
    """The check runs before the password does.

    A limiter that let a correct password through would still be useful, but it
    would also confirm to an attacker the moment they guessed right -- and it
    would have paid the scrypt cost to find out.
    """
    await _make_user(db_session)
    service = AuthService(db_session)

    for _ in range(2):
        await service.authenticate(USERNAME, "wrong")

    with pytest.raises(LoginThrottledError):
        await service.authenticate(USERNAME, PASSWORD)


async def test_waiting_out_the_delay_lets_the_user_back_in(db_session: AsyncSession) -> None:
    """Backoff, not lockout: nobody has to call an operator."""
    await _make_user(db_session)
    service = AuthService(db_session)

    for _ in range(2):
        await service.authenticate(USERNAME, "wrong")
    with pytest.raises(LoginThrottledError):
        await service.authenticate(USERNAME, PASSWORD)

    await _backdate(db_session, by=dt.timedelta(minutes=1))

    user = await service.authenticate(USERNAME, PASSWORD)
    assert user is not None


async def test_a_success_clears_the_count(db_session: AsyncSession) -> None:
    """Someone who mistypes twice and then gets in starts from zero."""
    await _make_user(db_session)
    service = AuthService(db_session)

    await service.authenticate(USERNAME, "wrong")
    assert await service.authenticate(USERNAME, PASSWORD) is not None

    # Two more failures would trip the limit if the earlier one still counted.
    for _ in range(2):
        assert await service.authenticate(USERNAME, "wrong") is None


async def test_an_unknown_username_is_throttled_identically(db_session: AsyncSession) -> None:
    """No oracle.

    If attempts against a non-existent account were not counted, "does this
    username throttle?" would answer the question the identical error messages
    exist to refuse.
    """
    service = AuthService(db_session)

    for _ in range(2):
        assert await service.authenticate("no-such-person", "wrong") is None

    with pytest.raises(LoginThrottledError):
        await service.authenticate("no-such-person", "wrong")


async def test_one_address_spraying_many_usernames_is_stopped(db_session: AsyncSession) -> None:
    """The per-username counter never sees this attack.

    Four usernames, one failure each, from one host: every username counter
    reads 1, and only the IP counter notices.
    """
    service = AuthService(db_session)

    for i in range(4):
        assert await service.authenticate(f"analyst-{i}", "wrong", client_ip="10.0.0.9") is None

    with pytest.raises(LoginThrottledError) as caught:
        await service.authenticate("analyst-99", "wrong", client_ip="10.0.0.9")

    assert caught.value.scope == "ip"
    # A different address is unaffected -- the limit is on the attacker, not on
    # the endpoint.
    assert await service.authenticate("analyst-99", "wrong", client_ip="10.0.0.10") is None


async def test_attempts_older_than_the_retention_window_are_purged(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOGIN_ATTEMPT_RETENTION_HOURS", "24")
    service = AuthService(db_session)

    await service.authenticate(USERNAME, "wrong")
    await _backdate(db_session, by=dt.timedelta(hours=25))
    assert await _attempts(db_session) == 1

    await service.authenticate(USERNAME, "wrong")

    # The stale row is gone; the fresh one is not.
    assert await _attempts(db_session) == 1


# -- through the HTTP surface -------------------------------------------------


async def test_the_attempt_outlives_a_rolled_back_transaction(db_session: AsyncSession) -> None:
    """The rollback trap.

    In production the caller's transaction is rolled back by `get_session` the
    moment the handler raises its 401. An attempt recorded inside that
    transaction disappears with it, so the count never passes one and the
    limiter silently does nothing.

    Rolled back here by hand, because the suite overrides `get_session` with
    the test's own session: an HTTP test would never reach the rollback and
    would pass with or without the commit in `record()`.
    """
    service = AuthService(db_session)

    await service.authenticate(USERNAME, "wrong")
    await db_session.rollback()

    assert await _attempts(db_session) == 1


async def test_the_endpoint_records_every_refused_attempt(
    anonymous_client: AsyncClient, db_session: AsyncSession
) -> None:
    """Two 401s leave two rows: the route reaches the limiter at all."""
    await _make_user(db_session)

    for _ in range(2):
        response = await anonymous_client.post(
            "/auth/login", json={"username": USERNAME, "password": "wrong"}
        )
        assert response.status_code == 401

    assert await _attempts(db_session) == 2


async def test_the_api_answers_429_with_retry_after(
    anonymous_client: AsyncClient, db_session: AsyncSession
) -> None:
    await _make_user(db_session)

    for _ in range(2):
        await anonymous_client.post("/auth/login", json={"username": USERNAME, "password": "wrong"})

    response = await anonymous_client.post(
        "/auth/login", json={"username": USERNAME, "password": "wrong"}
    )

    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0
    # The 429 body says no more than the 401 does about who exists.
    assert USERNAME not in response.text


async def test_the_login_form_renders_the_wait_rather_than_a_bare_429(
    anonymous_client: AsyncClient, db_session: AsyncSession
) -> None:
    await _make_user(db_session)

    for _ in range(2):
        await anonymous_client.post("/ui/login", data={"username": USERNAME, "password": "wrong"})

    response = await anonymous_client.post(
        "/ui/login", data={"username": USERNAME, "password": "wrong"}
    )

    assert response.status_code == 429
    assert "Too many failed sign-in attempts" in response.text
    # Still a form, so the person who is locked out for 30 seconds has
    # somewhere to type when the 30 seconds are up.
    assert 'name="password"' in response.text


async def test_the_limiter_records_the_client_address_the_request_came_from(
    anonymous_client: AsyncClient, db_session: AsyncSession
) -> None:
    """Per-IP limiting is only real if the address reaches the row."""
    await _make_user(db_session)

    await anonymous_client.post("/auth/login", json={"username": USERNAME, "password": "wrong"})

    recorded = (
        await db_session.execute(select(LoginAttempt).where(LoginAttempt.username == USERNAME))
    ).scalar_one()
    assert recorded.client_ip
    assert recorded.succeeded is False


async def test_the_limiter_never_stores_the_password(db_session: AsyncSession) -> None:
    """Cheap, and the failure it guards against is unrecoverable."""
    limiter = LoginRateLimiter(db_session)
    await limiter.record(username=USERNAME, client_ip="10.0.0.1", succeeded=False)

    row = (
        await db_session.execute(select(LoginAttempt).where(LoginAttempt.username == USERNAME))
    ).scalar_one()

    assert not hasattr(row, "password")
    assert PASSWORD not in str(row.__dict__)
