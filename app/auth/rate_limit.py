"""Login rate limiting: exponential backoff over a table of attempts.

**Why this exists.** `POST /auth/login` and `POST /ui/login` accepted unlimited
attempts. scrypt costs ~170 ms per attempt, which slows one connection to about
six guesses a second -- a side effect of the hash, not a control. It does
nothing against a distributed attack, and nothing at all against credential
stuffing, where the attacker has one guess per account and needs no speed.

**Backoff, not lockout.** A threshold that disables an account hands anyone who
knows a username a denial-of-service against that person: five wrong passwords
and the analyst is locked out until an operator intervenes. Backoff makes each
further guess cost more while leaving the real user a way in -- wait, then try.
The delay doubles and is capped, so a patient attacker is bounded at a handful
of guesses per window rather than stopped outright, which is the honest
description of what any rate limiter buys.

**Two keys.** Per-username stops one account being ground down; per-IP stops
one host spraying many usernames, which the username counter alone never sees.
An attempt with no client IP (a unix socket, a proxy that strips it) is limited
by username only.

The address is whatever `request.client.host` reports, so behind a load
balancer every user shares the balancer's counter and the per-IP threshold
becomes global. `docs/deploy.md` 6 says what to configure; the header must be
trusted from the proxy only, since a client free to set `X-Forwarded-For` is a
client free to reset its own counter.

**Failures are counted for usernames that do not exist.** Anything else would
make the limiter the username oracle `app/auth/service.py` avoids: "this one
throttles, that one never does" answers exactly the question the identical
error messages refuse to.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.repositories.auth import FailureWindow, LoginAttemptRepository

logger = get_logger(__name__)

# 2**32 seconds is already longer than the cap can be; anything beyond this is
# a large integer computed for nothing.
_MAX_EXPONENT = 32


class LoginThrottledError(Exception):
    """Raised instead of checking a password, when the caller must wait.

    An exception rather than a return value, deliberately. A caller that
    forgets to handle it fails loudly with a 500; a caller that forgets to
    check a boolean silently skips the rate limit, which is the same shape of
    defect as a guardrail whose grant was never applied.
    """

    def __init__(self, retry_after: dt.timedelta, *, scope: str) -> None:
        self.retry_after = retry_after
        self.scope = scope
        super().__init__(
            f"too many failed login attempts for this {scope}; retry in {self.retry_after_seconds}s"
        )

    @property
    def retry_after_seconds(self) -> int:
        """Whole seconds, rounded up. `Retry-After: 0` would invite an instant retry."""
        return max(1, int(self.retry_after.total_seconds() + 0.999))


@dataclass(frozen=True)
class Decision:
    """The limiter's view of one key. `delay` is zero when nothing is owed."""

    scope: str
    failures: int
    delay: dt.timedelta
    retry_after: dt.timedelta


def backoff_delay(
    failures: int, *, threshold: int, base_seconds: int, max_seconds: int
) -> dt.timedelta:
    """How long to wait after `failures` consecutive failures.

    Zero below the threshold -- a person who mistypes their password twice
    should not be made to wait. From the threshold on it doubles: with the
    defaults (threshold 5, base 2s, cap 900s) the sixth attempt waits 2s, the
    tenth about half a minute, and the sixteenth hits the fifteen-minute cap.

    Pure and total, so the policy can be read and tested without a database.
    """
    if failures < threshold:
        return dt.timedelta(0)
    exponent = min(failures - threshold, _MAX_EXPONENT)
    return dt.timedelta(seconds=min(base_seconds * 2**exponent, max_seconds))


class LoginRateLimiter:
    """Reads and writes `login_attempts`, and decides who has to wait.

    Held by `AuthService`, which is the only chokepoint both the JSON API and
    the HTML form pass through.
    """

    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._attempts = LoginAttemptRepository(session)

    # -- policy --------------------------------------------------------------

    def _decide(
        self, window: FailureWindow, *, scope: str, threshold: int, now: dt.datetime
    ) -> Decision:
        delay = backoff_delay(
            window.failures,
            threshold=threshold,
            base_seconds=self._settings.LOGIN_BACKOFF_BASE_SECONDS,
            max_seconds=self._settings.LOGIN_BACKOFF_MAX_SECONDS,
        )
        retry_after = dt.timedelta(0)
        if delay and window.last_failure_at is not None:
            retry_after = max(dt.timedelta(0), (window.last_failure_at + delay) - now)
        return Decision(scope=scope, failures=window.failures, delay=delay, retry_after=retry_after)

    # -- enforcement ---------------------------------------------------------

    async def check(self, *, username: str, client_ip: str | None) -> None:
        """Raise `LoginThrottledError` if this attempt must not be made yet.

        Called *before* the password is verified: the point is to not do the
        work, and a limiter that runs afterwards has already paid the 170 ms it
        was supposed to save.
        """
        now = _now()
        since = now - dt.timedelta(minutes=self._settings.LOGIN_FAILURE_WINDOW_MINUTES)

        decisions = [
            self._decide(
                await self._attempts.failures_for_username(username, since=since),
                scope="username",
                threshold=self._settings.LOGIN_MAX_FAILURES_PER_USERNAME,
                now=now,
            )
        ]
        if client_ip:
            decisions.append(
                self._decide(
                    await self._attempts.failures_for_ip(client_ip, since=since),
                    scope="ip",
                    threshold=self._settings.LOGIN_MAX_FAILURES_PER_IP,
                    now=now,
                )
            )

        # The strictest key wins; reporting the shortest wait would let an
        # attacker retry into a limit that is still closed.
        worst = max(decisions, key=lambda d: d.retry_after)
        if worst.retry_after > dt.timedelta(0):
            logger.warning(
                "auth.login_throttled",
                extra={
                    "scope": worst.scope,
                    "failures": worst.failures,
                    "retry_after_seconds": int(worst.retry_after.total_seconds()),
                    # The username is already in the audit trail for every
                    # failed login; the IP is not logged as a key on its own.
                    "username": username,
                },
            )
            raise LoginThrottledError(worst.retry_after, scope=worst.scope)

    async def record(self, *, username: str, client_ip: str | None, succeeded: bool) -> None:
        """Write the attempt down, and commit it.

        The commit is the load-bearing part. `get_session` rolls back when a
        handler raises, and a failed login raises `HTTPException(401)` -- so an
        attempt recorded in the request's transaction is erased by the very
        failure it exists to count, and the limiter never counts past one.

        Nothing else is in flight at this point in a login: the row and its
        purge are the whole transaction.
        """
        now = _now()
        await self._attempts.record(
            username=username, client_ip=client_ip, succeeded=succeeded, at=now
        )
        await self._attempts.purge_for_username(
            username, before=now - dt.timedelta(hours=self._settings.LOGIN_ATTEMPT_RETENTION_HOURS)
        )
        await self._session.commit()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
