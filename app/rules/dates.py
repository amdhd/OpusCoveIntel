"""Date windows: grace periods, notice deadlines and call windows.

Covenant language is full of clocks -- "within five business days of such
downgrade", "not paid when due within any applicable grace period", "on any of
the call dates set out below". Whether an obligation has been breached is often
purely a question of which side of a date we are on, which makes it exactly the
kind of decision that belongs in deterministic Python rather than in a model
(CLAUDE.md 1.1).

**Business days here are Monday to Friday.** Malaysia has no single national
weekend -- Johor, Kedah, Kelantan and Terengganu observe Friday-Saturday -- and
public holidays vary by state. Encoding a federal Mon-Fri week is a stated
approximation, not an oversight; a real holiday calendar is a data problem, and
pretending otherwise inside a breach test would be worse than being explicit.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Final

SATURDAY: Final[int] = 5
SUNDAY: Final[int] = 6


def is_business_day(day: dt.date) -> bool:
    """Mon-Fri. See the module docstring for the deliberate simplification."""
    return day.weekday() not in (SATURDAY, SUNDAY)


def add_business_days(start: dt.date, days: int) -> dt.date:
    """The date `days` business days after `start`.

    Counts forward from the day *after* `start`, which is how "within five
    business days of the downgrade" is read: the trigger day itself is day
    zero. `days=0` returns `start` unchanged.
    """
    if days < 0:
        raise ValueError("days must not be negative; use business_days_between instead")
    current = start
    remaining = days
    while remaining > 0:
        current += dt.timedelta(days=1)
        if is_business_day(current):
            remaining -= 1
    return current


def business_days_between(start: dt.date, end: dt.date) -> int:
    """Business days from `start` (exclusive) to `end` (inclusive).

    Negative when `end` precedes `start`, so it reads as a signed distance --
    "three days late" and "three days remaining" are the same computation.
    """
    if end == start:
        return 0
    step = 1 if end > start else -1
    count = 0
    current = start
    while current != end:
        current += dt.timedelta(days=step)
        if is_business_day(current):
            count += step
    return count


def deadline(event: dt.date, *, business_days: int) -> dt.date:
    """When an obligation triggered on `event` falls due."""
    return add_business_days(event, business_days)


def is_overdue(event: dt.date, *, business_days: int, as_of: dt.date) -> bool:
    """True once the deadline has passed. On the deadline itself it has not."""
    return as_of > deadline(event, business_days=business_days)


def grace_period_end(due: dt.date, *, calendar_days: int) -> dt.date:
    """End of a grace period expressed in calendar days.

    Grace periods in trust deeds are usually calendar days while notice periods
    are business days. Conflating the two moves a default date by a weekend,
    which is the difference between a cured payment and an event of default.
    """
    if calendar_days < 0:
        raise ValueError("calendar_days must not be negative")
    return due + dt.timedelta(days=calendar_days)


def is_within_window(day: dt.date, start: dt.date, end: dt.date) -> bool:
    """Inclusive on both ends, which is how call windows are written."""
    if end < start:
        raise ValueError("window end precedes its start")
    return start <= day <= end


def next_on_or_after(dates: Iterable[dt.date], as_of: dt.date) -> dt.date | None:
    """The earliest date on or after `as_of` -- e.g. the next call date."""
    upcoming = sorted(day for day in dates if day >= as_of)
    return upcoming[0] if upcoming else None


def days_until(target: dt.date, as_of: dt.date) -> int:
    """Calendar days from `as_of` to `target`; negative once it has passed."""
    return (target - as_of).days
