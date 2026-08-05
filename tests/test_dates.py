"""Date windows: grace periods, notice deadlines, call windows.

The distinction that matters here is business days versus calendar days. A
notice period counted in calendar days rather than business days moves a
default date across a weekend, which is the difference between a cured payment
and an event of default.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.rules.dates import (
    add_business_days,
    business_days_between,
    days_until,
    deadline,
    grace_period_end,
    is_business_day,
    is_overdue,
    is_within_window,
    next_on_or_after,
)

# 2026-08-05 is a Wednesday.
WEDNESDAY = dt.date(2026, 8, 5)
FRIDAY = dt.date(2026, 8, 7)
SATURDAY = dt.date(2026, 8, 8)
MONDAY = dt.date(2026, 8, 10)


def test_weekends_are_not_business_days() -> None:
    assert is_business_day(FRIDAY) is True
    assert is_business_day(SATURDAY) is False
    assert is_business_day(dt.date(2026, 8, 9)) is False  # Sunday
    assert is_business_day(MONDAY) is True


def test_five_business_days_from_a_wednesday_lands_on_the_next_wednesday() -> None:
    # Counting skips the intervening weekend, which is the entire point.
    assert add_business_days(WEDNESDAY, 5) == dt.date(2026, 8, 12)


def test_zero_business_days_is_the_same_day() -> None:
    assert add_business_days(WEDNESDAY, 0) == WEDNESDAY


def test_counting_forward_from_a_friday_skips_the_weekend() -> None:
    assert add_business_days(FRIDAY, 1) == MONDAY


def test_negative_business_days_are_rejected_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="negative"):
        add_business_days(WEDNESDAY, -1)


def test_business_days_between_is_a_signed_distance() -> None:
    assert business_days_between(WEDNESDAY, dt.date(2026, 8, 12)) == 5
    assert business_days_between(dt.date(2026, 8, 12), WEDNESDAY) == -5
    assert business_days_between(WEDNESDAY, WEDNESDAY) == 0


def test_a_deadline_is_not_overdue_on_the_deadline_itself() -> None:
    due = deadline(WEDNESDAY, business_days=5)

    assert is_overdue(WEDNESDAY, business_days=5, as_of=due) is False
    assert is_overdue(WEDNESDAY, business_days=5, as_of=due + dt.timedelta(days=1)) is True


def test_grace_periods_run_in_calendar_days_not_business_days() -> None:
    # A 14-day grace period from a Friday ends on a Friday, weekend included.
    assert grace_period_end(FRIDAY, calendar_days=14) == dt.date(2026, 8, 21)


def test_a_negative_grace_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        grace_period_end(FRIDAY, calendar_days=-1)


def test_call_windows_are_inclusive_at_both_ends() -> None:
    assert is_within_window(WEDNESDAY, WEDNESDAY, FRIDAY) is True
    assert is_within_window(FRIDAY, WEDNESDAY, FRIDAY) is True
    assert is_within_window(MONDAY, WEDNESDAY, FRIDAY) is False


def test_an_inverted_window_is_an_error_not_an_empty_result() -> None:
    with pytest.raises(ValueError, match="precedes"):
        is_within_window(WEDNESDAY, FRIDAY, WEDNESDAY)


def test_the_next_call_date_includes_today() -> None:
    dates = [dt.date(2028, 6, 15), dt.date(2029, 6, 15), dt.date(2030, 6, 15)]

    assert next_on_or_after(dates, dt.date(2028, 6, 15)) == dt.date(2028, 6, 15)
    assert next_on_or_after(dates, dt.date(2028, 6, 16)) == dt.date(2029, 6, 15)
    assert next_on_or_after(dates, dt.date(2031, 1, 1)) is None


def test_days_until_goes_negative_once_a_date_has_passed() -> None:
    assert days_until(FRIDAY, WEDNESDAY) == 2
    assert days_until(WEDNESDAY, FRIDAY) == -2
