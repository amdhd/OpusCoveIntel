"""Metric primitives. Pure functions, no I/O, no database.

Kept separate from the harness so the arithmetic is unit-testable without a
corpus, and so the comparison rules are stated once rather than re-derived at
each call site.

Three rules govern everything here:

**Money and ratios are `Decimal`** (CLAUDE.md 6). A numeric-tolerance metric
written in `float` drifts on exactly the values it exists to check: the
threshold `1.75` a covenant is breached against is not `1.7500000000000002`,
and a harness that cannot tell them apart is measuring its own rounding.

**An enum has no partial credit.** `gearing_ratio` and `interest_cover` are not
75 per cent the same covenant. Enum fields compare by identity or not at all.

**A missing value and a wrong value are different errors.** Recall falls when
the extractor said nothing; precision falls when it said something wrong. A
single "accuracy" number hides which of those is happening, and they have
opposite fixes -- so every field reports both.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# A ratio restated at a different precision ("1.75" vs "1.750") is the same
# covenant; 1.75 vs 1.80 is not. Half a per cent separates those cases with room
# to spare and is far tighter than the gap between any two thresholds a document
# is likely to state for the same covenant.
DEFAULT_NUMERIC_TOLERANCE: Decimal = Decimal("0.005")

# Call dates falling on a non-business day are habitually restated to the next
# business day, so an exact-match date metric fails on a correct reading. Three
# days absorbs a weekend without merging two distinct annual call dates.
DEFAULT_DATE_TOLERANCE: dt.timedelta = dt.timedelta(days=3)


@dataclass(frozen=True)
class Score:
    """Precision, recall and F1 over one field or one entity type.

    `true_positives` counts values the extractor got right, `false_positives`
    values it asserted and got wrong (including values the document does not
    state), `false_negatives` values it missed. `supported` is how many labels
    carried the field at all -- a field no label states scores nothing, and
    reporting 1.0 for it would be worse than reporting nothing.
    """

    name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def supported(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def predicted(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def precision(self) -> float | None:
        """None when nothing was predicted -- an undefined ratio, not a zero."""
        return self.true_positives / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.true_positives / self.supported if self.supported else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def __add__(self, other: Score) -> Score:
        return Score(
            name=self.name,
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "supported": self.supported,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


class ScoreBoard:
    """Accumulates per-field scores across many comparisons."""

    def __init__(self) -> None:
        self._scores: dict[str, Score] = {}

    def record(
        self,
        field: str,
        *,
        true_positive: int = 0,
        false_positive: int = 0,
        false_negative: int = 0,
    ) -> None:
        current = self._scores.get(field, Score(name=field))
        self._scores[field] = current + Score(
            name=field,
            true_positives=true_positive,
            false_positives=false_positive,
            false_negatives=false_negative,
        )

    def compare(self, field: str, expected: object | None, actual: object | None) -> None:
        """Score one field of one matched pair, using this module's rules.

        The four cases are deliberately explicit rather than collapsed into a
        truth table: "both absent" must score *nothing at all*, and the easy
        mistake is to count it as a true positive, which inflates every field
        that most documents do not state.
        """
        if expected is None and actual is None:
            return
        if expected is None:
            self.record(field, false_positive=1)
            return
        if actual is None:
            self.record(field, false_negative=1)
            return
        if values_match(expected, actual):
            self.record(field, true_positive=1)
        else:
            # A wrong value is both a claim that was not true and a label that
            # was not found. Counting it once would let an extractor improve its
            # precision by replacing silence with noise.
            self.record(field, false_positive=1, false_negative=1)

    def missing(self, field: str, expected: object | None) -> None:
        """A label whose whole entity went unmatched: every stated field is a miss."""
        if expected is not None:
            self.record(field, false_negative=1)

    def spurious(self, field: str, actual: object | None) -> None:
        """A prediction whose whole entity was unmatched: every asserted field is wrong."""
        if actual is not None:
            self.record(field, false_positive=1)

    def score(self, field: str) -> Score:
        return self._scores.get(field, Score(name=field))

    def scores(self) -> tuple[Score, ...]:
        return tuple(self._scores[name] for name in sorted(self._scores))

    def total(self, name: str = "micro_average") -> Score:
        """Micro-average across every field recorded."""
        total = Score(name=name)
        for score in self._scores.values():
            total = total + score
        return Score(
            name=name,
            true_positives=total.true_positives,
            false_positives=total.false_positives,
            false_negatives=total.false_negatives,
        )


def values_match(expected: object, actual: object) -> bool:
    """Compare two field values under the rule appropriate to their type."""
    if isinstance(expected, Decimal):
        return numeric_matches(expected, actual)
    if isinstance(expected, dt.date) and not isinstance(expected, dt.datetime):
        return date_matches(expected, actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().casefold() == actual.strip().casefold()
    return bool(expected == actual)


def numeric_matches(
    expected: Decimal,
    actual: object,
    *,
    tolerance: Decimal = DEFAULT_NUMERIC_TOLERANCE,
) -> bool:
    """Relative-tolerance comparison in `Decimal`, never `float`.

    Relative rather than absolute because the same tolerance has to serve a
    gearing ratio of 1.75 and a cross-default threshold of RM30,000,000. Zero is
    special-cased: a relative tolerance around zero is meaningless, and a
    covenant threshold of zero must match only zero.
    """
    other = to_decimal(actual)
    if other is None:
        return False
    if expected == other:
        return True
    if expected == 0 or other == 0:
        return False
    scale = max(abs(expected), abs(other))
    return abs(expected - other) / scale <= tolerance


def date_matches(
    expected: dt.date,
    actual: object,
    *,
    tolerance: dt.timedelta = DEFAULT_DATE_TOLERANCE,
) -> bool:
    if isinstance(actual, dt.datetime):
        actual = actual.date()
    if not isinstance(actual, dt.date):
        return False
    return abs(expected - actual) <= tolerance


def to_decimal(value: object) -> Decimal | None:
    """Coerce a stored value to `Decimal`, or None if it is not a number.

    Thresholds round-trip through JSONB as strings, so the harness sees `"1.75"`
    where the extractor held `Decimal("1.75")`. `float` is accepted through its
    `str()` rather than directly: `Decimal(1.75)` is 1.75000000000000011102...,
    which fails a 0.5 per cent comparison against nothing but itself.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            return Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            return None
    return None


def ratio(numerator: int, denominator: int) -> float | None:
    """A rate, or None when the denominator is zero.

    Zero-denominator rates are reported as "no data" throughout the harness
    rather than as 0.0 or 1.0. Both of those read as measurements, and a metric
    with nothing behind it is the one most likely to be quoted.
    """
    return numerator / denominator if denominator else None
