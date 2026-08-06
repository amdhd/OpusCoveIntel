"""The metric primitives, and the labels they score against.

Every metric here is tested in both directions. A metric that only ever gets
matching inputs passes against an extractor that returns nothing -- HANDOVER
records exactly that failure from the previous phase, where a "fix" test passed
against unfixed code and an assertion of `x >= 0` caught nothing. So each case
below pairs "this matches" with "this specifically does not".
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from app.evals import labels as label_module
from app.evals.metrics import (
    DEFAULT_DATE_TOLERANCE,
    Score,
    ScoreBoard,
    date_matches,
    numeric_matches,
    ratio,
    to_decimal,
    values_match,
)

# -- Score ------------------------------------------------------------------


def test_score_computes_precision_recall_f1() -> None:
    score = Score(name="x", true_positives=6, false_positives=2, false_negatives=2)
    assert score.precision == pytest.approx(0.75)
    assert score.recall == pytest.approx(0.75)
    assert score.f1 == pytest.approx(0.75)


def test_score_reports_none_rather_than_zero_when_undefined() -> None:
    """An undefined rate must not read as a measured zero."""
    nothing_predicted = Score(name="x", false_negatives=3)
    assert nothing_predicted.precision is None
    assert nothing_predicted.recall == 0.0

    nothing_labelled = Score(name="x", false_positives=3)
    assert nothing_labelled.recall is None
    assert nothing_labelled.f1 is None

    empty = Score(name="x")
    assert (empty.precision, empty.recall, empty.f1) == (None, None, None)


def test_scores_add() -> None:
    total = Score(name="a", true_positives=1, false_positives=1) + Score(
        name="a", true_positives=2, false_negatives=3
    )
    assert (total.true_positives, total.false_positives, total.false_negatives) == (3, 1, 3)


# -- ScoreBoard -------------------------------------------------------------


def test_both_absent_scores_nothing() -> None:
    """The case that inflates every metric if it is counted as a hit."""
    board = ScoreBoard()
    board.compare("threshold_ratio", None, None)
    score = board.score("threshold_ratio")
    assert (score.true_positives, score.false_positives, score.false_negatives) == (0, 0, 0)
    assert score.f1 is None


def test_hallucinated_value_is_a_false_positive() -> None:
    board = ScoreBoard()
    board.compare("threshold_ratio", None, Decimal("66"))
    assert board.score("threshold_ratio").false_positives == 1
    assert board.score("threshold_ratio").false_negatives == 0


def test_missed_value_is_a_false_negative() -> None:
    board = ScoreBoard()
    board.compare("threshold_amount", Decimal("30000000"), None)
    assert board.score("threshold_amount").false_negatives == 1
    assert board.score("threshold_amount").false_positives == 0


def test_wrong_value_counts_against_both_precision_and_recall() -> None:
    """Replacing silence with a wrong answer must not improve any rate."""
    board = ScoreBoard()
    board.compare("threshold_amount", Decimal("30000000"), Decimal("50000000"))
    score = board.score("threshold_amount")
    assert (score.true_positives, score.false_positives, score.false_negatives) == (0, 1, 1)


def test_micro_average_sums_every_field() -> None:
    board = ScoreBoard()
    board.compare("a", Decimal("1"), Decimal("1"))
    board.compare("b", Decimal("1"), None)
    total = board.total()
    assert total.true_positives == 1
    assert total.false_negatives == 1


# -- numeric tolerance ------------------------------------------------------


def test_numeric_tolerance_accepts_restated_precision() -> None:
    assert numeric_matches(Decimal("1.75"), Decimal("1.750"))
    assert numeric_matches(Decimal("30000000"), "30000000.00")


def test_numeric_tolerance_rejects_a_different_covenant() -> None:
    """1.75 and 1.80 are different covenants, and 30m is not 50m."""
    assert not numeric_matches(Decimal("1.75"), Decimal("1.80"))
    assert not numeric_matches(Decimal("30000000"), Decimal("50000000"))


def test_numeric_tolerance_is_relative_and_survives_float_input() -> None:
    # Decimal(1.75) from a float is 1.75000000000000011102230246251565404236316680908203125;
    # accepting it through str() is what keeps the comparison meaningful.
    assert numeric_matches(Decimal("1.75"), 1.75)
    assert numeric_matches(Decimal("1000000"), Decimal("1000001"))
    assert not numeric_matches(Decimal("1.75"), Decimal("1.7599"))


def test_zero_matches_only_zero() -> None:
    assert numeric_matches(Decimal("0"), Decimal("0"))
    assert not numeric_matches(Decimal("0"), Decimal("0.0001"))


def test_non_numeric_never_matches() -> None:
    assert not numeric_matches(Decimal("1"), "not a number")
    assert not numeric_matches(Decimal("1"), None)
    # A bool is an int in Python; treating True as 1 would silently match.
    assert to_decimal(True) is None


# -- date tolerance ---------------------------------------------------------


def test_date_tolerance_absorbs_a_business_day_shift() -> None:
    assert date_matches(dt.date(2028, 6, 15), dt.date(2028, 6, 17))
    assert DEFAULT_DATE_TOLERANCE == dt.timedelta(days=3)


def test_date_tolerance_does_not_merge_two_call_dates() -> None:
    """Consecutive annual call dates must never score as the same date."""
    assert not date_matches(dt.date(2028, 6, 15), dt.date(2029, 6, 15))
    assert not date_matches(dt.date(2028, 6, 15), dt.date(2028, 6, 30))


def test_date_matching_rejects_non_dates() -> None:
    assert not date_matches(dt.date(2028, 6, 15), "2028-06-15")


# -- enums and strings ------------------------------------------------------


def test_enum_comparison_has_no_partial_credit() -> None:
    from app.domain.enums import CovenantType

    assert values_match(CovenantType.GEARING_RATIO, CovenantType.GEARING_RATIO)
    assert not values_match(CovenantType.GEARING_RATIO, CovenantType.INTEREST_COVER)
    assert not values_match(CovenantType.GEARING_RATIO, None)


def test_string_comparison_ignores_case_and_surrounding_space_only() -> None:
    assert values_match("MYR", " myr ")
    assert not values_match("MYR", "USD")


def test_ratio_is_none_on_a_zero_denominator() -> None:
    assert ratio(0, 0) is None
    assert ratio(1, 2) == pytest.approx(0.5)


# -- labels -----------------------------------------------------------------


def test_label_hashes_still_match_the_fixture_builders() -> None:
    """The labels join on sha256, so a changed fixture must fail loudly.

    Without this, editing `synthetic_pdf.py` silently orphans every label: the
    harness reports the documents as "never ingested" and scores nothing, which
    looks like an empty corpus rather than like stale ground truth.
    """
    import hashlib

    from tests.fixtures.synthetic_pdf import (
        build_prospectus,
        build_rating_report,
        build_trust_deed,
    )

    expected = {
        label_module.PROSPECTUS_SHA: build_prospectus,
        label_module.TRUST_DEED_SHA: build_trust_deed,
        label_module.RATING_REPORT_SHA: build_rating_report,
    }
    for sha256, builder in expected.items():
        assert hashlib.sha256(builder()).hexdigest() == sha256


def test_every_label_quotes_text_that_is_in_its_document() -> None:
    """A label whose evidence is not in the document scores nothing for ever.

    Checked with `quotes_labelled_clause`, which refuses the fuzzy leg. Plain
    `verify_quote` passes this test against a label whose threshold has been
    changed to a number the document does not contain -- one digit inside a
    fifty-character phrase is comfortably above a 0.92 partial ratio, and this
    test was written, run green, and only then found to be worthless by
    deliberately corrupting a label to see whether it went red. It did not.
    """
    import hashlib

    import pymupdf

    from app.evals.extraction import quotes_labelled_clause
    from tests.fixtures.synthetic_pdf import CORPUS_BUILDERS

    text_by_sha: dict[str, str] = {}
    for builder in CORPUS_BUILDERS.values():
        data = builder()
        with pymupdf.open(stream=data, filetype="pdf") as document:
            text_by_sha[hashlib.sha256(data).hexdigest()] = "\n".join(
                page.get_text() for page in document
            )

    for label in label_module.COVENANT_LABELS:
        text = text_by_sha[label.document_sha256]
        assert quotes_labelled_clause(label.evidence, text), label.evidence


def test_a_near_miss_citation_is_not_a_citation() -> None:
    """The guard the test above depends on: a changed number must not pass.

    Three documents in this corpus state a gearing covenant at 1.75, 2.25 and
    2.50. A citation check that cannot tell them apart credits an extractor for
    quoting the wrong issuer's covenant.
    """
    from app.evals.extraction import quotes_labelled_clause
    from app.extract.citations import verify_quote

    evidence = "consolidated gearing ratio of not more than 2.25 times"
    wrong_clause = (
        "The Issuer shall maintain a consolidated gearing ratio of not more than 1.75 times."
    )

    assert not quotes_labelled_clause(evidence, wrong_clause)
    # And the reason a stricter check was needed: the pipeline's own gate,
    # correctly for its purpose, does accept this.
    assert verify_quote(evidence, wrong_clause).verified

    right_clause = (
        "The Issuer shall maintain a consolidated gearing\nratio of not more than 2.25 times."
    )
    assert quotes_labelled_clause(evidence, right_clause)


def test_labels_are_unique_enough_to_match_on() -> None:
    """Two labels of the same covenant type in one document need distinct pages."""
    seen: set[tuple[str, str, int]] = set()
    for label in label_module.COVENANT_LABELS:
        key = (label.document_sha256, label.covenant_type.value, label.page_number)
        assert key not in seen, f"ambiguous label: {key}"
        seen.add(key)
