"""Intent classification.

The class that matters is UNSUPPORTED. Everything else degrades gracefully --
a covenant question misrouted to document search still returns cited text --
but a forecast answered confidently is the failure mode that discredits the
whole system (docs/plan.md 7).
"""

from __future__ import annotations

import pytest

from app.domain.enums import QueryIntent
from app.query.intent import classify, mentioned_entities


@pytest.mark.parametrize(
    "question",
    [
        "Should we buy more Malaysian sukuk next quarter?",
        "Should I sell the Wakalah sukuk?",
        "Will the sukuk market rally next year?",
        "What is your forecast for MGS yields?",
        "Can you predict the next rating action?",
        "Is this a good investment?",
        "What is the fair value of the RM300m Green Ijarah Sukuk?",
        "Do you recommend the Retail REIT sukuk?",
    ],
)
def test_advice_and_forecasts_are_unsupported(question: str) -> None:
    assert classify(question) is QueryIntent.UNSUPPORTED


def test_refusal_beats_a_perfectly_answerable_looking_question() -> None:
    # Contains "sukuk" and "rated", which would otherwise route to an
    # instrument lookup and produce a confident, useless answer.
    question = "Should we buy the sukuk rated A- by MARC?"

    assert classify(question) is QueryIntent.UNSUPPORTED


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which holdings breach their covenants?", QueryIntent.COVENANT_BREACH_CHECK),
        (
            "Is the issuer in compliance with its gearing covenant?",
            QueryIntent.COVENANT_BREACH_CHECK,
        ),
        ("How much headroom is there on the gearing covenant?", QueryIntent.COVENANT_BREACH_CHECK),
        ("What is the total exposure of the Green Fixed Income Fund?", QueryIntent.PORTFOLIO_QUERY),
        ("How much do we hold of the Wakalah sukuk?", QueryIntent.PORTFOLIO_QUERY),
        ("What is the cross-default threshold?", QueryIntent.COVENANT_LOOKUP),
        ("Apakah nisbah gearan yang perlu dikekalkan?", QueryIntent.COVENANT_LOOKUP),
        ("When can the issuer redeem the sukuk?", QueryIntent.COVENANT_LOOKUP),
        ("Which instruments are rated below A?", QueryIntent.INSTRUMENT_LOOKUP),
        ("What is the maturity of the Wakalah sukuk?", QueryIntent.INSTRUMENT_LOOKUP),
        ("Tell me about the trustee arrangements", QueryIntent.DOCUMENT_SEARCH),
    ],
)
def test_answerable_questions_route_to_the_right_intent(
    question: str, expected: QueryIntent
) -> None:
    assert classify(question) is expected


def test_an_empty_question_is_unsupported_rather_than_a_search() -> None:
    assert classify("   ") is QueryIntent.UNSUPPORTED


def test_a_breach_question_outranks_the_covenant_words_it_contains() -> None:
    # "gearing covenant" alone is a lookup; asking whether it is breached is a
    # rules-engine call, and the two produce very different answers.
    assert classify("Is the gearing covenant breached?") is QueryIntent.COVENANT_BREACH_CHECK


def test_entity_matching_is_literal() -> None:
    names = ["Synthetic Green Energy Sdn Bhd", "Synthetic Retail REIT Berhad"]

    found = mentioned_entities("What covenants apply to synthetic green energy sdn bhd?", names)

    assert found == ["Synthetic Green Energy Sdn Bhd"]


def test_an_unmentioned_entity_is_not_matched() -> None:
    # Attaching an answer to the wrong issuer produces a confident, wrong
    # portfolio number -- worse than returning nothing.
    assert mentioned_entities("What covenants apply?", ["Synthetic Retail REIT Berhad"]) == []


# -- naming something without quoting the database ---------------------------
#
# Nobody types "RM300m Green Ijarah Sukuk". They type "the Green Ijarah Sukuk",
# and the verbatim rule missed it -- so both read paths answered a question
# about one instrument with every instrument in the corpus (finding 14).

INSTRUMENTS = [
    "RM300m Green Ijarah Sukuk",
    "RM500m Wakalah Sukuk",
    "RM250m Retail REIT Sukuk",
]


def test_a_partial_name_identifies_the_instrument() -> None:
    assert mentioned_entities("Who is the issuer of the Green Ijarah Sukuk?", INSTRUMENTS) == [
        "RM300m Green Ijarah Sukuk"
    ]


def test_a_word_every_candidate_shares_names_none_of_them() -> None:
    """ "Sukuk" is in all three names, so it identifies nothing.

    This is the case the phrase rule must not get wrong: matching here would
    attach an answer about one instrument to whichever row sorted first.
    """
    assert mentioned_entities("What is the maturity of the sukuk?", INSTRUMENTS) == []


def test_a_phrase_two_candidates_share_names_neither() -> None:
    """Ambiguity is answered broadly, never resolved by guessing.

    Two tranches of one programme differ only by year. A question that does not
    say which one gets both, which is noisy; picking one would be wrong.
    """
    tranches = ["Green Ijarah Sukuk 2030", "Green Ijarah Sukuk 2032"]

    assert mentioned_entities("When does the Green Ijarah Sukuk mature?", tranches) == []
    assert mentioned_entities("When does the Green Ijarah Sukuk 2032 mature?", tranches) == [
        "Green Ijarah Sukuk 2032"
    ]


def test_one_issuer_appearing_twice_does_not_clash_with_itself() -> None:
    """Callers pass overlapping lists.

    Two instruments from one issuer put that issuer's name in `candidates`
    twice. Counting phrase owners by position rather than by name would read
    that as ambiguity and silently stop narrowing.
    """
    candidates = ["Synthetic Green Energy Sdn Bhd", "Synthetic Green Energy Sdn Bhd"]

    found = mentioned_entities("What does Synthetic Green Energy hold?", candidates)

    assert found == candidates


def test_a_single_word_of_a_name_is_never_enough() -> None:
    """One word is not a name. "Green" is a fund, an instrument and a colour."""
    assert mentioned_entities("Which green instruments do we hold?", INSTRUMENTS) == []
