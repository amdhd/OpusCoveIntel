"""The vocabulary guard on the structured read path.

`app/query/answerable.py` decides whether a question routed to an intent that
answers from rows is one those rows can address. It exists because
`What is the CEO of the issuer paid?` contains the word "issuer", classifies as
`instrument_lookup`, and was answered with a list of instruments at confidence
0.95 and no citations.

Two directions matter equally and are tested as two blocks. Refusing questions
the data cannot address is the fix; **not** refusing ordinary questions is what
keeps the fix from being a regression, and a whitelist is exactly the kind of
guard that quietly over-refuses. The second block is deliberately larger than
the first.
"""

from __future__ import annotations

import pytest

from app.domain.enums import CovenantType, QueryIntent, SukukStructureType
from app.query.answerable import (
    KNOWN_TERMS,
    STRUCTURED_INTENTS,
    refusal_for,
    tokenize,
    unsupported_terms,
)
from app.query.intent import classify

# The names the synthetic corpus holds, as the callers pass them: instrument,
# issuer and portfolio names straight off the rows.
NAMES = (
    "Green Ijarah Sukuk",
    "Synthetic Green Energy Sdn Bhd",
    "Klang Valley Infrastructure Berhad",
    "KV Infra MTN",
    "Green Fixed Income Fund",
    "Balanced Sukuk Fund",
)


def unknown(question: str) -> tuple[str, ...]:
    return unsupported_terms(question, known_names=NAMES)


# -- questions the data cannot address ---------------------------------------


class TestUnanswerableQuestions:
    def test_the_reported_failure(self) -> None:
        """The question from the review, reproduced at the level that decides it."""
        assert unknown("What is the CEO of the issuer paid?") == ("ceo", "paid")

    def test_naming_a_real_issuer_does_not_make_it_answerable(self) -> None:
        """The weaker rule -- "mentions nothing we know" -- misses this one.

        The question names an issuer the database holds *and* a field it holds,
        and is still about executive pay.
        """
        assert unknown("What is the CEO of Synthetic Green Energy Sdn Bhd paid?") == (
            "ceo",
            "paid",
        )

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What is the ESG score of the issuer?", ("esg", "score")),
            ("How many employees does the issuer have?", ("employees",)),
            ("What is the credit spread on the Green Ijarah Sukuk?", ("credit", "spread")),
            ("What is the dividend yield of the issuer?", ("dividend", "yield")),
            ("How much exposure do we have to Indonesian coal?", ("indonesian", "coal")),
            ("Which holdings breach their ESG policy limits?", ("esg", "policy", "limits")),
        ],
    )
    def test_fields_no_column_holds(self, question: str, expected: tuple[str, ...]) -> None:
        assert unknown(question) == expected

    def test_the_refusal_names_what_was_not_understood(self) -> None:
        """ "No supporting evidence" alone reads as "the corpus is thin".

        The truth is that no column holds it at all, and an analyst can only
        act on the difference if the answer says which word failed.
        """
        message = refusal_for(("ceo", "paid"))

        assert "'ceo', 'paid'" in message
        assert "No supporting evidence in the corpus." in message


# -- questions that must still be answered ------------------------------------


class TestOrdinaryQuestionsAreNotRefused:
    """The regression half. A whitelist over-refuses long before it under-refuses."""

    @pytest.mark.parametrize(
        "question",
        [
            "Which instruments are rated below A?",
            "Which instruments mature before 2030?",
            "Who is the issuer of the Green Ijarah Sukuk?",
            "What is the maturity of the Green Ijarah Sukuk?",
            "What is the issue size of the KV Infra MTN?",
            "What is the ISIN for the Green Ijarah Sukuk?",
            "What is the current rating of the KV Infra MTN?",
            "What rating agency rates the Green Ijarah Sukuk?",
            "Which sukuk are wakalah structures?",
            "What is the profit rate of the Green Ijarah Sukuk?",
            "Which instruments are issued by Klang Valley Infrastructure Berhad?",
            "What is the issue size and maturity of every instrument?",
            "How much exposure do we have to Klang Valley Infrastructure Berhad?",
            "What is the total exposure of the Green Fixed Income Fund portfolio?",
            "What are our largest holdings?",
            "Show me the holdings of the Green Fixed Income Fund",
            "Which portfolios hold instruments rated below A-?",
            "What is the nav weight of the KV Infra MTN in the Balanced Sukuk Fund?",
            "Which holdings would breach their rating trigger at the current rating?",
            "Are any of our holdings in breach?",
            "Which instruments would be tripped by a downgrade below BBB-?",
            "Is the Green Fixed Income Fund compliant with its gearing covenants?",
        ],
    )
    def test_an_ordinary_question_has_no_unknown_terms(self, question: str) -> None:
        assert unknown(question) == ()

    def test_an_instrument_can_be_named_the_way_a_person_would(self) -> None:
        """Names are matched word by word, not as whole strings.

        "the RM300m Green Ijarah Sukuk" is the same instrument as
        "Green Ijarah Sukuk", and a question should not have to quote a
        database column to be understood.
        """
        assert unknown("When can the issuer redeem the RM300m Green Ijarah Sukuk?") == ()

    def test_numbers_dates_and_ratings_are_never_salient(self) -> None:
        """`RM30,000,000`, `2028-06-15` and `A-` carry no subject of their own."""
        assert tokenize("RM30,000,000 on 2028-06-15 at A-") == ["rm", "on", "at", "a"]


# -- what the guard is allowed to touch ---------------------------------------


def test_only_the_intents_that_answer_without_evidence_are_guarded() -> None:
    """`covenant_lookup` and `document_search` refuse on their own.

    They answer from retrieved spans, so an unanswerable question already ends
    in a refusal or in a cited passage at low confidence. Guarding them would
    also misjudge Bahasa Malaysia, whose vocabulary is not enumerated here.
    """
    assert STRUCTURED_INTENTS == {
        QueryIntent.INSTRUMENT_LOOKUP,
        QueryIntent.PORTFOLIO_QUERY,
        QueryIntent.COVENANT_BREACH_CHECK,
    }
    assert classify("Apakah nisbah gearan yang perlu dikekalkan oleh penerbit?") not in (
        STRUCTURED_INTENTS
    )


def test_the_controlled_vocabularies_contribute_themselves() -> None:
    """A covenant type added to the enum is understood without editing a list.

    The alternative -- a hand-copied list of covenant names -- goes stale the
    first time someone adds a member, and goes stale as silent over-refusal.
    """
    for covenant_type in CovenantType:
        for word in covenant_type.value.split("_"):
            assert word in KNOWN_TERMS, word
    for structure in SukukStructureType:
        assert structure.value in KNOWN_TERMS, structure.value
