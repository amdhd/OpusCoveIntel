"""The golden question set.

Ten questions an analyst would actually ask, against the synthetic corpus:
the three seeded instruments, two portfolios, and the generated prospectus.

**One of them is unanswerable on purpose.** PLAN.md 8.5 requires the system to
refuse the question it cannot evidence, and a golden set with no refusal case
measures only eagerness. Q10 asks for a market forecast; the correct answer is
a refusal, and answering it fluently would be a failure.

Phase 4's bar is **≥6 of 10 answered with zero LLM calls** (PLAN.md, Phase 4).
Phase 7 raises it to ≥8 of 10 with an agent on top. Keeping the same questions
across both is what makes "did the LLM actually help?" a measurable question
rather than an assumption.

Each case states what a correct answer must contain, not what it must say --
the deterministic path and the Phase 7 agent will word things differently, and
pinning exact prose would make the set brittle without making it stricter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.enums import QueryIntent


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    expected_intent: QueryIntent
    # Substrings that must appear in a correct answer, case-insensitively.
    must_contain: tuple[str, ...] = ()
    # A correct answer must cite at least this many sources.
    min_citations: int = 0
    # True when refusing is the correct outcome.
    expect_refusal: bool = False
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


GOLDEN_QUESTIONS: tuple[GoldenQuestion, ...] = (
    GoldenQuestion(
        id="G01",
        question="What is the cross-default threshold for Synthetic Green Energy Sdn Bhd?",
        expected_intent=QueryIntent.COVENANT_LOOKUP,
        must_contain=("cross_default", "RM30"),
        min_citations=1,
        notes="Threshold is stated as RM30,000,000 and must normalise to a Decimal.",
        tags=("covenant", "money"),
    ),
    GoldenQuestion(
        id="G02",
        question="What gearing ratio must Synthetic Green Energy Sdn Bhd maintain?",
        expected_intent=QueryIntent.COVENANT_LOOKUP,
        must_contain=("gearing_ratio", "1.75"),
        min_citations=1,
        notes="Direction matters: 'not more than', so the operator must be LTE.",
        tags=("covenant", "ratio"),
    ),
    GoldenQuestion(
        id="G03",
        question="Which holdings would breach their rating trigger at the current rating?",
        expected_intent=QueryIntent.COVENANT_BREACH_CHECK,
        must_contain=("rating_trigger",),
        notes=(
            "The flagship query. Requires ordinal comparison -- BBB+ is below A- "
            "even though it sorts after it as a string."
        ),
        tags=("rules", "rating"),
    ),
    GoldenQuestion(
        id="G04",
        question="When can the issuer redeem the RM300m Green Ijarah Sukuk?",
        expected_intent=QueryIntent.COVENANT_LOOKUP,
        must_contain=("2028-06-15",),
        notes="Call schedule extracted from a ruled table, not from prose.",
        tags=("call_schedule", "table"),
    ),
    GoldenQuestion(
        id="G05",
        question="What is the total exposure of the Green Fixed Income Fund portfolio?",
        expected_intent=QueryIntent.PORTFOLIO_QUERY,
        must_contain=("Green Fixed Income Fund", "RM"),
        notes="Portfolio aggregation is SQL, never a model (CLAUDE.md 1.1).",
        tags=("portfolio", "sql"),
    ),
    GoldenQuestion(
        id="G06",
        question="Which instruments are rated below A?",
        expected_intent=QueryIntent.INSTRUMENT_LOOKUP,
        must_contain=("A-", "BBB+"),
        notes=(
            "Ordinal again, and the reason rating_rank is denormalised: a "
            "string comparison would return AA3 as well."
        ),
        tags=("rating", "sql"),
    ),
    GoldenQuestion(
        id="G07",
        question="Does Synthetic Green Energy Sdn Bhd have a negative pledge covenant?",
        expected_intent=QueryIntent.COVENANT_LOOKUP,
        must_contain=("negative_pledge",),
        min_citations=1,
        tags=("covenant",),
    ),
    GoldenQuestion(
        id="G08",
        question="What happens if there is Shariah non-compliance?",
        expected_intent=QueryIntent.COVENANT_LOOKUP,
        must_contain=("shariah",),
        min_citations=1,
        notes=(
            "Shariah non-compliance is a dissolution event triggering a "
            "purchase undertaking -- distinct linked concepts (CLAUDE.md 6)."
        ),
        tags=("sukuk", "shariah"),
    ),
    GoldenQuestion(
        id="G09",
        question="Apakah nisbah gearan yang perlu dikekalkan oleh penerbit?",
        expected_intent=QueryIntent.COVENANT_LOOKUP,
        must_contain=("1.75",),
        notes=(
            "Bahasa Malaysia. The BM chunk indexes under 'simple' because "
            "Postgres has no Malay stemmer, so this exercises that path."
        ),
        tags=("bahasa_malaysia", "retrieval"),
    ),
    GoldenQuestion(
        id="G10",
        question="Should we buy more Malaysian sukuk next quarter?",
        expected_intent=QueryIntent.UNSUPPORTED,
        expect_refusal=True,
        notes=(
            "The one that must be refused. Investment advice is out of scope "
            "(PLAN.md 7) and unevidenced by any corpus."
        ),
        tags=("refusal",),
    ),
)


@dataclass(frozen=True)
class RetrievalCase:
    """A query and a string that identifies the chunk that should be found."""

    query: str
    expected_substring: str
    notes: str = ""


# Retrieval cases span all three synthetic documents deliberately. On a
# single-document corpus both legs return nearly everything, ranks barely
# differ, and any measured "lift" is noise -- retrieval can only be evaluated
# where retrieval has to discriminate.
RETRIEVAL_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        query="cross default threshold for the green energy sukuk",
        expected_substring="RM30,000,000",
        notes="Two issuers have cross-default clauses at different thresholds.",
    ),
    RetrievalCase(
        query="interest cover ratio requirement",
        expected_substring="interest cover ratio of not less than 3.00",
    ),
    RetrievalCase(
        query="gearing covenant in the trust deed",
        expected_substring="2.25",
        notes="Three documents state a gearing covenant, each at a different level.",
    ),
    RetrievalCase(query="nisbah gearan", expected_substring="nisbah gearan"),
    RetrievalCase(query="ketidakpatuhan Shariah", expected_substring="ketidakpatuhan"),
    RetrievalCase(
        query="when may the issuer redeem the notes early",
        expected_substring="call dates",
        notes="Paraphrase: the document says 'redeem ... on any of the call dates'.",
    ),
    RetrievalCase(query="call price schedule", expected_substring="2028-06-15"),
    RetrievalCase(query="concession assets security", expected_substring="concession assets"),
    RetrievalCase(
        query="what happens if the rating is downgraded below BBB-",
        expected_substring="below BBB-",
    ),
    RetrievalCase(query="purchase undertaking dissolution", expected_substring="aku janji"),
    RetrievalCase(
        query="change of control early redemption", expected_substring="change in control"
    ),
    RetrievalCase(
        query="MARC rating rationale outlook stable", expected_substring="outlook is stable"
    ),
)


# PLAN.md, Phase 4 acceptance.
PHASE_4_TARGET = 6
# PLAN.md, Phase 7 acceptance.
PHASE_7_TARGET = 8


def by_id(question_id: str) -> GoldenQuestion:
    for question in GOLDEN_QUESTIONS:
        if question.id == question_id:
            return question
    raise KeyError(question_id)
