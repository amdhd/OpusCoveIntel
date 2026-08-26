"""Deterministic intent classification.

Phase 7 replaces this with a classifier node in the LangGraph agent. Keyword
rules are enough for Phase 4 and have one property the model version will not:
they are free, and their mistakes are inspectable.

The important class is UNSUPPORTED. This system is decision support, not an
oracle -- it must refuse forecasts, recommendations and anything else the
corpus cannot evidence (docs/plan.md 7, CLAUDE.md 1.5). Refusal is checked *first*,
because "should I buy this sukuk?" contains the word "sukuk" and would
otherwise classify as a perfectly answerable instrument lookup.
"""

from __future__ import annotations

import re
from typing import Final

from app.domain.enums import QueryIntent

# Asking for a prediction, a recommendation, or an opinion. None of these are
# answerable from a corpus of offering documents at any confidence.
_UNSUPPORTED: Final[tuple[str, ...]] = (
    r"\bshould\s+(?:i|we)\b",
    r"\brecommend",
    r"\bwill\b.*\b(?:rally|fall|rise|outperform|default)\b",
    r"\bforecast",
    r"\bpredict",
    r"\bgoing\s+to\s+happen\b",
    r"\bworth\s+buying\b",
    r"\bgood\s+investment\b",
    r"\bfair\s+value\b",
    r"\bprice\s+target\b",
)

_BREACH: Final[tuple[str, ...]] = (
    r"\bbreach",
    r"\bviolat",
    r"\bin\s+compliance\b",
    r"\bcomply\b",
    r"\bcompliant\b",
    r"\bheadroom\b",
    r"\btrip(?:ped|s)?\b",
    r"\btrigger(?:ed)?\s+(?:by|at)\b",
    r"\bwould\s+be\s+triggered\b",
)

_PORTFOLIO: Final[tuple[str, ...]] = (
    r"\bportfolio",
    r"\bexposure\b",
    r"\bholding",
    r"\bnav\b",
    r"\bweight",
    r"\bfund\b",
    r"\bhow\s+much\s+do\s+we\s+(?:hold|own)\b",
)

_COVENANT: Final[tuple[str, ...]] = (
    r"\bcovenant",
    r"\bnegative\s+pledge\b",
    r"\bcross[\s-]?default\b",
    r"\bgearing\b",
    r"\bnisbah\s+gearan\b",
    r"\binterest\s+cover",
    r"\bfinance\s+service\s+cover",
    r"\brating\s+trigger\b",
    r"\bthreshold\b",
    r"\bcall\s+(?:date|price|schedule)\b",
    r"\bredeem\b",
    r"\bredemption\b",
    r"\bshariah\b",
    r"\bdissolution\b",
    r"\bpurchase\s+undertaking\b",
    r"\bevent\s+of\s+default\b",
    r"\bchange\s+of\s+control\b",
)

_INSTRUMENT: Final[tuple[str, ...]] = (
    r"\bissuer\b",
    r"\bmaturity\b",
    r"\bissue\s+size\b",
    r"\bisin\b",
    r"\brated\b",
    r"\brating\s+of\b",
    r"\bwhich\s+instruments?\b",
    r"\bsukuk\s+structure\b",
    r"\bijarah\b|\bwakalah\b|\bmusharakah\b|\bmudharabah\b|\bmurabahah\b|\bistisna",
)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify(question: str) -> QueryIntent:
    """Map a question onto the intent space (docs/plan.md 5)."""
    text = question.lower().strip()
    if not text:
        return QueryIntent.UNSUPPORTED

    # Order matters: refusal first, then the most specific answerable intents.
    if _matches(_UNSUPPORTED, text):
        return QueryIntent.UNSUPPORTED
    if _matches(_BREACH, text):
        return QueryIntent.COVENANT_BREACH_CHECK
    if _matches(_PORTFOLIO, text):
        return QueryIntent.PORTFOLIO_QUERY
    if _matches(_COVENANT, text):
        return QueryIntent.COVENANT_LOOKUP
    if _matches(_INSTRUMENT, text):
        return QueryIntent.INSTRUMENT_LOOKUP
    return QueryIntent.DOCUMENT_SEARCH


def _words(block: str) -> frozenset[str]:
    """A whitespace-separated block of words, as a set.

    A block rather than a list literal: this is a vocabulary to be read and
    edited by a person, and one word per line of quotes obscures what is in it.
    """
    return frozenset(block.split())


# A phrase shorter than this identifies nothing: "sukuk" is in every
# instrument name in the corpus, and "sdn bhd" is in half the issuers.
_MIN_PHRASE_WORDS: Final[int] = 2

_NAME_WORD = re.compile(r"[a-z0-9]+")


def name_words(text: str) -> list[str]:
    """The lower-case alphanumeric words of a name. Public: callers build the
    reserved vocabulary for `mentioned_documents` from it, and two tokenisers
    would disagree the first time one of them changed."""
    return _NAME_WORD.findall(text.lower())


_name_words = name_words


def _phrases(words: list[str], *, minimum: int) -> set[str]:
    """Every contiguous run of `words` of at least `minimum` length."""
    return {
        " ".join(words[start:end])
        for start in range(len(words))
        for end in range(start + minimum, len(words) + 1)
    }


def mentioned_entities(question: str, candidates: list[str]) -> list[str]:
    """Which known names the question names.

    Two ways to name one. The **whole registered name appearing verbatim** is
    the original rule and still the strongest: it cannot be ambiguous, so it
    matches regardless of what else is registered.

    The second exists because people do not quote database columns. The
    instrument is stored as "RM300m Green Ijarah Sukuk" and an analyst asks
    about "the Green Ijarah Sukuk", which the verbatim rule misses -- so the
    answer came back about every instrument in the corpus, on both read paths
    (finding 14). A candidate is also named when the question contains a
    **contiguous run of at least two of its words** that belongs to no other
    candidate.

    Uniqueness is what keeps this as literal as the original. A phrase two
    instruments share names neither of them, and the caller answers about both
    rather than guessing which was meant -- attaching an answer to the wrong
    issuer produces a confident, wrong portfolio number, which is the failure
    mode this system exists to prevent. Over-answering is merely noisy.
    """
    text = question.lower()
    asked = _phrases(_name_words(question), minimum=_MIN_PHRASE_WORDS)

    # Which candidates each phrase could refer to. A phrase claimed by two
    # names identifies neither. Counted over *distinct* names, because callers
    # pass overlapping lists -- two instruments from one issuer put that
    # issuer's name in `candidates` twice, and that must not read as a clash.
    owners: dict[str, set[str]] = {}
    candidate_phrases: list[set[str]] = []
    for name in candidates:
        phrases = _phrases(_name_words(name), minimum=_MIN_PHRASE_WORDS) & asked
        candidate_phrases.append(phrases)
        for phrase in phrases:
            owners.setdefault(phrase, set()).add(name.lower())

    return [
        name
        for name, phrases in zip(candidates, candidate_phrases, strict=True)
        if name.lower() in text or any(len(owners[phrase]) == 1 for phrase in phrases)
    ]


# Words that name a *kind* of document rather than one document. Every corpus
# of offering material is full of them, so a question saying "the prospectus"
# has named nothing -- and answering as though it had is how one document's
# covenants get attributed to another (docs/review.md, finding 15).
_GENERIC_DOCUMENT_WORDS: Final[frozenset[str]] = _words(
    """
    prospectus prospectuses base offering circular memorandum information
    trust deed certificate certificates rating report announcement statement
    supplemental supplement final draft sample scan scanned copy document
    documents file files pdf part annex appendix schedule volume programme
    program terms conditions
    sukuk sukuks bond bonds note notes paper issue issuance release press
    """
)

# Two letters is a file-naming artefact ("v2", "12b"), not a subject.
_MIN_DOCUMENT_WORD_LENGTH: Final[int] = 3


def mentioned_documents(
    question: str, filenames: list[str], *, reserved: set[str] | None = None
) -> list[str]:
    """Which documents the question names, by a word only one of them has.

    A single word is enough here, where `mentioned_entities` insists on two.
    Filenames are not written by the person asking -- nobody types
    "Dubai_12B_Project_Drive_-_Base_Prospectus_1.pdf", they type "the Dubai
    prospectus" -- so a contiguous run of the stored name almost never appears
    in a real question.

    What keeps a single word honest is the same rule as everywhere else:
    **uniqueness**. The word must belong to exactly one document in the corpus
    and must not be a word that names a kind of document. "Dubai" identifies
    one file; "prospectus" and "trust" identify none, however many files
    contain them. A question that names two documents gets both.

    Purely numeric words are ignored: a year or a tranche number in a filename
    is a coincidence waiting to match the wrong question.

    `reserved` is the vocabulary that already means something else -- the words
    in instrument and issuer names. "Sukuk" was unique to one filename in a
    corpus of nine, so "when can the issuer redeem the RM300m Green Ijarah
    Sukuk?" resolved to a press release and was refused for having no covenants.
    A word that names an instrument identifies an instrument, and that is a
    different lookup with its own answer.
    """
    asked = set(_name_words(question))
    off_limits = _GENERIC_DOCUMENT_WORDS | (reserved or set())

    owners: dict[str, set[str]] = {}
    for filename in filenames:
        for word in set(_name_words(filename)):
            if len(word) < _MIN_DOCUMENT_WORD_LENGTH or word.isdigit() or word in off_limits:
                continue
            owners.setdefault(word, set()).add(filename.lower())

    return [
        filename
        for filename in filenames
        if any(
            word in asked and owners.get(word, set()) == {filename.lower()}
            for word in _name_words(filename)
        )
    ]
