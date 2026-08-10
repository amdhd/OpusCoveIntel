"""Which questions the structured read path is entitled to answer.

**The failure this exists to stop.** `What is the CEO of the issuer paid?`
contains the word "issuer", so the intent classifier routes it to
`instrument_lookup` -- an intent answered from `instruments` rows without
retrieval. The refusal path is reached only when retrieval finds nothing, so it
was never reached at all: the system replied with a list of instruments at
confidence 0.95 and no citations. Executive compensation appears nowhere in the
corpus and no column holds it. A confident, uncited answer to a question the
data cannot address is the exact output this system exists not to produce
(CLAUDE.md 1.5).

**The rule.** A question routed to a structured intent is answerable only if
every *salient* word in it is one the system has a meaning for -- a field it
holds, a controlled vocabulary it knows, or the name of something in the
database. One unknown salient word and the answer is a refusal that names the
word.

That is deliberately stricter than "does the question mention anything we
know". The weaker rule cannot see this failure: "issuer" *is* a field, so the
CEO question mentions something known and would still be answered. It is also
blind to `What is the CEO of Synthetic Green Energy Sdn Bhd paid?`, which names
a real issuer.

**Why a whitelist and not a blocklist.** A list of out-of-scope topics (pay,
share price, ESG score, headcount) is a list that is never finished, and each
missing entry is a confident wrong answer. A whitelist fails the other way: an
unrecognised phrasing produces a refusal that names what it did not understand.
Refusing an answerable question is visible and recoverable; answering an
unanswerable one is neither.

**Scope: the three structured intents only.** `covenant_lookup` and
`document_search` answer from retrieved spans and already refuse when nothing
is retrieved, so the guard would add nothing and would misjudge the Bahasa
Malaysia questions, whose vocabulary is not enumerated here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from app.domain.enums import (
    CallType,
    ClauseType,
    CovenantType,
    InstrumentType,
    QueryIntent,
    RatingAgency,
    SukukStructureType,
    TriggerDirection,
)

# Answered from rows and rules rather than from retrieved spans, so nothing
# downstream can notice that the question was about something else.
STRUCTURED_INTENTS: Final[frozenset[QueryIntent]] = frozenset(
    {
        QueryIntent.INSTRUMENT_LOOKUP,
        QueryIntent.PORTFOLIO_QUERY,
        QueryIntent.COVENANT_BREACH_CHECK,
    }
)


def _words(block: str) -> frozenset[str]:
    """A whitespace-separated block of words, as a set.

    A block rather than a list literal: these are vocabularies to be read and
    edited by a person, and one word per line of quotes obscures what is in
    them.
    """
    return frozenset(block.split())


# Grammar, not subject matter. A word here carries no claim about the data, so
# its presence neither supports nor blocks an answer.
_STOPWORDS: Final[frozenset[str]] = _words(
    """
    a an the this that these those there here it its
    i we us our you your they them their he she his her
    is are was were be been being am
    do does did done doing
    have has had having
    can could will would shall may might must
    what which who whom whose when where why how
    and or not no nor but if then else so as than
    of for to in on at by with from into about across over under
    up down out off between per within
    any some each every both all
    much many more less most least few several
    please tell show give list find get me
    right now still yet also just only same other another
    does need want know say says said
    """
)

# Words that describe *how* a set is sliced rather than what is in it. Known,
# because the structured branches implement them: a rating threshold filter, a
# sum over holdings, an as-of date.
_QUERY_TERMS: Final[frozenset[str]] = _words(
    """
    above below under over worse better higher lower
    before after since until prior earlier later next last
    largest biggest smallest highest top bottom first
    total sum aggregate count number amount value
    current currently latest today outstanding
    breach breached breaching violate violates violated violation
    comply complies compliant compliance headroom trip trips tripped
    trigger triggers triggered triggering
    """
)

# The structured surface: what `instruments`, `portfolios`, `portfolio_holdings`
# and the rules engine can actually answer about. Every term here corresponds to
# a column the answer formatters read or a filter they apply -- if one stops
# being true, a question about it should start being refused.
_SCHEMA_TERMS: Final[frozenset[str]] = _words(
    """
    instrument instruments bond bonds sukuk note notes paper security securities
    issuer issuers obligor name named
    rating ratings rated rate rates agency agencies notch grade
    downgrade downgraded upgrade upgraded withdrawn
    maturity mature matures matured maturing tenor date dates due
    issue issued issuing size issuance nominal principal denomination
    isin ticker currency myr rm ringgit
    structure structures type types
    portfolio portfolios fund funds mandate holding holdings position positions
    hold holds held own owns owned
    exposure weight weighting nav quantity market
    covenant covenants clause clauses default defaults ratio ratios threshold thresholds
    call calls callable redeem redemption redeemable price schedule
    profit coupon distribution
    """
)

# Controlled vocabularies, contributed by the enums themselves so a new member
# is understood the day it is added rather than the day someone remembers this
# file.
_ENUM_TERMS: Final[frozenset[str]] = frozenset(
    word
    for enum_cls in (
        CovenantType,
        ClauseType,
        InstrumentType,
        SukukStructureType,
        RatingAgency,
        TriggerDirection,
        CallType,
    )
    for member in enum_cls
    for word in str(member.value).lower().replace("_", " ").split()
)

# Rating letters, so "rated below BBB+" reads as a rating rather than as three
# unknown words. The `+`/`-` and any numeric suffix are stripped by tokenising.
_RATING_LETTERS: Final[frozenset[str]] = frozenset({"aaa", "aa", "bbb", "bb", "ccc", "cc", "id"})

KNOWN_TERMS: Final[frozenset[str]] = (
    _STOPWORDS | _QUERY_TERMS | _SCHEMA_TERMS | _ENUM_TERMS | _RATING_LETTERS
)

# Splits on anything that is not a letter, so numbers, money, dates and the
# `+`/`-` of a rating never reach the vocabulary check. `RM30,000,000`,
# `2028-06-15` and `A-` carry no subject matter of their own.
_WORD = re.compile(r"[a-z]+")

# One-letter tokens are ratings (`A`), initials, or noise, and are never the
# thing a question is about.
_MIN_SALIENT_LENGTH: Final[int] = 2


def tokenize(text: str) -> list[str]:
    """Lower-case alphabetic words, in order, duplicates kept."""
    return _WORD.findall(text.lower())


def _name_terms(names: Iterable[str]) -> frozenset[str]:
    """Every word appearing in a name the database holds.

    Word-level rather than whole-name: "the Green Ijarah Sukuk" and "Green
    Ijarah Sukuk due 2030" are the same subject, and a question is free to name
    an instrument the way a person would.
    """
    return frozenset(word for name in names for word in tokenize(name))


def unsupported_terms(question: str, *, known_names: Iterable[str] = ()) -> tuple[str, ...]:
    """Salient words in `question` that the structured path has no meaning for.

    Empty means every word is accounted for. Order follows the question, and
    duplicates are dropped, so the result reads back as a list of what was not
    understood.
    """
    known = KNOWN_TERMS | _name_terms(known_names)
    unknown: list[str] = []
    for word in tokenize(question):
        if len(word) < _MIN_SALIENT_LENGTH or word in known or word in unknown:
            continue
        unknown.append(word)
    return tuple(unknown)


def refusal_for(terms: Iterable[str]) -> str:
    """The refusal text, naming what was not understood.

    Naming the words matters: "no supporting evidence" alone reads as "the
    corpus is missing something", when the truth is that the system holds no
    such field at all. An analyst who sees *which* word failed can rephrase, or
    conclude that this is not the tool for that question.
    """
    quoted = ", ".join(f"'{term}'" for term in terms)
    return (
        "No supporting evidence in the corpus. This question asks about "
        f"{quoted}, which is not something the extracted data records."
    )
