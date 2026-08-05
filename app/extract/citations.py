"""Citation verification.

CLAUDE.md 1.3: the model returns `source_quote`; we assert that quote actually
occurs in the cited chunk. Never persist an unverified quote.

Phase 4's extractor is regex-based, so its quotes are literal slices and always
verify. Running them through this check anyway is the point: the verification
path is exercised from the day it exists, rather than being written for the
first time in Phase 6 against output that can be wrong.

**The normalised-exact leg is here; the fuzzy leg is not.** CLAUDE.md specifies
a ≥0.92 rapidfuzz ratio as the fallback for a model that paraphrases whitespace
or drops a footnote marker. A regex extractor never does that, so adding
rapidfuzz now would be a dependency carrying no weight. `verify_quote` returns
a score, and Phase 6 adds the fuzzy branch behind the same signature.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, NamedTuple

# Ligatures, non-breaking spaces and smart quotes differ between a PDF's text
# layer and anything that has round-tripped through JSON. Normalising them is
# not "loosening" the check -- the characters are the same characters.
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_QUOTE_TRANSLATION: Final[dict[int, str]] = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-",
    0x00A0: " ",
}  # fmt: skip

EXACT_SCORE: Final[float] = 1.0
NORMALISED_SCORE: Final[float] = 0.99


class CitationCheck(NamedTuple):
    verified: bool
    score: float
    char_start: int | None
    char_end: int | None
    method: str


def normalise(text: str) -> str:
    """Collapse the differences that are not differences."""
    folded = unicodedata.normalize("NFKC", text).translate(_QUOTE_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", folded).strip()


def verify_quote(quote: str, chunk_text: str) -> CitationCheck:
    """Check that `quote` occurs in `chunk_text`, and say where.

    Returns offsets into the *original* chunk text, not the normalised form, so
    a verified citation still points at real characters in the stored chunk.
    """
    if not quote.strip():
        return CitationCheck(False, 0.0, None, None, "empty")

    exact = chunk_text.find(quote)
    if exact >= 0:
        return CitationCheck(True, EXACT_SCORE, exact, exact + len(quote), "exact")

    span = _find_normalised(quote, chunk_text)
    if span is not None:
        return CitationCheck(True, NORMALISED_SCORE, span[0], span[1], "normalised")

    return CitationCheck(False, 0.0, None, None, "not_found")


def _find_normalised(quote: str, chunk_text: str) -> tuple[int, int] | None:
    """Locate a quote that differs only in whitespace or lookalike characters.

    Matching is done on a token-wise regex over the original text, so the
    offsets returned index the original string. Rebuilding offsets from a
    normalised copy would be off by however many characters normalisation
    removed -- silently, and only for the documents that needed it.
    """
    tokens = normalise(quote).split()
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens))
    match = pattern.search(chunk_text)
    return (match.start(), match.end()) if match else None
