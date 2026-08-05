"""Citation verification.

CLAUDE.md 1.3: the model returns `source_quote`; we assert that quote actually
occurs in the cited chunk. Never persist an unverified quote.

Three-leg verification, in order:
1. **Exact match** — the quote appears verbatim in the chunk. Score 1.0.
2. **Normalised match** — differs only in whitespace or lookalike characters
   (smart quotes, dashes, non-breaking spaces). Score 0.99.
3. **Fuzzy match** (Phase 6) — rapidfuzz partial_ratio ≥ 0.92. Handles a model
   that drops a footnote marker, inserts a bracketed reference, or normalises
   formatting the normalised leg cannot catch. Offset is None because the fuzzy
   alignment may not map 1:1 to character positions.

Phase 4's regex extractor never reaches leg 3 — its quotes are literal slices.
Phase 6's LLM extractor exercises all three.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, NamedTuple

from rapidfuzz import fuzz

from app.core.config import get_settings

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

    fuzzy = _find_fuzzy(quote, chunk_text)
    if fuzzy is not None:
        return fuzzy

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


def _find_fuzzy(quote: str, chunk_text: str) -> CitationCheck | None:
    """Verify via rapidfuzz when the model made a minor edit.

    A model that drops a footnote marker ("shall not create any security
    interest[1]") or adds a bracketed reference will fail both the exact and
    normalised legs but pass this one. The score is the rapidfuzz
    partial_ratio scaled to [0, 1].

    Offsets are None for fuzzy matches — the alignment spans the best-matching
    region but the character mapping may not be 1:1.
    """
    threshold = get_settings().CITATION_FUZZY_THRESHOLD
    norm_quote = normalise(quote)
    norm_chunk = normalise(chunk_text)

    # rapidfuzz scores are 0-100.
    score = fuzz.partial_ratio(norm_quote, norm_chunk) / 100.0
    if score >= threshold:
        return CitationCheck(True, score, None, None, "fuzzy")
    return None
