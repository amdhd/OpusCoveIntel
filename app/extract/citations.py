"""Citation verification.

CLAUDE.md 1.3: the model returns `source_quote`; we assert that quote actually
occurs in the cited chunk. Never persist an unverified quote.

Three-leg verification, in order:
1. **Exact match** — the quote appears verbatim in the chunk. Score 1.0.
2. **Normalised match** — differs only in whitespace or lookalike characters
   (smart quotes, dashes, non-breaking spaces). Score 0.99.
3. **Fuzzy match** (Phase 6) — rapidfuzz partial_ratio ≥ 0.92, over quotes of
   at least `MIN_FUZZY_QUOTE_CHARS`. Handles a model that drops a footnote
   marker, inserts a bracketed reference, or normalises formatting the
   normalised leg cannot catch. Offsets come from the alignment, so a fuzzy
   citation still names a span in the stored chunk.

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

# Shortest quote the fuzzy leg will consider. `partial_ratio` matches the best
# substring, so short quotes score near-perfectly against unrelated text and
# the threshold stops meaning anything. Roughly a clause fragment, not a phrase.
MIN_FUZZY_QUOTE_CHARS: Final[int] = 40


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
    normalised legs but pass this one.

    Two properties this leg must have, and did not:

    **A floor on quote length.** `partial_ratio` scores the *best-matching
    substring* of the chunk, so a short quote scores ~1.0 against almost any
    text: "the Issuer" is a perfect partial match for every page of a
    prospectus. Below `MIN_FUZZY_QUOTE_CHARS` the leg is refused outright,
    because a passing score there is evidence of nothing and this is the last
    gate before a covenant is persisted (CLAUDE.md 1.3).

    **Real offsets.** `partial_ratio_alignment` reports which slice of the
    chunk it matched, so a fuzzy citation still points at characters a reviewer
    can open. Returning None offsets — the previous behaviour — persisted a
    clause whose span was unknown, which is precisely what CLAUDE.md 1.2
    forbids.
    """
    if len(quote.strip()) < MIN_FUZZY_QUOTE_CHARS:
        return None

    threshold = get_settings().CITATION_FUZZY_THRESHOLD
    norm_quote = normalise(quote)
    norm_chunk, offsets = _normalise_with_offsets(chunk_text)

    # rapidfuzz scores are 0-100.
    alignment = fuzz.partial_ratio_alignment(norm_quote, norm_chunk, score_cutoff=threshold * 100)
    if alignment is None:
        return None

    score = alignment.score / 100.0
    if score < threshold:
        return None

    # Map the normalised span back onto the original chunk. `offsets` holds the
    # original index of every character kept by normalisation, so the ends
    # translate without guessing at how many characters were collapsed.
    start = offsets[alignment.dest_start] if alignment.dest_start < len(offsets) else None
    end = offsets[alignment.dest_end - 1] + 1 if 0 < alignment.dest_end <= len(offsets) else None
    if start is None or end is None or end <= start:
        return None

    return CitationCheck(True, score, start, end, "fuzzy")


def _normalise_with_offsets(text: str) -> tuple[str, list[int]]:
    """`normalise(text)`, plus each kept character's index in the original.

    Normalisation is done character by character rather than by regex so the
    index map stays exact. NFKC can expand one character into several (a
    ligature becomes two letters); every expanded character maps back to the
    single source index it came from, which is the correct answer for a span.
    """
    kept: list[str] = []
    offsets: list[int] = []
    pending_space = False
    started = False

    for index, char in enumerate(text):
        folded = unicodedata.normalize("NFKC", char).translate(_QUOTE_TRANSLATION)
        for piece in folded:
            if piece.isspace():
                pending_space = started
                continue
            if pending_space:
                kept.append(" ")
                offsets.append(index)
                pending_space = False
            kept.append(piece)
            offsets.append(index)
            started = True

    return "".join(kept), offsets
