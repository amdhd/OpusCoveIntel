"""English / Bahasa Malaysia detection.

Malaysian offering documents mix both languages, often on the same page, and
Postgres ships no Malay stemmer -- so BM text must be indexed under `simple`
while English uses `english` (CLAUDE.md 6). That choice is per chunk, which is
why detection lives at chunk granularity rather than document granularity.

A stopword vote rather than a model: it is free, deterministic, offline, and
the two languages share almost no function words. Words that exist in both
(`dan` is not English, but `is`/`ini` are near-homographs) are excluded.
"""

from __future__ import annotations

import re

from app.domain.enums import Language

_TOKEN = re.compile(r"[a-z']+")

# Function words only -- content words drift with the document type.
_MALAY_MARKERS = frozenset(
    {
        "yang", "dan", "dengan", "untuk", "adalah", "ini", "itu", "pada",
        "dalam", "akan", "tidak", "oleh", "kepada", "daripada", "bagi",
        "atau", "hendaklah", "sekiranya", "boleh", "telah", "juga", "serta",
        "seperti", "kerana", "tersebut", "mana-mana", "syarikat", "perjanjian",
        "tarikh", "nilai", "terbitan", "pemegang", "amaun",
    }
)  # fmt: skip

_ENGLISH_MARKERS = frozenset(
    {
        "the", "and", "of", "to", "in", "for", "shall", "is", "are", "this",
        "that", "with", "be", "by", "or", "any", "such", "which", "from",
        "not", "as", "has", "have", "been", "may", "will", "at", "on",
        "under", "each", "other", "all",
    }
)  # fmt: skip

# Below this, a vote is noise: a two-word heading tells us nothing.
_MIN_MARKER_HITS = 2


def detect_language(text: str) -> Language:
    """Classify a span. Returns UNKNOWN when there is too little evidence."""
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return Language.UNKNOWN

    malay = sum(1 for token in tokens if token in _MALAY_MARKERS)
    english = sum(1 for token in tokens if token in _ENGLISH_MARKERS)

    if malay >= _MIN_MARKER_HITS and malay > english:
        return Language.MS
    if english >= _MIN_MARKER_HITS and english > malay:
        return Language.EN
    return Language.UNKNOWN


def fts_config_for(language: Language) -> str:
    """Map a language onto a Postgres text-search configuration.

    Only `english` and `simple` are permitted by the CHECK constraint on
    `document_chunks.fts_config`. `simple` (no stemming, no stopword list) is
    the honest choice for Malay: a wrong stemmer is worse than none.
    """
    return "simple" if language is Language.MS else "english"


def aggregate_language(languages: list[Language]) -> Language:
    """Document-level language from its chunks."""
    seen = {lang for lang in languages if lang is not Language.UNKNOWN}
    if not seen:
        return Language.UNKNOWN
    if len(seen) > 1:
        return Language.MIXED
    return seen.pop()
