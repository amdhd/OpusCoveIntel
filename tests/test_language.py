"""Language detection and the text-search configuration it selects.

Getting this wrong is not cosmetic: an English stemmer applied to Malay turns
"kejadian" and "keingkaran" into nonsense stems, and the clause becomes
unfindable (CLAUDE.md 6).
"""

from __future__ import annotations

from app.domain.enums import Language
from app.ingest.language import aggregate_language, detect_language, fts_config_for

ENGLISH = (
    "The Issuer shall not create any security interest over the whole or any "
    "part of its present or future assets without the consent of the Trustee."
)
MALAY = (
    "Penerbit hendaklah pada setiap masa mengekalkan nisbah gearan yang tidak "
    "melebihi 1.75 kali seperti yang dinyatakan dalam perjanjian ini."
)


def test_detects_english() -> None:
    assert detect_language(ENGLISH) is Language.EN


def test_detects_bahasa_malaysia() -> None:
    assert detect_language(MALAY) is Language.MS


def test_reports_unknown_rather_than_guessing_on_thin_evidence() -> None:
    # A heading carries no function words to vote with.
    assert detect_language("REDEMPTION AND CALL SCHEDULE") is Language.UNKNOWN
    assert detect_language("") is Language.UNKNOWN
    assert detect_language("RM30,000,000") is Language.UNKNOWN


def test_malay_selects_the_simple_configuration() -> None:
    # Postgres ships no Malay stemmer; no stemming beats the wrong stemming.
    assert fts_config_for(Language.MS) == "simple"


def test_english_and_unknown_select_the_english_configuration() -> None:
    assert fts_config_for(Language.EN) == "english"
    assert fts_config_for(Language.UNKNOWN) == "english"


def test_configurations_are_limited_to_what_the_check_constraint_permits() -> None:
    permitted = {"english", "simple"}

    assert {fts_config_for(language) for language in Language} <= permitted


def test_a_document_holding_both_languages_is_mixed() -> None:
    assert aggregate_language([Language.EN, Language.MS]) is Language.MIXED
    assert aggregate_language([Language.EN, Language.UNKNOWN]) is Language.EN
    assert aggregate_language([Language.UNKNOWN]) is Language.UNKNOWN
    assert aggregate_language([]) is Language.UNKNOWN
