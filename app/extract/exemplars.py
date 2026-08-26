"""Canonical phrasings of each clause type, for semantic candidate detection.

docs/plan.md 2 describes candidate narrowing as "regex + Postgres FTS + pgvector kNN
against clause-type exemplars". These are the exemplars: what FTS searches for
and what kNN is compared against.

**Why exemplars rather than more regexes.** A regex fires on the phrasing it was
written for. `app/extract/patterns.py` is deliberately precise -- docs/plan.md is
blunt that a pattern firing on anything containing "shall not" is worse than
nothing -- and precision bought that way costs recall on every paraphrase. An
exemplar is not matched literally: FTS scores a chunk on how many of its terms
appear and how densely, and kNN on vector proximity. A clause that says
"undertakes not to encumber" reaches the model through the negative-pledge
exemplar without anybody having written a regex for "encumber".

**These are search queries, not extraction rules.** Nothing here is asserted
about a document. A false positive costs one LLM call on a chunk that turns out
to hold no covenant; a false negative means a covenant is never seen by the
model at all. So they are written to be broad where the patterns are narrow --
the opposite bias, on purpose, because the regex leg already covers precision.

Both languages, because a Malaysian trust deed states the same covenant in
English and Bahasa Malaysia and the FTS config is chosen per chunk (CLAUDE.md 6).
"""

from __future__ import annotations

from typing import Final

from app.domain.enums import ClauseType

# One or more canonical phrasings per clause type. Kept as prose rather than
# keyword lists: `ts_rank_cd` rewards term density, and the embedder is a
# bag-of-words, so a natural sentence carries more signal than a bare term.
CLAUSE_EXEMPLARS: Final[dict[ClauseType, tuple[str, ...]]] = {
    ClauseType.NEGATIVE_PLEDGE: (
        "The Issuer shall not create or permit to subsist any security interest, "
        "mortgage, charge, pledge, lien or encumbrance over its assets or revenues, "
        "other than permitted security interests, without the prior written consent "
        "of the Trustee.",
        "Penerbit tidak boleh mewujudkan sebarang kepentingan sekuriti, gadaian atau "
        "bebanan ke atas asetnya tanpa kebenaran bertulis Pemegang Amanah.",
    ),
    ClauseType.CROSS_DEFAULT: (
        "An event of default occurs if any indebtedness of the Issuer or any "
        "material subsidiary becomes due and payable prior to its stated maturity, "
        "or is not paid when due, in an aggregate principal amount exceeding the "
        "specified threshold.",
        "Kejadian keingkaran berlaku sekiranya mana-mana hutang Penerbit menjadi "
        "matang lebih awal atau tidak dibayar apabila tiba tempohnya.",
    ),
    ClauseType.CHANGE_OF_CONTROL: (
        "A change in control occurs where any person or persons acting in concert "
        "acquire beneficial ownership of the voting share capital of the Issuer, or "
        "the existing shareholder ceases to control the Issuer.",
        "Pertukaran kawalan berlaku apabila mana-mana pihak memperoleh pemilikan "
        "modal saham mengundi Penerbit.",
    ),
    ClauseType.FINANCIAL_COVENANT: (
        "The Issuer shall at all times maintain a consolidated gearing ratio of not "
        "more than the stated multiple, a finance service cover ratio of not less "
        "than the stated multiple, and a consolidated net worth or shareholders' "
        "funds of not less than the stated amount, tested semi-annually by reference "
        "to its audited financial statements.",
        "Penerbit hendaklah pada setiap masa mengekalkan nisbah gearan yang tidak "
        "melebihi kadar yang dinyatakan dan nilai bersih yang tidak kurang daripada "
        "jumlah yang dinyatakan.",
    ),
    ClauseType.RATING_TRIGGER: (
        "In the event the rating assigned to the Sukuk by MARC or RAM is downgraded, "
        "reduced, lowered below the specified rating, placed on negative watch or "
        "withdrawn, the Issuer shall notify the Trustee and procure additional "
        "security.",
        "Sekiranya penarafan yang diberikan kepada Sukuk diturunkan di bawah "
        "penarafan yang dinyatakan, Penerbit hendaklah memaklumkan Pemegang Amanah.",
    ),
    ClauseType.CALL_OPTION: (
        "The Issuer may at its option redeem the Sukuk in whole or in part on any "
        "call date at the call price set out in the redemption schedule, including "
        "optional, make-whole, clean-up, tax and regulatory redemption.",
        "Penerbit boleh atas pilihannya menebus Sukuk pada tarikh panggilan pada "
        "harga panggilan yang dinyatakan.",
    ),
    ClauseType.PUT_OPTION: (
        "Each holder may require the Issuer to repurchase or redeem its Sukuk at "
        "the put price upon the occurrence of a put event.",
    ),
    ClauseType.PROFIT_STEP_UP: (
        "The profit rate shall step up by the specified margin if the Issuer fails "
        "to satisfy the stated condition, and the increased rate accrues until the "
        "condition is remedied.",
    ),
    ClauseType.DISSOLUTION_EVENT: (
        "Upon the occurrence of a dissolution event the Trustee may declare the "
        "Sukuk immediately due and payable and the Trust Assets shall be dissolved.",
        "Apabila berlakunya kejadian pembubaran, Pemegang Amanah boleh mengisytiharkan "
        "Sukuk perlu dibayar dengan serta-merta.",
    ),
    ClauseType.PURCHASE_UNDERTAKING: (
        "The Issuer irrevocably undertakes to purchase the Trust Assets from the "
        "Trustee at the exercise price upon a dissolution event, pursuant to the "
        "purchase undertaking.",
        "Penerbit dengan ini memberi aku janji pembelian untuk membeli Aset Amanah "
        "pada harga pelaksanaan.",
    ),
    ClauseType.SHARIAH_COMPLIANCE: (
        "In the event of Shariah non-compliance, as determined by the Shariah "
        "adviser, the transaction shall constitute a dissolution event and the "
        "purchase undertaking shall be exercised.",
        "Sekiranya berlaku ketidakpatuhan Shariah, ia adalah suatu kejadian "
        "pembubaran dan Penerbit hendaklah melaksanakan aku janji pembelian.",
    ),
    ClauseType.EVENT_OF_DEFAULT: (
        "Each of the following constitutes an event of default: non-payment of any "
        "amount when due, breach of any covenant or obligation, insolvency, "
        "winding-up, appointment of a receiver, or cessation of business.",
        "Kejadian keingkaran termasuk kegagalan membayar amaun yang perlu dibayar, "
        "pelanggaran waad, kebankrapan atau penggulungan.",
    ),
    ClauseType.COVENANT_OTHER: (
        "The Issuer covenants and undertakes that it shall comply with all "
        "obligations, restrictions on disposals of assets and restrictions on the "
        "declaration or payment of dividends and distributions.",
    ),
}


def exemplar_queries() -> tuple[tuple[ClauseType, str], ...]:
    """Every exemplar, flattened, paired with the clause type it stands for.

    Flat because both retrieval legs run one query at a time and the clause type
    travels with the result as a hint on the resulting candidate.
    """
    return tuple(
        (clause_type, text) for clause_type, texts in CLAUSE_EXEMPLARS.items() for text in texts
    )
