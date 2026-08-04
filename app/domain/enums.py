"""Controlled vocabularies.

CLAUDE.md 7: enums for every controlled vocabulary -- no bare strings for
`clause_type`, `covenant_type` and friends.

These are stored as VARCHAR + CHECK rather than native Postgres ENUM types
(see `app/db/types.py` for the rationale): `clause_type` and `covenant_type`
will keep growing as new deal structures appear, and widening a CHECK is a
plain, transactional migration.
"""

from __future__ import annotations

from enum import StrEnum


class DocumentType(StrEnum):
    PROSPECTUS = "prospectus"
    INFORMATION_MEMORANDUM = "information_memorandum"
    TRUST_DEED = "trust_deed"
    RATING_REPORT = "rating_report"
    ANNOUNCEMENT = "announcement"
    SUPPLEMENTAL = "supplemental"
    UNKNOWN = "unknown"


class SourceType(StrEnum):
    UPLOAD = "upload"
    EMAIL = "email"
    BURSA = "bursa"
    RATING_AGENCY = "rating_agency"
    INTERNAL = "internal"


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    FAILED = "failed"
    # PLAN.md 2: a document abandoned because it hit a budget ceiling is not
    # "failed" -- it is complete work that stopped for a policy reason.
    BUDGET_EXCEEDED = "budget_exceeded"


class Language(StrEnum):
    EN = "en"
    MS = "ms"  # Bahasa Malaysia
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ParseMethod(StrEnum):
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    VLM = "vlm"
    NONE = "none"


class VlmReason(StrEnum):
    """Why a page failed the text-layer confidence checks (CLAUDE.md 4).

    Recorded on `document_pages.vlm_reason` so "why did this document cost $8?"
    is a query. Phase 3 detects these; Phase 5 acts on them.
    """

    NO_TEXT_LAYER = "no_text_layer"
    LOW_CHAR_COUNT = "low_char_count"
    HIGH_IMAGE_AREA = "high_image_area"
    TABLE_EXTRACTION_FAILED = "table_extraction_failed"
    GARBLED_UNICODE = "garbled_unicode"


class ChunkType(StrEnum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


class InstrumentType(StrEnum):
    SUKUK = "sukuk"
    CONVENTIONAL_BOND = "conventional_bond"
    MTN = "mtn"
    PERPETUAL = "perpetual"
    UNKNOWN = "unknown"


class SukukStructureType(StrEnum):
    """Shariah contract underlying the issuance."""

    IJARAH = "ijarah"
    WAKALAH = "wakalah"
    MUSHARAKAH = "musharakah"
    MUDHARABAH = "mudharabah"
    MURABAHAH = "murabahah"
    ISTISNA = "istisna"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class RatingAgency(StrEnum):
    MARC = "MARC"
    RAM = "RAM"
    SP = "SP"
    MOODYS = "MOODYS"
    FITCH = "FITCH"
    UNKNOWN = "unknown"


class ClauseType(StrEnum):
    NEGATIVE_PLEDGE = "negative_pledge"
    CROSS_DEFAULT = "cross_default"
    CHANGE_OF_CONTROL = "change_of_control"
    FINANCIAL_COVENANT = "financial_covenant"
    RATING_TRIGGER = "rating_trigger"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"
    PROFIT_STEP_UP = "profit_step_up"
    DISSOLUTION_EVENT = "dissolution_event"
    PURCHASE_UNDERTAKING = "purchase_undertaking"
    SHARIAH_COMPLIANCE = "shariah_compliance"
    EVENT_OF_DEFAULT = "event_of_default"
    COVENANT_OTHER = "covenant_other"


class CovenantType(StrEnum):
    NEGATIVE_PLEDGE = "negative_pledge"
    CROSS_DEFAULT = "cross_default"
    CHANGE_OF_CONTROL = "change_of_control"
    GEARING_RATIO = "gearing_ratio"
    INTEREST_COVER = "interest_cover"
    FINANCE_SERVICE_COVER = "finance_service_cover"
    MINIMUM_NET_WORTH = "minimum_net_worth"
    RATING_TRIGGER = "rating_trigger"
    DISPOSAL_RESTRICTION = "disposal_restriction"
    DISTRIBUTION_RESTRICTION = "distribution_restriction"
    SHARIAH_NON_COMPLIANCE = "shariah_non_compliance"
    OTHER = "other"


class Severity(StrEnum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriggerDirection(StrEnum):
    DOWNGRADE_BELOW = "downgrade_below"
    UPGRADE_ABOVE = "upgrade_above"
    WITHDRAWAL = "withdrawal"
    ANY_CHANGE = "any_change"


class CallType(StrEnum):
    OPTIONAL = "optional"
    MANDATORY = "mandatory"
    MAKE_WHOLE = "make_whole"
    CLEAN_UP = "clean_up"
    TAX = "tax"
    REGULATORY = "regulatory"


class ExtractionMethod(StrEnum):
    """How a value was obtained. PLAN.md 3 -- rule and llm run in parallel."""

    RULE = "rule"
    LLM = "llm"
    VLM = "vlm"
    HUMAN = "human"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    VALIDATION_FAILED = "validation_failed"
    # CLAUDE.md 1.3: a quote that cannot be found in its cited chunk never
    # reaches a covenant row.
    CITATION_FAILED = "citation_failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"


class ReviewTrigger(StrEnum):
    """Why an item entered the review queue (CLAUDE.md 5)."""

    LOW_CONFIDENCE = "low_confidence"
    RULE_LLM_DISAGREEMENT = "rule_llm_disagreement"
    VALIDATION_RETRY = "validation_retry"
    CITATION_UNVERIFIED = "citation_unverified"
    VLM_SOURCED = "vlm_sourced"
    HIGH_VALUE_THRESHOLD = "high_value_threshold"
    MANUAL = "manual"


class JobType(StrEnum):
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    CLASSIFY = "classify"
    EXTRACT_COVENANT = "extract_covenant"
    EXTRACT_SUKUK_STRUCTURE = "extract_sukuk_structure"
    EXTRACT_CALL_SCHEDULE = "extract_call_schedule"
    EXTRACT_RATING_TRIGGER = "extract_rating_trigger"
    VALIDATE = "validate"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


class LLMStage(StrEnum):
    """Cost attribution bucket for `make cost-report`."""

    CLASSIFY = "classify"
    SECTION_DETECT = "section_detect"
    VLM_OCR = "vlm_ocr"
    EMBED = "embed"
    EXTRACT = "extract"
    VALIDATE = "validate"
    SYNTHESIZE = "synthesize"
    JUDGE = "judge"


class ActorType(StrEnum):
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"


class QueryIntent(StrEnum):
    """PLAN.md 5 -- the LangGraph classifier's output space."""

    DOCUMENT_SEARCH = "document_search"
    COVENANT_LOOKUP = "covenant_lookup"
    INSTRUMENT_LOOKUP = "instrument_lookup"
    PORTFOLIO_QUERY = "portfolio_query"
    COVENANT_BREACH_CHECK = "covenant_breach_check"
    UNSUPPORTED = "unsupported"
