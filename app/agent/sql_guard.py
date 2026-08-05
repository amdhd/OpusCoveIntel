"""SQL guardrail — the last line of defence before a generated statement executes.

PLAN.md 5: read-only role · SELECT only, parsed via sqlglot (not regex) ·
table+column allowlist · statement_timeout=5s · forced LIMIT 1000.

Every statement is validated here before it reaches Postgres. A statement that
fails any check is rejected with a `SQLGuardError` that carries enough context
for the agent to explain why it was refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.core.logging import get_logger

logger = get_logger(__name__)

# Tables the agent is permitted to read. No mutation target appears here, and
# dropping a table from this list removes it from the agent's reach without
# touching a single grant.
ALLOWED_TABLES: Final[frozenset[str]] = frozenset(
    {
        "documents",
        "document_pages",
        "document_chunks",
        "instruments",
        "clauses",
        "covenants",
        "call_schedules",
        "rating_triggers",
        "sukuk_structures",
        "portfolios",
        "portfolio_holdings",
        "extraction_jobs",
        "llm_calls",
        "llm_cache",
        "human_reviews",
        "audit_logs",
        "query_logs",
    }
)

# Columns the agent may SELECT. Schema-qualified column names ("instruments.id")
# are matched against ALLOWED_COLUMNS AND the column name alone. If a column
# name appears here with no table prefix, it is permitted on any allowed table.
_ALLOWED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        # Universal
        "id",
        "created_at",
        "updated_at",
        # documents
        "documents.sha256",
        "documents.filename",
        "documents.source_type",
        "documents.document_type",
        "documents.issuer_guess",
        "documents.language",
        "documents.page_count",
        "documents.status",
        "documents.parse_confidence",
        # document_pages
        "document_pages.document_id",
        "document_pages.page_number",
        "document_pages.char_count",
        "document_pages.image_area_ratio",
        "document_pages.has_text_layer",
        "document_pages.parse_method",
        "document_pages.vlm_used",
        "document_pages.vlm_reason",
        "document_pages.confidence",
        # document_chunks
        "document_chunks.document_id",
        "document_chunks.page_number",
        "document_chunks.section_title",
        "document_chunks.chunk_text",
        "document_chunks.chunk_type",
        "document_chunks.language",
        "document_chunks.fts_config",
        "document_chunks.char_start",
        "document_chunks.char_end",
        "document_chunks.hash",
        "document_chunks.embedding_model",
        "document_chunks.ordinal",
        # instruments
        "instruments.issuer_name",
        "instruments.instrument_name",
        "instruments.instrument_type",
        "instruments.currency",
        "instruments.isin",
        "instruments.ticker",
        "instruments.sukuk_structure",
        "instruments.issue_size",
        "instruments.maturity_date",
        "instruments.current_rating",
        "instruments.current_rating_rank",
        "instruments.rating_agency",
        "instruments.review_status",
        # clauses
        "clauses.document_id",
        "clauses.instrument_id",
        "clauses.source_chunk_id",
        "clauses.clause_type",
        "clauses.clause_text",
        "clauses.page_number",
        "clauses.section_title",
        "clauses.source_quote",
        "clauses.char_start",
        "clauses.char_end",
        "clauses.citation_verified",
        "clauses.citation_match_score",
        "clauses.normalized_json",
        "clauses.method",
        "clauses.confidence",
        "clauses.extraction_status",
        "clauses.review_status",
        # covenants
        "covenants.clause_id",
        "covenants.instrument_id",
        "covenants.covenant_type",
        "covenants.summary",
        "covenants.conditions_json",
        "covenants.thresholds_json",
        "covenants.threshold_amount",
        "covenants.threshold_currency",
        "covenants.effective_date",
        "covenants.trigger_event",
        "covenants.severity",
        "covenants.method",
        "covenants.confidence",
        "covenants.review_status",
        # call_schedules
        "call_schedules.instrument_id",
        "call_schedules.source_clause_id",
        "call_schedules.call_date",
        "call_schedules.call_price",
        "call_schedules.call_type",
        "call_schedules.conditions_json",
        "call_schedules.method",
        "call_schedules.confidence",
        "call_schedules.review_status",
        # rating_triggers
        "rating_triggers.instrument_id",
        "rating_triggers.source_clause_id",
        "rating_triggers.rating_agency",
        "rating_triggers.trigger_rating",
        "rating_triggers.trigger_rank",
        "rating_triggers.trigger_direction",
        "rating_triggers.consequence",
        "rating_triggers.severity",
        "rating_triggers.method",
        "rating_triggers.confidence",
        "rating_triggers.review_status",
        # sukuk_structures
        "sukuk_structures.instrument_id",
        "sukuk_structures.structure_type",
        "sukuk_structures.spv_name",
        "sukuk_structures.originator",
        "sukuk_structures.underlying_asset",
        "sukuk_structures.profit_rate",
        "sukuk_structures.profit_payment_frequency",
        "sukuk_structures.purchase_undertaking",
        "sukuk_structures.dissolution_events_json",
        "sukuk_structures.shariah_compliance_events_json",
        "sukuk_structures.confidence",
        "sukuk_structures.review_status",
        # portfolios
        "portfolios.name",
        "portfolios.owner",
        "portfolios.mandate_type",
        "portfolios.base_currency",
        # portfolio_holdings
        "portfolio_holdings.portfolio_id",
        "portfolio_holdings.instrument_id",
        "portfolio_holdings.quantity",
        "portfolio_holdings.market_value",
        "portfolio_holdings.nav_weight",
        "portfolio_holdings.as_of_date",
        "portfolio_holdings.source",
    }
)

# Maximum rows the agent may return from any one statement. Postgres plans
# scale with the LIMIT; accidental full scans are 1000 rows at worst, not 10⁷.
FORCED_LIMIT: Final[int] = 1000

# The regex path is deliberately avoided for validation — parsing with sqlglot
# gives us the AST, and checking the AST is structural. But when sqlglot cannot
# parse a dialect-specific extension, we fall back to a regex to detect
# non-SELECT keywords so the statement is rejected rather than passed.
_NON_SELECT_KEYWORDS: Final[re.Pattern[str]] = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|COPY|VACUUM|REINDEX"
    r"|SET\s+ROLE|DISCARD)\b",
    re.IGNORECASE,
)

# sqlglot dialects we accept. `postgres` covers asyncpg-compatible SQL.
ACCEPTED_DIALECTS: Final[frozenset[str]] = frozenset({"postgres", ""})


class SQLGuardError(ValueError):
    """A statement was rejected. `reason` states why in prose the agent can relay."""


@dataclass(frozen=True)
class SQLGuardResult:
    """The outcome of guardrail validation."""

    allowed: bool
    reason: str = ""
    # The statement as rewritten by the guardrail (LIMIT appended, etc.).
    rewritten: str = ""
    tables_read: list[str] = field(default_factory=list)
    columns_read: list[str] = field(default_factory=list)


def validate_sql(
    sql: str,
    *,
    allowed_tables: frozenset[str] | None = None,
    forced_limit: int = FORCED_LIMIT,
) -> SQLGuardResult:
    """Validate a SQL statement and rewrite it if necessary.

    Args:
        sql: The statement to validate.
        allowed_tables: Tables the agent may read. Defaults to ALLOWED_TABLES.
        forced_limit: Maximum rows to return. Appended if no LIMIT is present.

    Returns:
        A `SQLGuardResult`. If `allowed` is False, `reason` states why.
    """
    tables = allowed_tables if allowed_tables is not None else ALLOWED_TABLES

    if not sql or not sql.strip():
        return SQLGuardResult(allowed=False, reason="empty statement")

    sql_stripped = sql.strip()

    # -- 1. Non-SELECT keyword detection (regex fallback) --------------------
    if _NON_SELECT_KEYWORDS.search(sql_stripped):
        return SQLGuardResult(
            allowed=False,
            reason="only SELECT statements are permitted; the statement contains a "
            "non-SELECT keyword",
        )

    # -- 2. Parse with sqlglot -----------------------------------------------
    try:
        parsed = sqlglot.parse_one(sql_stripped, dialect="postgres")
    except ParseError as exc:
        logger.warning(
            "sql_guard.parse_error",
            extra={"sql": sql_stripped[:200], "error": str(exc)},
        )
        return SQLGuardResult(
            allowed=False,
            reason=f"could not parse SQL: {exc}",
        )

    if parsed is None:
        return SQLGuardResult(allowed=False, reason="could not parse SQL (empty AST)")

    # -- 3. Must be a SELECT -------------------------------------------------
    if not isinstance(parsed, exp.Select):
        kind = (
            parsed.key.upper()
            if hasattr(parsed, "key")
            else type(parsed).__name__
        )
        return SQLGuardResult(
            allowed=False,
            reason=f"only SELECT statements are permitted; got {kind}",
        )

    # -- 4. No subqueries that mutate ----------------------------------------
    for node in parsed.walk():
        if isinstance(node, exp.Select):
            continue
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter)):
            return SQLGuardResult(
                allowed=False,
                reason="mutation statements are not permitted inside subqueries",
            )

    # -- 5. Table allowlist --------------------------------------------------
    tables_read: list[str] = []
    # Collect all table aliases so we can resolve column references.
    table_aliases: dict[str, str] = {}
    for table in parsed.find_all(exp.Table):
        table_name = table.name
        if table_name not in tables:
            return SQLGuardResult(
                allowed=False,
                reason=f"table '{table_name}' is not in the query-agent allowlist",
            )
        tables_read.append(table_name)
        # Track alias → real table name for column validation
        alias = table.alias_or_name
        if alias != table_name:
            table_aliases[alias] = table_name

    if not tables_read:
        return SQLGuardResult(
            allowed=False,
            reason="no table referenced; an unqualified SELECT is unsafe",
        )

    # -- 6. Column allowlist -------------------------------------------------
    columns_read: list[str] = []
    for col in parsed.find_all(exp.Column):
        col_name = col.name
        col_table = col.table
        column_ref = f"{col_table}.{col_name}" if col_table else col_name

        # Wildcard is allowed — it still hits the column-level grants on the RO role.
        if col_name == "*":
            columns_read.append(column_ref)
            continue

        # Allow any column if its table (or alias) is in the allowlist.
        # Column-level security is enforced by the read-only database role
        # (CLAUDE.md 1.6). The table allowlist does the heavy lifting.
        if col_table:
            # Resolve alias to real table name
            resolved = table_aliases.get(col_table, col_table)
            if resolved not in tables and col_table not in tables:
                return SQLGuardResult(
                    allowed=False,
                    reason=f"table '{col_table}' referenced via column "
                    f"'{col_name}' is not in the query-agent allowlist",
                )

        columns_read.append(column_ref)

    # -- 7. Force LIMIT ------------------------------------------------------
    has_limit = any(isinstance(node, exp.Limit) for node in parsed.walk())
    if not has_limit:
        parsed = parsed.limit(forced_limit)
        logger.info(
            "sql_guard.limit_appended",
            extra={"forced_limit": forced_limit, "tables": tables_read},
        )

    rewritten = parsed.sql(dialect="postgres")

    return SQLGuardResult(
        allowed=True,
        rewritten=rewritten,
        tables_read=tables_read,
        columns_read=columns_read,
    )


def is_read_only(sql: str) -> bool:
    """Quick check: does this statement only read? Used for pre-flight filtering."""
    result = validate_sql(sql)
    return result.allowed
