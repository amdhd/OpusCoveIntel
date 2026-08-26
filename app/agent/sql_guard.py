"""SQL guardrail — the last line of defence before a generated statement executes.

docs/plan.md 5: read-only role · SELECT only, parsed via sqlglot (not regex) ·
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
#
# The operational tables -- `extraction_jobs`, `llm_calls`, `llm_cache`,
# `human_reviews`, `audit_logs`, `query_logs` -- are deliberately absent. They
# were on this list while `ALLOWED_COLUMNS` named not one of their columns,
# which reads as intent that they be unreadable; the column check was never
# applied, so in practice every column of all six was. That exposed each
# reviewer's identity and notes, every other user's questions and answers, raw
# cached model output, and the audit trail itself -- an agent able to read the
# record of what it did. None of it is needed to answer a covenant question.
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
    }
)

# Columns the agent may SELECT, as `table.column`. Bare names are universal:
# permitted on any allowed table.
#
# This list is enforced. It previously was not -- `validate_sql` checked only
# that a qualified column's *table* was allowed, never the column itself -- so
# the exclusions below were decorative and `documents.storage_uri`,
# `document_chunks.embedding` and the rest were readable. docs/plan.md 5 specifies a
# "table+column allowlist"; only half of it existed.
ALLOWED_COLUMNS: Final[frozenset[str]] = frozenset(
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
        # The column is `issuer_name_guess`; the list said `issuer_guess`, which
        # names nothing. Invisible while the list went unapplied.
        "documents.issuer_name_guess",
        "documents.instrument_name_guess",
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
# Enforced as a ceiling, not just a default: a statement arriving with
# `LIMIT 100000000` used to keep it, which is the runaway scan this exists to
# prevent, waved through because *a* limit was present.
FORCED_LIMIT: Final[int] = 1000

# Functions the agent may call. **Deny by default**, which is the only posture
# that survives contact with Postgres's function catalogue.
#
# The reason this list exists at all: `query_to_xml` and its family take a SQL
# *string* and execute it, so `SELECT query_to_xml('SELECT usename FROM
# pg_user', ...) FROM portfolios` reads a table the allowlist never sees. The
# guardrail reported `tables_read=['portfolios']` and returned the cluster's
# role list -- and the audit trail recorded the same false account of what had
# been read. Enumerating the dangerous functions instead would mean keeping
# pace with every extension anyone installs; enumerating the safe ones does not.
ALLOWED_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        # Aggregates — the reason an analyst reaches for raw SQL at all.
        "COUNT", "SUM", "AVG", "MIN", "MAX",
        # Null handling and conditionals.
        "COALESCE", "NULLIF", "GREATEST", "LEAST", "CASE", "IF",
        # Text.
        "UPPER", "LOWER", "LENGTH", "TRIM", "SUBSTRING", "CONCAT", "REPLACE",
        # Numeric.
        "ABS", "ROUND", "CEIL", "CEILING", "FLOOR", "MOD",
        # Dates. `NOW`/`CURRENT_DATE` are read-only and deterministic per
        # statement, so they cannot be used to probe anything.
        "NOW", "CURRENT_DATE", "CURRENT_TIMESTAMP", "DATE_TRUNC", "EXTRACT", "AGE",
        # Casting and ordering.
        "CAST", "TRY_CAST", "DISTINCT",
    }
)  # fmt: skip

# The regex path is deliberately avoided for validation — parsing with sqlglot
# gives us the AST, and checking the AST is structural. But when sqlglot cannot
# parse a dialect-specific extension, we fall back to a regex to detect
# non-SELECT keywords so the statement is rejected rather than passed.
#
# It runs over `_mask_literals(sql)`, never the raw statement. Applied raw it
# matched inside string literals, and the corpus is trust deeds: "shall not
# create or permit to subsist any security interest" is the canonical negative
# pledge, and every covenant question about it was refused as a mutation
# attempt. So were "shall not grant", "may alter the terms", and anything
# mentioning an update. The check is about SQL structure, and a literal is data.
_NON_SELECT_KEYWORDS: Final[re.Pattern[str]] = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|COPY|VACUUM|REINDEX"
    r"|SET\s+ROLE|DISCARD)\b",
    re.IGNORECASE,
)

# Single-quoted strings (with '' escapes), dollar-quoted strings, and both
# comment forms. Replaced with same-shaped blanks so offsets in any error
# message still line up with the statement the caller sent.
_LITERALS_AND_COMMENTS: Final[re.Pattern[str]] = re.compile(
    r"'(?:[^']|'')*'"  # single-quoted, '' escape
    r"|\$(\w*)\$.*?\$\1\$"  # dollar-quoted
    r"|--[^\n]*"  # line comment
    r"|/\*.*?\*/",  # block comment
    re.DOTALL,
)


def _mask_literals(sql: str) -> str:
    """Blank out string literals and comments, preserving length.

    Keyword detection is a question about SQL structure. Text inside a literal
    is data the analyst is searching *for*, and in this corpus that data is
    full of words like "create" and "grant".
    """
    return _LITERALS_AND_COMMENTS.sub(lambda m: " " * len(m.group(0)), sql)


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
    # Masked, so a keyword inside a string literal is data rather than a verb.
    if _NON_SELECT_KEYWORDS.search(_mask_literals(sql_stripped)):
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

    # -- 3. Must be a SELECT, or a set operation over SELECTs -----------------
    # `exp.Union` and friends are not `exp.Select`, so a plain isinstance check
    # refused every UNION -- including legitimate ones like "holdings breaching
    # gearing UNION holdings breaching interest cover". That failed closed, so
    # it was a capability gap rather than a hole, but the guardrail should
    # reject unsafe SQL, not useful SQL.
    if not isinstance(parsed, exp.Select | exp.SetOperation):
        kind = parsed.key.upper() if hasattr(parsed, "key") else type(parsed).__name__
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

    # -- 6. Function allowlist -----------------------------------------------
    # Before columns, because a disallowed function can read a table no column
    # check would ever see. Deny by default.
    for func in parsed.find_all(exp.Func):
        name = _function_name(func)
        if name not in ALLOWED_FUNCTIONS:
            logger.warning(
                "sql_guard.function_rejected",
                extra={"function": name, "sql": sql_stripped[:200]},
            )
            return SQLGuardResult(
                allowed=False,
                reason=(
                    f"function '{name}' is not on the query-agent allowlist. "
                    f"Only aggregates and simple scalar functions are permitted, "
                    f"because functions such as query_to_xml take a SQL string and "
                    f"execute it, which would bypass the table allowlist entirely."
                ),
            )

    # -- 7. Column allowlist -------------------------------------------------
    # Genuinely applied. `ALLOWED_COLUMNS` used to be consulted for nothing:
    # this loop checked the column's *table* and then accepted any name at all.
    columns_read: list[str] = []
    for col in parsed.find_all(exp.Column):
        col_name = col.name
        col_table = col.table
        column_ref = f"{col_table}.{col_name}" if col_table else col_name

        if col_table:
            resolved = table_aliases.get(col_table, col_table)
            if resolved not in tables and col_table not in tables:
                return SQLGuardResult(
                    allowed=False,
                    reason=f"table '{col_table}' referenced via column "
                    f"'{col_name}' is not in the query-agent allowlist",
                )
            if not _column_allowed(col_name, resolved):
                return SQLGuardResult(
                    allowed=False,
                    reason=f"column '{resolved}.{col_name}' is not in the query-agent allowlist",
                )
        # An unqualified column could belong to any table in the statement, and
        # the guardrail has no schema to resolve it against. Permitted if it is
        # readable on at least one of them -- which still rejects a name that
        # appears on none, and Postgres rejects a genuinely ambiguous one.
        elif not any(_column_allowed(col_name, table) for table in tables_read):
            return SQLGuardResult(
                allowed=False,
                reason=f"column '{col_name}' is not in the query-agent allowlist "
                f"for any of {sorted(set(tables_read))}",
            )

        columns_read.append(column_ref)

    # -- 8. Wildcards --------------------------------------------------------
    # `SELECT *` would return every column including the ones deliberately left
    # out of ALLOWED_COLUMNS, so it defeats the check above. `COUNT(*)` is fine:
    # it returns a row count and no column values.
    for star in parsed.find_all(exp.Star):
        if not _is_inside_count(star):
            return SQLGuardResult(
                allowed=False,
                reason=(
                    "'SELECT *' is not permitted; name the columns you need. "
                    "A wildcard would return columns held back from the "
                    "allowlist, such as document storage paths and raw embeddings."
                ),
            )

    # -- 9. Force LIMIT ------------------------------------------------------
    # A ceiling, not a default. This used to append a LIMIT only when none was
    # present, so `LIMIT 100000000` was honoured: the presence of any limit at
    # all satisfied the check, and the runaway scan the cap exists to prevent
    # went through with the guardrail's blessing.
    existing = _limit_value(parsed)
    if existing is None:
        parsed = parsed.limit(forced_limit)
        logger.info(
            "sql_guard.limit_appended",
            extra={"forced_limit": forced_limit, "tables": tables_read},
        )
    elif existing > forced_limit:
        parsed = parsed.limit(forced_limit)
        logger.info(
            "sql_guard.limit_clamped",
            extra={"requested": existing, "forced_limit": forced_limit, "tables": tables_read},
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


# -- helpers -----------------------------------------------------------------

# Stands in for a LIMIT the guardrail cannot evaluate. Chosen above any
# plausible ceiling so the clamp treats "I could not read this" as "unbounded".
_UNREADABLE_LIMIT: Final[int] = 2**62


def _function_name(func: exp.Func) -> str:
    """The callable's name as written, upper-cased.

    sqlglot gives known functions their own node type (`exp.Count`) and leaves
    everything else as `exp.Anonymous` with the name in `this` -- which is
    where `query_to_xml` and every other unrecognised function land.
    """
    if isinstance(func, exp.Anonymous):
        name = func.this
        return str(name).upper() if name else type(func).__name__.upper()
    names = func.sql_names() if hasattr(func, "sql_names") else ()
    return str(names[0]).upper() if names else type(func).__name__.upper()


def _column_allowed(column: str, table: str) -> bool:
    """Whether `column` may be read from `table`.

    A bare name in `ALLOWED_COLUMNS` (`id`, `created_at`) is universal; a
    qualified one grants only on the table it names.
    """
    return column in ALLOWED_COLUMNS or f"{table}.{column}" in ALLOWED_COLUMNS


def _is_inside_count(node: exp.Expression) -> bool:
    """Whether this node sits within a COUNT(), the one safe home for `*`."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Count):
            return True
        parent = parent.parent
    return False


def _limit_value(parsed: exp.Expression) -> int | None:
    """The statement's outermost LIMIT as an int, or None when it has none.

    A LIMIT the guardrail cannot read as a plain integer -- a parameter, a
    subquery, an expression -- comes back as `_UNREADABLE_LIMIT`, which is
    above any sane ceiling and so is clamped. Returning 0 here (the first
    version of this) meant "unreadable" compared as *below* the ceiling and
    sailed through unclamped, which is the opposite of failing closed.
    """
    limit = parsed.args.get("limit") if isinstance(parsed, exp.Select | exp.SetOperation) else None
    if limit is None:
        for node in parsed.walk():
            if isinstance(node, exp.Limit):
                limit = node
                break
    if limit is None:
        return None
    expression = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(expression, exp.Literal) and expression.is_int:
        return int(expression.this)
    return _UNREADABLE_LIMIT
