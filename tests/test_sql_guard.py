"""SQL guardrail tests.

PLAN.md, Phase 7 acceptance: "non-SELECT SQL rejected (test)."

The guardrail is a pure function — no database required. Every test constructs
a statement and asserts the guardrail either passes or rejects it with a
specific reason.
"""

from __future__ import annotations

from app.agent.sql_guard import (
    validate_sql,
)


class TestSelectAllowed:
    def test_a_simple_select_on_an_allowed_table_passes(self) -> None:
        result = validate_sql("SELECT * FROM instruments")
        assert result.allowed
        assert "instruments" in result.tables_read

    def test_a_select_with_explicit_columns_passes(self) -> None:
        result = validate_sql(
            "SELECT instrument_name, issuer_name, current_rating FROM instruments"
        )
        assert result.allowed

    def test_a_select_with_a_where_clause_passes(self) -> None:
        result = validate_sql(
            "SELECT covenant_type, threshold_amount FROM covenants "
            "WHERE instrument_id = '00000000-0000-0000-0000-000000000001'"
        )
        assert result.allowed

    def test_a_select_with_a_join_passes(self) -> None:
        result = validate_sql(
            "SELECT i.instrument_name, c.covenant_type "
            "FROM instruments i "
            "JOIN covenants c ON c.instrument_id = i.id"
        )
        assert result.allowed

    def test_a_select_with_an_aggregate_passes(self) -> None:
        result = validate_sql(
            "SELECT p.name, COUNT(ph.id), SUM(ph.market_value) "
            "FROM portfolios p "
            "JOIN portfolio_holdings ph ON ph.portfolio_id = p.id "
            "GROUP BY p.name"
        )
        assert result.allowed

    def test_a_select_with_order_by_passes(self) -> None:
        result = validate_sql(
            "SELECT * FROM instruments ORDER BY current_rating_rank"
        )
        assert result.allowed


class TestNonSelectRejected:
    def test_an_insert_is_rejected(self) -> None:
        result = validate_sql("INSERT INTO instruments (issuer_name) VALUES ('test')")
        assert not result.allowed
        assert "SELECT" in result.reason

    def test_an_update_is_rejected(self) -> None:
        result = validate_sql("UPDATE instruments SET issuer_name = 'x' WHERE id = 1")
        assert not result.allowed
        assert "SELECT" in result.reason or "non-SELECT" in result.reason

    def test_a_delete_is_rejected(self) -> None:
        result = validate_sql("DELETE FROM instruments WHERE id = 1")
        assert not result.allowed

    def test_a_drop_statement_is_rejected(self) -> None:
        result = validate_sql("DROP TABLE instruments")
        assert not result.allowed

    def test_a_create_statement_is_rejected(self) -> None:
        result = validate_sql("CREATE TABLE test (id INT)")
        assert not result.allowed

    def test_a_truncate_is_rejected(self) -> None:
        result = validate_sql("TRUNCATE TABLE instruments")
        assert not result.allowed


class TestTableAllowlist:
    def test_a_select_on_a_disallowed_table_is_rejected(self) -> None:
        result = validate_sql("SELECT * FROM pg_stat_activity")
        assert not result.allowed
        assert "not in the query-agent allowlist" in result.reason

    def test_a_select_on_a_non_existent_table_is_rejected(self) -> None:
        result = validate_sql("SELECT * FROM evil_table")
        assert not result.allowed
        assert "not in the query-agent allowlist" in result.reason

    def test_custom_allowlist_replaces_default(self) -> None:
        custom = frozenset({"instruments"})
        result = validate_sql("SELECT * FROM documents", allowed_tables=custom)
        assert not result.allowed
        result2 = validate_sql("SELECT * FROM instruments", allowed_tables=custom)
        assert result2.allowed


class TestLimitEnforcement:
    def test_a_limit_is_appended_when_absent(self) -> None:
        result = validate_sql("SELECT * FROM instruments")
        assert result.allowed
        assert "LIMIT" in result.rewritten.upper()

    def test_an_existing_limit_is_preserved(self) -> None:
        result = validate_sql("SELECT * FROM instruments LIMIT 10")
        assert result.allowed
        # LIMIT 10 should stay, not be doubled
        assert result.rewritten.upper().count("LIMIT") == 1

    def test_forced_limit_value_is_configurable(self) -> None:
        result = validate_sql("SELECT * FROM instruments", forced_limit=50)
        assert result.allowed
        assert "LIMIT 50" in result.rewritten.upper()


class TestEdgeCases:
    def test_an_empty_string_is_rejected(self) -> None:
        result = validate_sql("")
        assert not result.allowed
        assert "empty" in result.reason

    def test_a_whitespace_only_string_is_rejected(self) -> None:
        result = validate_sql("   \n  ")
        assert not result.allowed

    def test_a_subquery_that_selects_is_allowed(self) -> None:
        result = validate_sql(
            "SELECT * FROM instruments WHERE id IN "
            "(SELECT instrument_id FROM covenants WHERE covenant_type = 'gearing_ratio')"
        )
        assert result.allowed

    def test_unparseable_sql_is_rejected(self) -> None:
        result = validate_sql("SELECT FROM WHERE x = y AND AND z")
        assert not result.allowed
        assert "parse" in result.reason.lower()

    def test_a_select_with_no_from_is_rejected(self) -> None:
        result = validate_sql("SELECT 1")
        assert not result.allowed
        # No table referenced → unsafe
