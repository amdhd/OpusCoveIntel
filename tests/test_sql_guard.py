"""SQL guardrail tests.

docs/plan.md, Phase 7 acceptance: "non-SELECT SQL rejected (test)."

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
        result = validate_sql("SELECT id, issuer_name FROM instruments")
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
        result = validate_sql("SELECT id FROM instruments ORDER BY current_rating_rank")
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
        result = validate_sql("SELECT id FROM documents", allowed_tables=custom)
        assert not result.allowed
        result2 = validate_sql("SELECT id FROM instruments", allowed_tables=custom)
        assert result2.allowed


class TestLimitEnforcement:
    def test_a_limit_is_appended_when_absent(self) -> None:
        result = validate_sql("SELECT id FROM instruments")
        assert result.allowed
        assert "LIMIT" in result.rewritten.upper()

    def test_an_existing_limit_is_preserved(self) -> None:
        result = validate_sql("SELECT id FROM instruments LIMIT 10")
        assert result.allowed
        # LIMIT 10 should stay, not be doubled
        assert result.rewritten.upper().count("LIMIT") == 1

    def test_forced_limit_value_is_configurable(self) -> None:
        result = validate_sql("SELECT id FROM instruments", forced_limit=50)
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
            "SELECT id FROM instruments WHERE id IN "
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


class TestFunctionAllowlist:
    """Functions are denied by default.

    The vector this closes: several Postgres functions take a SQL *string* and
    execute it, so they read tables the allowlist never sees — and the guardrail
    reports the outer table as the only one read, putting a false account into
    the audit trail.
    """

    def test_query_to_xml_cannot_smuggle_a_second_query(self) -> None:
        result = validate_sql(
            "SELECT query_to_xml('SELECT usename FROM pg_user', true, false, '') FROM portfolios"
        )
        assert not result.allowed
        assert "query_to_xml" in result.reason.lower()

    def test_the_whole_query_to_xml_family_is_refused(self) -> None:
        for fn in ("query_to_xmlschema", "table_to_xml", "cursor_to_xml"):
            result = validate_sql(f"SELECT {fn}('SELECT 1', true, false, '') FROM portfolios")
            assert not result.allowed, fn

    def test_sleep_cannot_be_used_to_tie_up_a_connection(self) -> None:
        assert not validate_sql("SELECT pg_sleep(10) FROM instruments").allowed

    def test_server_file_reads_are_refused(self) -> None:
        for fn in ("pg_read_file('/etc/passwd')", "lo_import('/etc/passwd')"):
            assert not validate_sql(f"SELECT {fn} FROM instruments").allowed, fn

    def test_ordinary_aggregates_still_work(self) -> None:
        """Denying by default must not deny the reason SQL was offered."""
        for sql in (
            "SELECT COUNT(*) FROM covenants",
            "SELECT SUM(market_value) FROM portfolio_holdings",
            "SELECT MAX(threshold_amount) FROM covenants",
            "SELECT UPPER(issuer_name) FROM instruments",
        ):
            assert validate_sql(sql).allowed, sql


class TestColumnAllowlist:
    """The column half of docs/plan.md 5's "table+column allowlist", which was absent."""

    def test_a_column_held_back_from_the_allowlist_is_refused(self) -> None:
        for sql in (
            "SELECT storage_uri FROM documents",
            "SELECT uploaded_by FROM documents",
            "SELECT embedding FROM document_chunks",
            "SELECT ocr_text FROM document_pages",
        ):
            result = validate_sql(sql)
            assert not result.allowed, sql
            assert "allowlist" in result.reason

    def test_a_column_that_does_not_exist_is_refused(self) -> None:
        assert not validate_sql("SELECT nope_not_real FROM instruments").allowed

    def test_qualified_and_aliased_columns_are_checked(self) -> None:
        assert not validate_sql("SELECT d.storage_uri FROM documents d").allowed
        assert validate_sql("SELECT d.filename FROM documents d").allowed

    def test_select_star_is_refused_because_it_defeats_the_column_check(self) -> None:
        result = validate_sql("SELECT * FROM documents")
        assert not result.allowed
        assert "name the columns" in result.reason

    def test_count_star_is_still_allowed(self) -> None:
        """COUNT(*) returns a row count, not column values."""
        assert validate_sql("SELECT COUNT(*) FROM documents").allowed


class TestOperationalTablesAreOutOfReach:
    """The agent answers covenant questions; it does not read the back office."""

    def test_the_audit_trail_is_not_readable(self) -> None:
        assert not validate_sql("SELECT payload_json FROM audit_logs").allowed

    def test_other_users_questions_are_not_readable(self) -> None:
        assert not validate_sql("SELECT user_id, question FROM query_logs").allowed

    def test_reviewer_identity_is_not_readable(self) -> None:
        assert not validate_sql("SELECT reviewer_id, review_notes FROM human_reviews").allowed

    def test_raw_cached_model_output_is_not_readable(self) -> None:
        assert not validate_sql("SELECT response_json FROM llm_cache").allowed


class TestLimitIsACeiling:
    def test_an_oversized_limit_is_clamped(self) -> None:
        """A limit that is present but absurd is the runaway scan, not an exemption."""
        result = validate_sql("SELECT id FROM instruments LIMIT 100000000")
        assert result.allowed
        assert "LIMIT 1000" in result.rewritten.upper()

    def test_a_modest_limit_is_left_alone(self) -> None:
        result = validate_sql("SELECT id FROM instruments LIMIT 5")
        assert "LIMIT 5" in result.rewritten.upper()

    def test_a_limit_the_guard_cannot_read_is_clamped_not_trusted(self) -> None:
        """ "Unreadable" must compare as unbounded, not as zero."""
        result = validate_sql("SELECT id FROM instruments LIMIT (SELECT 10)")
        assert not result.allowed or "LIMIT 1000" in result.rewritten.upper()


class TestLegalTextIsNotMistakenForSQL:
    """The corpus is trust deeds, and trust deeds are full of SQL keywords.

    Scanning the raw statement matched inside string literals, so the single
    most common negative-pledge phrasing was refused as a mutation attempt.
    """

    def test_the_canonical_negative_pledge_phrase_is_searchable(self) -> None:
        result = validate_sql(
            "SELECT clause_text FROM clauses "
            "WHERE clause_text ILIKE '%create or permit to subsist%'"
        )
        assert result.allowed, result.reason

    def test_other_keyword_bearing_legal_phrases_are_searchable(self) -> None:
        for phrase in ("shall not grant", "may alter the terms", "update of the register"):
            result = validate_sql(
                f"SELECT clause_text FROM clauses WHERE clause_text ILIKE '%{phrase}%'"
            )
            assert result.allowed, f"{phrase}: {result.reason}"

    def test_a_real_mutation_is_still_refused(self) -> None:
        """Masking literals must not blind the check to actual SQL."""
        for sql in (
            "DELETE FROM instruments",
            "UPDATE instruments SET issuer_name = 'x'",
            "DROP TABLE instruments",
            "SELECT id FROM instruments; DELETE FROM instruments",
        ):
            assert not validate_sql(sql).allowed, sql

    def test_a_comment_cannot_hide_a_keyword_from_the_parser(self) -> None:
        result = validate_sql("SELECT id FROM instruments -- harmless\n")
        assert result.allowed


class TestSetOperations:
    def test_a_union_of_two_allowed_selects_passes(self) -> None:
        """Rejecting every UNION failed closed, but refused useful queries."""
        result = validate_sql("SELECT id FROM covenants UNION SELECT id FROM clauses")
        assert result.allowed, result.reason

    def test_a_union_reaching_a_forbidden_table_is_refused(self) -> None:
        assert not validate_sql("SELECT id FROM covenants UNION SELECT oid FROM pg_class").allowed
