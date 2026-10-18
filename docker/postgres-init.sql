-- Runs on first initialisation of the Postgres data volume; safe to replay
-- with psql -v ON_ERROR_STOP=1 -f docker/postgres-init.sql after migrations.
-- Alembic's initial schema migration installs vector, pg_trgm and unaccent.
\set ON_ERROR_STOP on

-- Read-only role for the query agent (CLAUDE.md 1.6) -----------------------
-- The agent's SQL guardrail is defence in depth; this is the actual boundary.
-- A write from the agent path must fail at the database, not at code review.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'opuscovintel_ro') THEN
        CREATE ROLE opuscovintel_ro LOGIN PASSWORD 'opuscovintel_ro';
    END IF;
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO opuscovintel_ro', current_database());
END
$$;

-- Role administration stays in operator provisioning. Ordinary table-owner
-- migrations can reconcile table grants without CREATEROLE or role membership.
ALTER ROLE opuscovintel_ro SET default_transaction_read_only = on;
ALTER ROLE opuscovintel_ro SET statement_timeout = '5s';

-- Resolve relative to this script, both from a checkout and the Compose mount.
-- Fresh databases have no tables yet. The migration applies this same policy
-- once tables exist; late role provisioning and replay also reconcile it here.
\ir ../app/db/readonly_v1.sql
