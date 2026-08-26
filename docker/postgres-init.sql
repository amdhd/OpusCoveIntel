-- Runs once, on first initialisation of the Postgres data volume.
-- To re-run after editing: `make down-volumes && make up`.

-- Extensions -------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector: embedding storage + HNSW
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- trigram similarity for citation matching
CREATE EXTENSION IF NOT EXISTS unaccent;   -- accent-insensitive FTS

-- Read-only role for the query agent (CLAUDE.md 1.6) -----------------------
-- The agent's SQL guardrail is defence in depth; this is the actual boundary.
-- A write from the agent path must fail at the database, not at code review.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'opuscovintel_ro') THEN
        CREATE ROLE opuscovintel_ro LOGIN PASSWORD 'opuscovintel_ro';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE opuscovintel TO opuscovintel_ro;
GRANT USAGE ON SCHEMA public TO opuscovintel_ro;

-- Cover tables that already exist...
GRANT SELECT ON ALL TABLES IN SCHEMA public TO opuscovintel_ro;
-- ...and every table Alembic creates from here on (Phase 2 onward).
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO opuscovintel_ro;

-- Explicitly withhold write capability, including on future objects.
REVOKE CREATE ON SCHEMA public FROM opuscovintel_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON TABLES FROM opuscovintel_ro;

-- Belt-and-braces: the session also sets default_transaction_read_only=on
-- (app/db/session.py), so a write fails even if a grant is later widened.
ALTER ROLE opuscovintel_ro SET default_transaction_read_only = on;
ALTER ROLE opuscovintel_ro SET statement_timeout = '5s';
