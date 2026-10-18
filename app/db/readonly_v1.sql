-- Versioned privilege policy shared by bootstrap and migration.
-- Keep this version immutable after release; introduce v2 with a new migration
-- when the readable schema changes. New tables are unreadable by default.
DO $readonly_policy$
DECLARE
    query_role constant text := 'opuscovintel_ro';
    readable constant text[] := ARRAY[
        'call_schedules', 'clauses', 'covenants', 'document_chunks',
        'document_pages', 'documents', 'instruments', 'portfolio_holdings',
        'portfolios', 'rating_triggers', 'sukuk_structures'
    ];
    role_id oid;
    public_id oid := 'public'::regnamespace;
    item record;
    attribute record;
    defaults record;
BEGIN
    SELECT oid INTO role_id FROM pg_roles WHERE rolname = query_role;
    -- Managed deployments may provision the role after schema migration.
    -- Bootstrap applies this same policy when it creates that role later.
    IF role_id IS NULL THEN
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_roles WHERE oid = role_id
        AND (rolsuper OR rolcreaterole OR rolcreatedb OR rolreplication OR rolbypassrls)
    ) OR EXISTS (SELECT 1 FROM pg_auth_members WHERE member = role_id)
      OR EXISTS (SELECT 1 FROM pg_database WHERE datname = current_database()
                 AND datdba = role_id)
      OR EXISTS (SELECT 1 FROM pg_namespace WHERE oid = public_id AND nspowner = role_id)
      OR EXISTS (SELECT 1 FROM pg_class WHERE relnamespace = public_id AND relowner = role_id) THEN
        RAISE EXCEPTION 'Query role % must be unprivileged, without role memberships or application object ownership', query_role;
    END IF;

    IF NOT has_schema_privilege(query_role, 'public', 'USAGE') THEN
        EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', query_role);
    END IF;
    IF has_schema_privilege(query_role, 'public', 'CREATE') THEN
        EXECUTE format('REVOKE CREATE ON SCHEMA public FROM %I', query_role);
    END IF;
    EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I', query_role);
    EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I', query_role);

    IF EXISTS (
        SELECT 1 FROM pg_default_acl d
        CROSS JOIN LATERAL aclexplode(d.defaclacl) acl
        WHERE d.defaclobjtype IN ('r', 'S') AND acl.grantee = 0
          AND (d.defaclnamespace = 0 OR d.defaclnamespace = public_id)
    ) THEN
        RAISE EXCEPTION 'PUBLIC default privileges would expose future objects to query role %', query_role;
    END IF;

    -- Remove legacy default grants, including global defaults and grants
    -- established by a different schema creator. Lack of authority fails loudly.
    FOR defaults IN
        SELECT DISTINCT pg_get_userbyid(d.defaclrole) AS creator,
               n.nspname AS schema_name
        FROM pg_default_acl d
        LEFT JOIN pg_namespace n ON n.oid = d.defaclnamespace
        CROSS JOIN LATERAL aclexplode(d.defaclacl) acl
        WHERE d.defaclobjtype IN ('r', 'S') AND acl.grantee = role_id
          AND (d.defaclnamespace = 0 OR d.defaclnamespace = public_id)
    LOOP
        EXECUTE format(
            'ALTER DEFAULT PRIVILEGES FOR ROLE %I %s REVOKE ALL ON TABLES FROM %I',
            defaults.creator,
            CASE WHEN defaults.schema_name IS NULL THEN ''
                 ELSE format('IN SCHEMA %I', defaults.schema_name) END,
            query_role
        );
        EXECUTE format(
            'ALTER DEFAULT PRIVILEGES FOR ROLE %I %s REVOKE ALL ON SEQUENCES FROM %I',
            defaults.creator,
            CASE WHEN defaults.schema_name IS NULL THEN ''
                 ELSE format('IN SCHEMA %I', defaults.schema_name) END,
            query_role
        );
    END LOOP;

    FOR item IN
        SELECT oid, relname FROM pg_class
        WHERE relnamespace = public_id AND relkind IN ('r', 'p', 'v', 'm', 'f')
    LOOP
        -- Table-level REVOKE does not remove separate column-level grants.
        FOR attribute IN
            SELECT attname FROM pg_attribute
            WHERE attrelid = item.oid AND attnum > 0 AND NOT attisdropped
              AND attacl IS NOT NULL
        LOOP
            EXECUTE format('REVOKE ALL (%I) ON TABLE public.%I FROM %I',
                           attribute.attname, item.relname, query_role);
        END LOOP;

        IF item.relname = ANY(readable) THEN
            EXECUTE format('GRANT SELECT ON TABLE public.%I TO %I', item.relname, query_role);
        END IF;

        -- PUBLIC privileges and ownership survive direct REVOKE. Refuse to
        -- declare success when those alternate paths make the policy ineffective.
        IF has_table_privilege(query_role, item.oid, 'SELECT')
                IS DISTINCT FROM (item.relname = ANY(readable))
           OR (NOT item.relname = ANY(readable)
               AND has_any_column_privilege(query_role, item.oid, 'SELECT'))
           OR has_table_privilege(query_role, item.oid, 'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
           OR has_any_column_privilege(query_role, item.oid, 'INSERT,UPDATE,REFERENCES') THEN
            RAISE EXCEPTION 'Query role % has unexpected effective privileges on public.%', query_role, item.relname;
        END IF;
    END LOOP;

    IF has_schema_privilege(query_role, 'public', 'CREATE') THEN
        RAISE EXCEPTION 'Query role % still has effective CREATE on public', query_role;
    END IF;
    FOR item IN
        SELECT oid FROM pg_class WHERE relnamespace = public_id AND relkind = 'S'
    LOOP
        IF has_sequence_privilege(query_role, item.oid, 'USAGE,SELECT,UPDATE') THEN
            RAISE EXCEPTION 'Query role % still has effective sequence privileges', query_role;
        END IF;
    END LOOP;

END
$readonly_policy$;
