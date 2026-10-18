"""Run actual bootstrap/migration SQL on disposable, synthetic databases.

READONLY_POLICY_TEST_URL explicitly selects a disposable test cluster. No .env
or application database settings are consulted. No pgvector is needed: these
tests exercise privileges, with thin synthetic tables bearing the real names.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy.engine import URL, make_url

from app.agent.sql_guard import ALLOWED_TABLES

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "app/db/readonly_v1.sql"
BOOTSTRAP = ROOT / "docker/postgres-init.sql"
REVISION = ROOT / "migrations/versions/20261018_0001_reconcile_query_role.py"
PRIVATE_TABLES = {
    "users",
    "user_sessions",
    "login_attempts",
    "audit_logs",
    "query_logs",
    "llm_calls",
    "llm_cache",
    "human_reviews",
    "extraction_jobs",
    "alembic_version",
}


def load_revision() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_readonly_policy_revision", REVISION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_allowlist_matches_the_query_guard() -> None:
    policy = POLICY.read_text()
    names = policy.split("readable constant text[] := ARRAY[", 1)[1].split("]", 1)[0]
    assert set(re.findall(r"'([a-z_]+)'", names)) == ALLOWED_TABLES
    assert "\\ir ../app/db/readonly_v1.sql" in BOOTSTRAP.read_text()
    assert load_revision().reconcile_sql() == policy


@dataclass
class Probe:
    url: URL
    psql: str
    database: str
    role: str
    bootstrap: Path
    extra_roles: list[str] = field(default_factory=list)

    def run(self, sql: str = "", *, file: Path | None = None, database: str | None = None) -> str:
        command = [
            self.psql,
            "-X",
            "-A",
            "-t",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "--host",
            str(self.url.query.get("host", self.url.host or "localhost")),
            "--port",
            str(self.url.query.get("port", self.url.port or 5432)),
            "--username",
            str(self.url.username),
            "--dbname",
            database or self.database,
        ]
        command += ["--file", str(file)] if file else ["--command", sql]
        try:
            result = subprocess.run(
                command,
                env={"PATH": os.defpath, "PGPASSWORD": self.url.password or ""},
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.CalledProcessError as error:
            error.add_note(error.stderr)
            raise
        return result.stdout.strip()

    def init(self) -> None:
        self.run(file=self.bootstrap)

    def schema(self) -> None:
        for name in sorted(ALLOWED_TABLES | PRIVATE_TABLES):
            self.run(f"CREATE TABLE public.{name} (id integer, secret text)")
            self.run(f"INSERT INTO public.{name} VALUES (1, 'synthetic')")

    def migrate(self, monkeypatch: pytest.MonkeyPatch, *, as_role: str | None = None) -> None:
        revision = load_revision()
        with monkeypatch.context() as patch:
            patch.setattr(
                revision.op,
                "execute",
                lambda statement: self.run(
                    (f"SET ROLE {as_role}; " if as_role else "")
                    + str(statement).replace("opuscovintel_ro", self.role)
                ),
            )
            revision.upgrade()

    def readable(self, table: str) -> bool:
        return (
            self.run(f"SELECT has_table_privilege('{self.role}', 'public.{table}', 'SELECT')")
            == "t"
        )

    def assert_boundary(self) -> None:
        assert {
            table for table in ALLOWED_TABLES | PRIVATE_TABLES if self.readable(table)
        } == ALLOWED_TABLES
        for table in sorted(ALLOWED_TABLES):
            assert (
                self.run(f"SET ROLE {self.role}; SELECT secret FROM public.{table}") == "synthetic"
            )
        for table in sorted(PRIVATE_TABLES):
            with pytest.raises(subprocess.CalledProcessError) as error:
                self.run(f"SET ROLE {self.role}; SELECT secret FROM public.{table}")
            assert "permission denied" in error.value.stderr
        # SET ROLE does not load the role's read-only login default. These
        # failures therefore prove ACL enforcement independently of that default.
        with pytest.raises(subprocess.CalledProcessError) as error:
            self.run(f"SET ROLE {self.role}; INSERT INTO public.documents VALUES (2, 'dummy')")
        assert "permission denied" in error.value.stderr


@pytest.fixture
def probe(tmp_path: Path) -> Iterator[Probe]:
    dsn = os.getenv("READONLY_POLICY_TEST_URL")
    if not dsn:
        pytest.skip("set READONLY_POLICY_TEST_URL to a disposable PostgreSQL test cluster")
    psql = shutil.which("psql")
    if psql is None:
        pytest.fail("READONLY_POLICY_TEST_URL requires a local psql client")
    url = make_url(dsn)
    suffix = uuid.uuid4().hex
    role = "ro_policy_" + suffix
    database = "ro_policy_" + suffix
    # Preserve native psql relative includes; only substitute the disposable
    # role name, so tests never alter a configured application role.
    for source in (BOOTSTRAP, POLICY):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text().replace("opuscovintel_ro", role))
    state = Probe(url, psql, database, role, tmp_path / BOOTSTRAP.relative_to(ROOT))
    state.run(f'CREATE DATABASE "{database}"', database=url.database or "postgres")
    try:
        yield state
    finally:
        state.run(f'DROP DATABASE "{database}"', database=url.database or "postgres")
        for name in [role, *state.extra_roles]:
            state.run(f'DROP ROLE IF EXISTS "{name}"', database=url.database or "postgres")


@pytest.mark.parametrize("ordering", ["fresh", "late_role", "replay"])
def test_bootstrap_and_migration_ordering_preserves_the_boundary(
    probe: Probe, monkeypatch: pytest.MonkeyPatch, ordering: str
) -> None:
    if ordering != "late_role":
        probe.init()
    probe.schema()
    probe.migrate(monkeypatch)
    if ordering != "fresh":
        probe.init()
    probe.assert_boundary()


def test_legacy_table_column_and_default_grants_are_repaired(
    probe: Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe.init()
    probe.schema()
    creator = "legacy_" + uuid.uuid4().hex
    probe.extra_roles.append(creator)
    probe.run(f"CREATE ROLE {creator}; GRANT USAGE, CREATE ON SCHEMA public TO {creator}")
    probe.run(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {probe.role}")
    probe.run(f"GRANT SELECT (secret) ON public.users TO {probe.role}")
    probe.run(f"GRANT UPDATE ON public.documents TO {probe.role}")
    probe.run(
        "CREATE SEQUENCE public.synthetic_sequence; "
        f"GRANT USAGE, SELECT ON SEQUENCE public.synthetic_sequence TO {probe.role}"
    )
    probe.run(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {probe.role}")
    probe.run(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO {probe.role}")
    probe.run(f"ALTER DEFAULT PRIVILEGES FOR ROLE {creator} GRANT SELECT ON TABLES TO {probe.role}")
    assert probe.readable("users")  # Prove the privilege being repaired exists.

    probe.migrate(monkeypatch)
    probe.init()  # Replaying bootstrap cannot restore the old broad privileges.
    probe.assert_boundary()
    probe.run("CREATE TABLE public.future_secret (id integer)")
    probe.run(f"SET ROLE {creator}; CREATE TABLE public.other_future_secret (id integer)")
    assert not probe.readable("future_secret")
    assert not probe.readable("other_future_secret")
    probe.run("CREATE SEQUENCE public.future_sequence")
    for name in ("synthetic_sequence", "future_sequence"):
        assert (
            probe.run(
                f"SELECT has_sequence_privilege('{probe.role}', 'public.{name}', "
                "'USAGE,SELECT,UPDATE')"
            )
            == "f"
        )


def test_table_owner_can_migrate_without_role_administration(
    probe: Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe.init()
    probe.schema()
    owner = "owner_" + uuid.uuid4().hex
    probe.extra_roles.append(owner)
    probe.run(f"CREATE ROLE {owner}; GRANT USAGE ON SCHEMA public TO {owner}")
    for table in sorted(ALLOWED_TABLES | PRIVATE_TABLES):
        probe.run(f"ALTER TABLE public.{table} OWNER TO {owner}")

    probe.migrate(monkeypatch, as_role=owner)
    probe.assert_boundary()


@pytest.mark.parametrize("bypass", ["public_table", "public_default", "membership", "ownership"])
def test_reconciliation_refuses_effective_privilege_bypasses(
    probe: Probe, monkeypatch: pytest.MonkeyPatch, bypass: str
) -> None:
    probe.init()
    probe.schema()
    if bypass == "public_table":
        probe.run("GRANT SELECT ON public.users TO PUBLIC")
    elif bypass == "public_default":
        probe.run("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO PUBLIC")
    elif bypass == "membership":
        parent = "parent_" + uuid.uuid4().hex
        probe.extra_roles.append(parent)
        probe.run(f"CREATE ROLE {parent}; GRANT {parent} TO {probe.role}")
    else:
        probe.run(f"ALTER TABLE public.users OWNER TO {probe.role}")

    with pytest.raises(subprocess.CalledProcessError) as error:
        probe.migrate(monkeypatch)
    assert "Query role" in error.value.stderr or "PUBLIC default privileges" in error.value.stderr
