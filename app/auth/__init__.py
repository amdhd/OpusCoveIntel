"""Authentication: passwords, sessions, and the identity behind an audit row.

Deliberately small. CLAUDE.md 9 defers OIDC and RBAC, and the gap this closes
is narrower than either: review decisions were recorded against a `reviewer_id`
that the client simply asserted, so the audit trail said whatever the caller
typed. Sessions are server-side rows in Postgres -- revocable, and no broker.
"""
