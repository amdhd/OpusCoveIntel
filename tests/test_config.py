"""Settings tests.

These lock down the invariants that cost money or break audits if they drift.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_budget_guards_are_decimal_not_float() -> None:
    """CLAUDE.md 6: money is Decimal. Float budget arithmetic silently drifts."""
    settings = Settings()
    assert isinstance(settings.MAX_COST_PER_DOCUMENT_USD, Decimal)
    assert isinstance(settings.MAX_TOTAL_COST_USD, Decimal)
    assert isinstance(settings.MAX_COST_PER_CALL_USD, Decimal)


def test_budget_defaults_match_the_plan() -> None:
    """PLAN.md 2. A silent change here is a silent change to the spend ceiling."""
    settings = Settings()
    assert settings.MAX_COST_PER_DOCUMENT_USD == Decimal("2.00")
    assert settings.MAX_TOTAL_COST_USD == Decimal("200.00")
    assert settings.MAX_COST_PER_CALL_USD == Decimal("0.50")
    assert settings.MAX_VLM_PAGES_PER_DOC == 40


@pytest.mark.parametrize("value", ["0", "-1", "-0.01"])
def test_non_positive_budget_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A zero ceiling would read as 'unlimited' to a naive comparison."""
    monkeypatch.setenv("MAX_COST_PER_DOCUMENT_USD", value)
    with pytest.raises(ValidationError):
        Settings()


def test_vector_dimension_default_is_1024() -> None:
    """Changing this forces a full re-embed and index rebuild (PLAN.md 9, Q2)."""
    assert Settings().VECTOR_DIMENSION == 1024


def test_read_write_and_read_only_urls_are_distinct() -> None:
    """CLAUDE.md 1.6: the agent's read-only role is a hard invariant."""
    settings = Settings()
    assert str(settings.DATABASE_URL) != str(settings.DATABASE_URL_RO)


def test_secrets_are_not_exposed_by_model_dump() -> None:
    """`make config` and log lines must never print a raw key."""
    settings = Settings(ANTHROPIC_API_KEY="sk-ant-secret-value")  # type: ignore[arg-type]
    assert "sk-ant-secret-value" not in str(settings.model_dump())
    assert settings.ANTHROPIC_API_KEY is not None
    assert settings.ANTHROPIC_API_KEY.get_secret_value() == "sk-ant-secret-value"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_environment_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prod")  # the valid literal is "production"
    with pytest.raises(ValidationError):
        Settings()
