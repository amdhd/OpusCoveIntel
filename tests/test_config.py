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


class TestBlankApiKeysMeanAbsent:
    """`QWEN_API_KEY=` in `.env` means "no key", not "the key is the empty string".

    An unset-but-present variable arrives as `SecretStr("")`, which is not
    None -- so every `if settings.X_API_KEY is None` fallback was skipped and
    the code built a provider it had no credentials for. `get_embedder()` did
    exactly that on a real machine: it returned a QwenEmbedder instead of the
    offline one, and indexing died on "QWEN_API_KEY is not set" with a working
    $0 fallback sitting unused.
    """

    def test_an_empty_key_normalises_to_none(self) -> None:
        settings = Settings(
            ENVIRONMENT="test",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="   ",
            QWEN_API_KEY="",
        )

        assert settings.ANTHROPIC_API_KEY is None
        assert settings.OPENAI_API_KEY is None
        assert settings.QWEN_API_KEY is None

    def test_a_real_key_survives(self) -> None:
        settings = Settings(ENVIRONMENT="test", ANTHROPIC_API_KEY="sk-ant-not-a-real-key")

        assert settings.ANTHROPIC_API_KEY is not None
        assert settings.ANTHROPIC_API_KEY.get_secret_value() == "sk-ant-not-a-real-key"

    def test_the_offline_embedder_is_chosen_when_the_key_is_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure this actually caused, pinned end to end."""
        from app.llm.embeddings import HashingEmbedder, get_embedder

        blank = Settings(ENVIRONMENT="local", QWEN_API_KEY="", EMBEDDING_MODEL="text-embedding-v4")
        monkeypatch.setattr("app.llm.embeddings.get_settings", lambda: blank)

        assert isinstance(get_embedder(), HashingEmbedder)
