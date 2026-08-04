"""Application settings.

Single source of truth for configuration. Everything is env-driven; nothing is
hardcoded at a call site (CLAUDE.md 2, 7). Import `get_settings()` rather than
instantiating `Settings` directly so the object is built once per process.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # -- application ------------------------------------------------------
    ENVIRONMENT: Environment = "local"
    LOG_LEVEL: LogLevel = "INFO"
    SERVICE_NAME: str = "opuscovintel"

    # -- database ---------------------------------------------------------
    # Read-write role, used by the API and ingestion workers.
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql+asyncpg://opuscovintel:opuscovintel@localhost:5432/opuscovintel"  # type: ignore[assignment]
    )
    # Read-only role. CLAUDE.md 1.6: the query agent MUST use this one.
    DATABASE_URL_RO: PostgresDsn = Field(
        default="postgresql+asyncpg://opuscovintel_ro:opuscovintel_ro@localhost:5432/opuscovintel"  # type: ignore[assignment]
    )
    DB_POOL_SIZE: int = 5
    DB_POOL_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False
    # Applied to every read-only agent session (PLAN.md 5).
    DB_STATEMENT_TIMEOUT_MS: int = 5_000

    # -- object storage ---------------------------------------------------
    # Local filesystem for the MVP behind an S3-shaped interface (PLAN.md, Phase 3).
    STORAGE_DIR: Path = Path("./var/storage")

    # -- retrieval --------------------------------------------------------
    # Qwen text-embedding-v4. Changing this forces a full re-embed and index
    # rebuild -- see PLAN.md 9, open question 2.
    VECTOR_DIMENSION: int = 1024

    # -- llm providers ----------------------------------------------------
    ANTHROPIC_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    QWEN_API_KEY: SecretStr | None = None
    QWEN_BASE_URL: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # Model IDs are configuration, never literals in code (CLAUDE.md 2).
    EXTRACTION_MODEL: str = "claude-opus-5"
    SYNTHESIS_MODEL: str = "claude-opus-5"
    JUDGE_MODEL: str = "claude-opus-5"
    CHEAP_MODEL: str = "qwen-plus"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    # Needs a current, verified ID before the VLM cap can be tuned (PLAN.md 9, Q3).
    VLM_MODEL: str = "gpt-4o"

    # -- budget guards ----------------------------------------------------
    # PLAN.md 2. Money is Decimal, never float (CLAUDE.md 6).
    MAX_COST_PER_DOCUMENT_USD: Decimal = Decimal("2.00")
    MAX_TOTAL_COST_USD: Decimal = Decimal("200.00")
    MAX_COST_PER_CALL_USD: Decimal = Decimal("0.50")
    MAX_VLM_PAGES_PER_DOC: int = 40

    # -- ingestion limits -------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 100
    MAX_PDF_PAGES: int = 600

    # -- extraction -------------------------------------------------------
    # Below this, a field is routed to the human review queue (CLAUDE.md 5).
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.85
    # Verbatim-quote match floor for citation verification (CLAUDE.md 1.3).
    CITATION_FUZZY_THRESHOLD: float = 0.92

    # -- feature flags ----------------------------------------------------
    AUTH_ENABLED: bool = False
    AUDIT_ENABLED: bool = True

    @field_validator("STORAGE_DIR")
    @classmethod
    def _expand_storage_dir(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator(
        "MAX_COST_PER_DOCUMENT_USD",
        "MAX_TOTAL_COST_USD",
        "MAX_COST_PER_CALL_USD",
    )
    @classmethod
    def _budget_must_be_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError(
                "budget guards must be positive; use a feature flag to disable a stage"
            )
        return v

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
