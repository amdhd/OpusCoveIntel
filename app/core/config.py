"""Application settings.

Single source of truth for configuration. Everything is env-driven; nothing is
hardcoded at a call site (CLAUDE.md 2, 7). Import `get_settings()` rather than
instantiating `Settings` directly so the object is built once per process.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator, model_validator
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
    # Calibrated for real 200-535pp prospectuses, whose dry-run ceilings run
    # $4-21 (docs/review.md finding 4). The old $2.00 default was sized for the
    # 1-5pp synthetic fixtures and aborted every real document mid-extraction.
    # $8.00 covers the expected real spend ($3-7) with headroom; a document
    # whose dry-run ceiling still exceeds it is refused before the first call
    # rather than part-way through (ExtractionPipeline preflight).
    MAX_COST_PER_DOCUMENT_USD: Decimal = Decimal("8.00")
    # Lowered from $200.00 on 2026-08-16 after an agent ran `extract --all
    # --yes` and spent $0.39 of real money on a document nobody had approved.
    # $200 was sized for a build phase that has since been built; the corpus is
    # three synthetic fixtures plus three real prospectuses, and the whole of
    # Phase 10 costs single-digit dollars. A ceiling only does its job when it
    # is close enough to normal spend to stop something. Closes PLAN.md 9 Q6.
    MAX_TOTAL_COST_USD: Decimal = Decimal("10.00")
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

    # -- authentication ---------------------------------------------------
    # Session cookie name and lifetime. Sessions are rows in `user_sessions`,
    # so shortening this does not orphan anything -- expiry is checked in SQL.
    SESSION_COOKIE_NAME: str = "opuscovintel_session"
    SESSION_TTL_HOURS: int = 12
    # Sent only over HTTPS when true. Off locally because the dev stack is
    # plain HTTP and a Secure cookie would simply never be stored; the
    # validator below refuses to let that combination reach production.
    SESSION_COOKIE_SECURE: bool = False

    # -- login rate limiting ----------------------------------------------
    # Failures older than this stop counting, so a locked-out user recovers
    # without an operator (app/auth/rate_limit.py).
    LOGIN_FAILURE_WINDOW_MINUTES: int = 15
    # Free attempts before backoff starts, per username and per client IP. The
    # IP threshold is higher because a shared office address is one IP for the
    # whole desk.
    LOGIN_MAX_FAILURES_PER_USERNAME: int = 5
    LOGIN_MAX_FAILURES_PER_IP: int = 20
    # The delay doubles from the threshold on: 2s, 4s, 8s ... capped at 15
    # minutes, which is also the window, so the cap is where an attacker's
    # throughput settles.
    LOGIN_BACKOFF_BASE_SECONDS: int = 2
    LOGIN_BACKOFF_MAX_SECONDS: int = 900
    # How long an attempt row is kept. Long enough to answer "was this account
    # under attack last night?", short enough that the table stays small.
    LOGIN_ATTEMPT_RETENTION_HOURS: int = 24

    # -- feature flags ----------------------------------------------------
    # Secure by default. The escape hatch exists for local demos, and the
    # validator below makes it unreachable in production -- a flag that can
    # silently disable authentication is how an internal tool ends up open.
    AUTH_ENABLED: bool = True
    AUDIT_ENABLED: bool = True

    @field_validator("STORAGE_DIR")
    @classmethod
    def _expand_storage_dir(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "QWEN_API_KEY", mode="after")
    @classmethod
    def _blank_key_means_absent(cls, v: SecretStr | None) -> SecretStr | None:
        """`QWEN_API_KEY=` in `.env` means "I have no Qwen key", not "my key is ''".

        Without this, an unset-but-present variable arrives as `SecretStr("")`,
        which is not None -- so every `if settings.X_API_KEY is None` fallback
        was skipped and the code went on to build a provider it had no
        credentials for. `get_embedder()` did exactly that: it returned a
        QwenEmbedder rather than the offline one, and indexing died on
        "QWEN_API_KEY is not set" while a perfectly good $0 fallback sat unused.
        """
        if v is not None and not v.get_secret_value().strip():
            return None
        return v

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

    @field_validator(
        "LOGIN_FAILURE_WINDOW_MINUTES",
        "LOGIN_MAX_FAILURES_PER_USERNAME",
        "LOGIN_MAX_FAILURES_PER_IP",
        "LOGIN_BACKOFF_BASE_SECONDS",
        "LOGIN_BACKOFF_MAX_SECONDS",
    )
    @classmethod
    def _login_limits_must_be_positive(cls, v: int) -> int:
        """Zero would read as "no limit" but means "throttle the first attempt".

        A negative or zero threshold turns the backoff on before anyone has
        failed anything, locking every account out of a system that looks
        configured. There is deliberately no "disable rate limiting" setting.
        """
        if v <= 0:
            raise ValueError("login rate-limit settings must be positive")
        return v

    @model_validator(mode="after")
    def _production_must_be_authenticated(self) -> Settings:
        """Refuse to start an unauthenticated or cookie-insecure production.

        Failing at startup rather than serving is the whole point. An
        `AUTH_ENABLED=false` that merely warns gets deployed and then lives
        there, because nothing stops it -- and the review queue and audit log
        are precisely what an unauthenticated deployment exposes.
        """
        if self.ENVIRONMENT != "production":
            return self
        if not self.AUTH_ENABLED:
            raise ValueError(
                "AUTH_ENABLED=false is not permitted when ENVIRONMENT=production; "
                "the flag exists for local demos only"
            )
        if not self.SESSION_COOKIE_SECURE:
            raise ValueError(
                "SESSION_COOKIE_SECURE=false is not permitted when ENVIRONMENT=production; "
                "a session cookie must not travel over plain HTTP"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def session_ttl(self) -> dt.timedelta:
        return dt.timedelta(hours=self.SESSION_TTL_HOURS)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
