from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Flat settings model. Every field maps directly to an env var in .env.example."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Trading mode (CRITICAL: live_trading_enabled defaults to False — never change default)
    live_trading_enabled: bool = False
    paper_sessions_completed: int = 0
    paper_starting_capital: float = 83000.0
    max_monthly_api_spend_inr: float = 1500.0

    # Data paths
    parquet_cache_dir: Path = Path("./data/cache")
    journal_db_path: Path = Path("./data/journal.db")

    # Broker — Upstox
    broker: str = "upstox"
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_access_token: str = ""
    upstox_redirect_uri: str = "http://localhost:3000"

    # AI
    anthropic_api_key: str = ""
    claude_model_primary: str = "claude-sonnet-4-6"
    claude_model_cheap: str = "claude-haiku-4-5-20251001"
    claude_model_review: str = "claude-opus-4-6"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_dir: Path = Path("./logs")

    # Deployment
    deployment_env: str = "development"
    static_ip: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
