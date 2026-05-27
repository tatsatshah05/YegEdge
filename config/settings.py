from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import model_validator
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
    paper_starting_capital: Decimal = Decimal("83000.00")
    max_monthly_api_spend_inr: Decimal = Decimal("1500.00")

    @model_validator(mode="after")
    def require_60_paper_sessions_before_live(self) -> AppSettings:
        if self.live_trading_enabled and self.paper_sessions_completed < 60:
            raise ValueError(
                f"live_trading_enabled=True requires paper_sessions_completed >= 60, "
                f"got {self.paper_sessions_completed}"
            )
        return self

    # Data paths
    parquet_cache_dir: Path = Path("./data/cache")
    journal_db_path: Path = Path("./data/journal.db")

    # Broker — set to "yfinance" for no-auth paper trading, "upstox" for live NSE
    broker: str = "yfinance"
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_access_token: str = ""
    upstox_redirect_uri: str = "http://localhost:3000"

    # Alpaca — paper trading for NYSE (LIVE_TRADING_ENABLED must remain False)
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Finnhub — real-time NYSE tick streaming (free tier: 60 symbols)
    finnhub_api_key: str = ""

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
