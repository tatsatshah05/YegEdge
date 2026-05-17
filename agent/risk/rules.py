from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class PerTradeRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_risk_fraction: float
    max_position_fraction: float
    min_reward_risk: float


class PortfolioRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_concurrent_positions: int
    min_cash_fraction: float


class LossCaps(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_daily_loss_fraction: float
    max_weekly_loss_fraction: float
    max_drawdown_fraction: float


class FrequencyRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_new_orders_per_day: int
    symbol_cooldown_minutes: int


class TradingWindowRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trade_start_ist: str
    trade_end_ist: str


class RiskRules(BaseModel):
    per_trade: PerTradeRules
    portfolio: PortfolioRules
    loss_caps: LossCaps
    frequency: FrequencyRules
    windows: TradingWindowRules


def load_risk_rules(path: Path = Path("config/risk_rules.yaml")) -> RiskRules:
    """Load and parse risk_rules.yaml into typed Pydantic models.

    Extra YAML keys (stop_loss, take_profit, etc.) are ignored by each sub-model.
    """
    with path.open() as f:
        data = yaml.safe_load(f)
    return RiskRules(
        per_trade=PerTradeRules(**data["per_trade"]),
        portfolio=PortfolioRules(**data["portfolio"]),
        loss_caps=LossCaps(**data["loss_caps"]),
        frequency=FrequencyRules(**data["frequency"]),
        windows=TradingWindowRules(**data["windows"]),
    )
