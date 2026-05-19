from __future__ import annotations

from fastapi.testclient import TestClient

from server.main import app


def test_status_returns_200() -> None:
    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert "timeframe" in data
    assert "symbols_count" in data


def test_portfolio_returns_null_when_no_session() -> None:
    client = TestClient(app)
    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    assert resp.json()["portfolio"] is None


def test_market_data_returns_empty_when_no_session() -> None:
    client = TestClient(app)
    resp = client.get("/api/market-data")
    assert resp.status_code == 200
    assert resp.json()["bars"] == {}


def test_stop_when_not_running_returns_200() -> None:
    client = TestClient(app)
    resp = client.post("/api/session/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


def test_start_without_token_returns_400() -> None:
    """Attempting to start when UPSTOX_ACCESS_TOKEN is unset must return 400."""
    from unittest.mock import MagicMock, patch

    client = TestClient(app)
    with patch("server.main.AppSettings") as MockSettings:
        s = MagicMock()
        s.upstox_access_token = ""
        MockSettings.return_value = s
        resp = client.post("/api/session/start", json={"timeframe": "60m", "warmup_bars": 10})
    assert resp.status_code == 400
