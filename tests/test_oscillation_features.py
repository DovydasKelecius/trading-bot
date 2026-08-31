import math
import unittest
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backtesting.oscillation import run_backtest
from core.profile_manager import ProfileManager
from optimization.random_search import run_random_search
from dashboard.routes import router


def wave_bars(count=420):
    rows = []
    for index in range(count):
        close = 100 + math.sin(index / 6) * 10 + math.sin(index / 19)
        rows.append({
            "timestamp": f"2025-01-{(index % 28) + 1:02d}T00:00:00",
            "open": close - 0.3, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": 1_000_000,
        })
    return rows


class OscillationFeatureTests(unittest.TestCase):
    def test_backtest_is_deterministic_and_trades(self):
        frame = pd.DataFrame(wave_bars())
        params = {
            "OSCILLATION_ENTRY_Z": 1.0,
            "OSCILLATION_RSI_LOW": 48,
            "OSCILLATION_RSI_HIGH": 52,
            "OSCILLATION_MIN_CYCLE_SCORE": 0.25,
            "OSCILLATION_MAX_TREND_STRENGTH": 1.0,
        }
        first = run_backtest(frame, params)
        second = run_backtest(frame, params)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertGreater(first["metrics"]["trades"], 0)

    def test_seeded_random_search_is_reproducible(self):
        frame = pd.DataFrame(wave_bars(260))
        first = run_random_search(frame, tests=3, seed=7, store=False)
        second = run_random_search(frame, tests=3, seed=7, store=False)
        self.assertEqual(first["leaderboard"], second["leaderboard"])

    def test_profile_round_trip(self):
        manager = ProfileManager(Path("profiles"))
        saved = manager.save("Automated Wave Test", {"OSCILLATION_ENTRY_Z": 1.7})
        try:
            self.assertEqual(manager.load(saved["id"])["settings"]["OSCILLATION_ENTRY_Z"], 1.7)
            self.assertIn(saved, manager.list())
        finally:
            manager.delete(saved["id"])

    def test_backtest_api_accepts_supplied_bars(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        response = client.post("/api/backtest/oscillation", json={
            "bars": wave_bars(260),
            "params": {
                "OSCILLATION_ENTRY_Z": 1.0,
                "OSCILLATION_RSI_LOW": 48,
                "OSCILLATION_RSI_HIGH": 52,
                "OSCILLATION_MIN_CYCLE_SCORE": 0.25,
                "OSCILLATION_MAX_TREND_STRENGTH": 1.0,
            },
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("metrics", response.json())


if __name__ == "__main__":
    unittest.main()
