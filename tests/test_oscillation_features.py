import math
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backtesting.oscillation import run_backtest
from core.profile_manager import ProfileManager
from optimization.random_search import run_random_search
from optimization.adaptive_search import run_adaptive_search
from core.scheduler import _price_exit
from dashboard.routes import router


def wave_bars(count=420):
    rows = []
    for index in range(count):
        close = 100 + math.sin(index / 6) * 10 + math.sin(index / 19)
        rows.append({
            "timestamp": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=index)),
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
        self.assertEqual({trade["side"] for trade in first["trades"]}, {"long", "short"})

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

    def test_short_thresholds_are_inverted(self):
        self.assertEqual(_price_exit("sell", 111, 110, 90)[0], "stopped_out")
        self.assertEqual(_price_exit("sell", 89, 110, 90)[0], "closed")
        self.assertIsNone(_price_exit("sell", 100, 110, 90))

    def test_adaptive_search_mutates_and_compares_periods(self):
        result = run_adaptive_search(
            pd.DataFrame(wave_bars(320)),
            base_params={"OSCILLATION_ENTRY_Z": 1.8},
            max_iterations=15, patience=5, folds=3, seed=9, store=False,
        )
        self.assertEqual(result["iterations"], len(result["history"]))
        self.assertEqual(result["comparison"]["periods_tested"], 3)
        self.assertIn("holdout_better", result["comparison"])
        self.assertIn("Optuna", result["engine"])
        self.assertTrue(any(item["changed_count"] > 1 for item in result["history"][1:]))

    def test_new_pages_have_shared_navigation_and_categories(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        settings = client.get("/settings")
        experiments = client.get("/experiments")
        self.assertEqual(settings.status_code, 200)
        self.assertIn("Oscillation strategy", settings.text)
        self.assertIn("Portfolio risk", settings.text)
        self.assertIn("Learn & Test", settings.text)
        self.assertEqual(experiments.status_code, 200)
        self.assertIn("Optimize Multiple Settings", experiments.text)

    def test_learning_api_returns_holdout_comparison(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        with patch("optimization.adaptive_search._store"):
            response = client.post("/api/optimize/oscillation/learn", json={
                "bars": wave_bars(260), "max_iterations": 10,
                "patience": 5, "folds": 3, "seed": 2,
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("holdout", response.json())


if __name__ == "__main__":
    unittest.main()
