"""Seeded random search for oscillation parameters."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from backtesting.oscillation import run_backtest


SEARCH_SPACE = {
    # Execution/risk settings are included so experiments can lock or mutate
    # them alongside signal settings (the live risk gate consumes these keys).
    "DAY_TRADE_ALLOCATION": (0.10, 0.80, "float"),
    "SWING_TRADE_ALLOCATION": (0.10, 0.90, "float"),
    "MAX_RISK_PER_TRADE": (0.005, 0.05, "float"),
    "MAX_POSITION_VALUE_PCT": (0.05, 0.30, "float"),
    "DAY_MAX_POSITIONS": (1, 20, "int"),
    "SWING_MAX_POSITIONS": (1, 20, "int"),
    "OSCILLATION_LOOKBACK": (20, 120, "int"),
    "OSCILLATION_ENTRY_Z": (1.0, 3.0, "float"),
    "OSCILLATION_EXIT_Z": (0.0, 0.75, "float"),
    "OSCILLATION_RSI_LOW": (20, 45, "int"),
    "OSCILLATION_RSI_HIGH": (55, 80, "int"),
    "OSCILLATION_MIN_CYCLE_SCORE": (0.35, 0.85, "float"),
    "OSCILLATION_MAX_TREND_STRENGTH": (0.15, 0.65, "float"),
    "OSCILLATION_MAX_HOLD_BARS": (5, 40, "int"),
    "OSCILLATION_STOP_ATR": (1.0, 3.5, "float"),
    "OSCILLATION_TAKE_PROFIT_ATR": (1.0, 4.0, "float"),
    "OSCILLATION_RSI_PERIOD": (5, 30, "int"),
    "OSCILLATION_MIN_CROSSINGS": (2, 12, "int"),
    "OSCILLATION_FEE_BPS": (0.0, 10.0, "float"),
    "OSCILLATION_SLIPPAGE_BPS": (0.0, 15.0, "float"),
    "OSCILLATION_POSITION_PCT": (0.02, 0.25, "float"),
}


def fitness(metrics: Dict[str, Any]) -> float:
    trade_penalty = max(0, 12 - metrics["trades"]) * 0.5
    profit_factor = min(metrics["profit_factor"], 4)
    benchmark_bonus = float(metrics.get("strategy_minus_benchmark_pct", 0.0)) * 0.5
    target_bonus = float(metrics.get("target_hit_rate_pct", 0.0)) * 0.1
    return round(
        metrics["return_pct"] + metrics["sharpe"] * 2 + profit_factor
        - metrics["max_drawdown_pct"] * 1.5 - trade_penalty
        + benchmark_bonus + target_bonus,
        4,
    )


def _candidate(rng: random.Random) -> Dict[str, Any]:
    values = {}
    for key, (low, high, kind) in SEARCH_SPACE.items():
        values[key] = rng.randint(low, high) if kind == "int" else round(rng.uniform(low, high), 4)
    if values["OSCILLATION_RSI_LOW"] >= values["OSCILLATION_RSI_HIGH"]:
        values["OSCILLATION_RSI_HIGH"] = values["OSCILLATION_RSI_LOW"] + 20
    return values


def run_random_search(df: pd.DataFrame, tests: int = 50, seed: int = 42,
                      base_params: Optional[Dict[str, Any]] = None,
                      initial_equity: float = 100_000,
                      start: str | None = None, end: str | None = None,
                      store: bool = True, benchmark_df: Optional[pd.DataFrame] = None,
                      locked_keys: Optional[list[str]] = None) -> Dict[str, Any]:
    tests = max(1, min(int(tests), 500))
    rng = random.Random(seed)
    results = []
    for run_number in range(1, tests + 1):
        candidate = _candidate(rng)
        if locked_keys:
            candidate = {key: value for key, value in candidate.items() if key not in locked_keys}
        params = {**(base_params or {}), **candidate}
        result = run_backtest(df, params, initial_equity=initial_equity, start=start, end=end, benchmark_df=benchmark_df)
        changed = [key for key in candidate if not base_params or params.get(key) != base_params.get(key)]
        results.append({"run": run_number, "score": fitness(result["metrics"]), "metrics": result["metrics"], "params": params, "changed_count": len(changed), "changed_parameters": changed})
    results.sort(key=lambda item: item["score"], reverse=True)
    payload = {"seed": seed, "tests": tests, "locked_keys": list(locked_keys or []), "best": results[0], "leaderboard": results[:25]}
    if store:
        _store(payload)
    return payload


def _store(payload: Dict[str, Any]):
    directory = Path(__file__).resolve().parent.parent / "experiments"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"random-search-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
