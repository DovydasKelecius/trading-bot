"""Oscillation/mean-reversion signals with range-regime filtering."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


DEFAULTS = {
    "OSCILLATION_LOOKBACK": 50,
    "OSCILLATION_ENTRY_Z": 1.8,
    "OSCILLATION_EXIT_Z": 0.25,
    "OSCILLATION_RSI_PERIOD": 14,
    "OSCILLATION_RSI_LOW": 35,
    "OSCILLATION_RSI_HIGH": 65,
    "OSCILLATION_MIN_CYCLE_SCORE": 0.55,
    "OSCILLATION_MAX_TREND_STRENGTH": 0.35,
    "OSCILLATION_MIN_CROSSINGS": 3,
    "OSCILLATION_MAX_HOLD_BARS": 20,
    "OSCILLATION_STOP_ATR": 2.0,
    "OSCILLATION_TAKE_PROFIT_ATR": 2.5,
    "OSCILLATION_FEE_BPS": 1.0,
    "OSCILLATION_SLIPPAGE_BPS": 2.0,
}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous).abs(),
        (df["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_oscillation_indicators(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    lookback = int(p["OSCILLATION_LOOKBACK"])
    result = df.copy()
    close = result["close"].astype(float)
    result["osc_mean"] = close.rolling(lookback).mean()
    result["osc_std"] = close.rolling(lookback).std(ddof=0)
    result["osc_z"] = (close - result["osc_mean"]) / result["osc_std"].replace(0, np.nan)
    result["osc_rsi"] = _rsi(close, int(p["OSCILLATION_RSI_PERIOD"]))
    result["osc_atr"] = _atr(result)

    centered = close - result["osc_mean"]
    signs = np.sign(centered)
    crossings = signs.ne(signs.shift(1)).rolling(lookback).sum()
    expected = max(int(p["OSCILLATION_MIN_CROSSINGS"]), 1)
    crossing_score = (crossings / (expected * 2)).clip(0, 1)

    slope = (result["osc_mean"] - result["osc_mean"].shift(max(2, lookback // 5))).abs()
    result["osc_trend_strength"] = (slope / result["osc_std"].replace(0, np.nan)).clip(0, 2)
    trend_score = (1 - result["osc_trend_strength"] / max(float(p["OSCILLATION_MAX_TREND_STRENGTH"]), 0.01)).clip(0, 1)
    result["osc_cycle_score"] = (0.65 * crossing_score + 0.35 * trend_score).clip(0, 1)
    return result


def generate_signal(symbol: str, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = {**DEFAULTS, **(params or {})}
    enriched = add_oscillation_indicators(df, p)
    row = enriched.iloc[-1]
    required = ["osc_z", "osc_rsi", "osc_atr", "osc_cycle_score", "osc_trend_strength"]
    if any(pd.isna(row[key]) for key in required):
        return {"symbol": symbol, "signal": "hold", "reason": "Insufficient oscillation history"}

    regime_ok = (
        row["osc_cycle_score"] >= float(p["OSCILLATION_MIN_CYCLE_SCORE"])
        and row["osc_trend_strength"] <= float(p["OSCILLATION_MAX_TREND_STRENGTH"])
    )
    signal = "hold"
    if regime_ok and row["osc_z"] <= -float(p["OSCILLATION_ENTRY_Z"]) and row["osc_rsi"] <= float(p["OSCILLATION_RSI_LOW"]):
        signal = "buy"
    elif regime_ok and row["osc_z"] >= float(p["OSCILLATION_ENTRY_Z"]) and row["osc_rsi"] >= float(p["OSCILLATION_RSI_HIGH"]):
        signal = "short"

    price = float(row["close"])
    atr = float(row["osc_atr"])
    direction = 1 if signal == "buy" else -1
    return {
        "symbol": symbol,
        "signal": signal,
        "entry_price": price,
        "atr": atr,
        "stop_loss": round(price - direction * atr * float(p["OSCILLATION_STOP_ATR"]), 4) if signal != "hold" else None,
        "take_profit": round(price + direction * atr * float(p["OSCILLATION_TAKE_PROFIT_ATR"]), 4) if signal != "hold" else None,
        "z_score": round(float(row["osc_z"]), 4),
        "rsi": round(float(row["osc_rsi"]), 2),
        "cycle_score": round(float(row["osc_cycle_score"]), 4),
        "trend_strength": round(float(row["osc_trend_strength"]), 4),
        "reason": "Oscillating regime entry" if signal != "hold" else "No oscillation entry",
    }


def generate_signals_batch(symbols: list, daily_cache: Optional[Dict[str, pd.DataFrame]] = None,
                           params: Optional[Dict[str, Any]] = None) -> list:
    results = []
    for symbol in symbols:
        df = daily_cache.get(symbol) if daily_cache else None
        if df is None:
            results.append({"symbol": symbol, "signal": "hold", "reason": "No daily data"})
        else:
            results.append(generate_signal(symbol, df, params=params))
    return results
