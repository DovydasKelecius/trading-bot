"""Deterministic, close-signal/next-open oscillation backtester."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.signals.oscillation import DEFAULTS, add_oscillation_indicators


def _metrics(initial: float, equity: float, trades: list, curve: list,
             benchmark_df: Optional[pd.DataFrame] = None,
             monthly_target_pct: float = 10.0) -> Dict[str, Any]:
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    values = np.array([p["equity"] for p in curve] or [initial], dtype=float)
    peaks = np.maximum.accumulate(values)
    drawdowns = np.divide(peaks - values, peaks, out=np.zeros_like(values), where=peaks != 0)
    returns = pd.Series(values).pct_change().dropna()
    sharpe = float(np.sqrt(252) * returns.mean() / returns.std()) if len(returns) > 1 and returns.std() > 0 else 0.0
    metrics = {
        "initial_equity": round(initial, 2),
        "final_equity": round(equity, 2),
        "net_profit": round(equity - initial, 2),
        "return_pct": round((equity / initial - 1) * 100, 3),
        "max_drawdown_pct": round(float(drawdowns.max()) * 100, 3),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 3) if losses else (999.0 if wins else 0.0),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "trades": len(trades),
    }
    curve_df = pd.DataFrame(curve)
    if not curve_df.empty:
        # Periods have no timezone; normalize aware timestamps explicitly to
        # avoid pandas' noisy implicit conversion warning.
        curve_dates = pd.to_datetime(curve_df["date"], utc=True).dt.tz_localize(None)
        curve_df["period"] = curve_dates.dt.to_period("M")
        month_end = curve_df.groupby("period")["equity"].last()
        monthly_returns = month_end.pct_change().fillna(month_end.iloc[0] / initial - 1) * 100
        metrics["monthly_returns_pct"] = {str(period): round(float(value), 3) for period, value in monthly_returns.items()}
        metrics["monthly_target_pct"] = monthly_target_pct
        metrics["months_at_target"] = int((monthly_returns >= monthly_target_pct).sum())
        metrics["months_tested"] = int(len(monthly_returns))
        metrics["target_hit_rate_pct"] = round(float((monthly_returns >= monthly_target_pct).mean() * 100), 2)
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark = benchmark_df.copy()
        if "timestamp" not in benchmark.columns:
            if "date" in benchmark.columns:
                benchmark["timestamp"] = benchmark["date"]
            else:
                benchmark["timestamp"] = benchmark.index
        benchmark["timestamp"] = pd.to_datetime(benchmark["timestamp"], utc=True).dt.tz_localize(None)
        benchmark = benchmark.sort_values("timestamp")
        if len(benchmark) >= 2:
            benchmark_return = (float(benchmark["close"].iloc[-1]) / float(benchmark["close"].iloc[0]) - 1) * 100
            metrics["benchmark_return_pct"] = round(benchmark_return, 3)
            metrics["strategy_minus_benchmark_pct"] = round(metrics["return_pct"] - benchmark_return, 3)
            metrics["outperformed_benchmark"] = metrics["strategy_minus_benchmark_pct"] > 0
            benchmark["period"] = benchmark["timestamp"].dt.to_period("M")
            benchmark_month_end = benchmark.groupby("period")["close"].last()
            benchmark_monthly = benchmark_month_end.pct_change().fillna(benchmark_month_end.iloc[0] / float(benchmark["close"].iloc[0]) - 1) * 100
            metrics["benchmark_monthly_returns_pct"] = {str(period): round(float(value), 3) for period, value in benchmark_monthly.items()}
            strategy_monthly = metrics.get("monthly_returns_pct", {})
            comparisons = [value - metrics["benchmark_monthly_returns_pct"].get(period, 0) for period, value in strategy_monthly.items() if period in metrics["benchmark_monthly_returns_pct"]]
            metrics["months_beating_benchmark"] = int(sum(value > 0 for value in comparisons))
    return metrics


def run_backtest(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None,
                 initial_equity: float = 100_000, position_pct: float | None = None,
                 start: str | None = None, end: str | None = None,
                 benchmark_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    p = {**DEFAULTS, **(params or {})}
    if position_pct is None:
        position_pct = float(p.get("OSCILLATION_POSITION_PCT", 0.10))
    data = add_oscillation_indicators(df.sort_index().reset_index(drop=True), p)
    if "timestamp" not in data.columns:
        data["timestamp"] = data.get("date", data.index)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_localize(None)
    if start:
        start_cutoff = pd.to_datetime(start, utc=True).tz_localize(None)
        data = data[data["timestamp"] >= start_cutoff]
    if end:
        end_cutoff = pd.to_datetime(end, utc=True).tz_localize(None)
        data = data[data["timestamp"] <= end_cutoff]
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("The selected dates contain no strategy bars")
    test_start, test_end = data["timestamp"].min(), data["timestamp"].max()

    cash = float(initial_equity)
    position = None
    pending = None
    trades, curve = [], []
    signal_diagnostics = {"eligible_bars": 0, "long_entries": 0, "short_entries": 0}
    fee_rate = float(p.get("OSCILLATION_FEE_BPS", 1.0)) / 10_000
    slip_rate = float(p.get("OSCILLATION_SLIPPAGE_BPS", 2.0)) / 10_000

    for index, row in data.iterrows():
        date_value = str(row.get("timestamp", row.get("date", index)))
        if pending and position is None:
            side = pending
            direction = 1 if side == "long" else -1
            entry = float(row["open"]) * (1 + direction * slip_rate)
            quantity = max(1, int(cash * position_pct / entry))
            atr = float(row["osc_atr"])
            position = {
                "side": side, "direction": direction, "entry": entry, "quantity": quantity,
                "stop": entry - direction * atr * float(p["OSCILLATION_STOP_ATR"]),
                "target": entry + direction * atr * float(p["OSCILLATION_TAKE_PROFIT_ATR"]),
                "entry_date": date_value, "bars": 0,
            }
            cash -= entry * quantity * fee_rate
            pending = None

        if position:
            position["bars"] += 1
            direction = position["direction"]
            exit_price = reason = None
            if direction == 1 and row["low"] <= position["stop"]:
                exit_price, reason = position["stop"] * (1 - slip_rate), "stop_loss"
            elif direction == -1 and row["high"] >= position["stop"]:
                exit_price, reason = position["stop"] * (1 + slip_rate), "stop_loss"
            elif direction == 1 and row["high"] >= position["target"]:
                exit_price, reason = position["target"] * (1 - slip_rate), "take_profit"
            elif direction == -1 and row["low"] <= position["target"]:
                exit_price, reason = position["target"] * (1 + slip_rate), "take_profit"
            elif (direction == 1 and row["osc_z"] >= -float(p["OSCILLATION_EXIT_Z"])) or (direction == -1 and row["osc_z"] <= float(p["OSCILLATION_EXIT_Z"])):
                exit_price, reason = float(row["close"]) * (1 - direction * slip_rate), "mean_reversion"
            elif position["bars"] >= int(p["OSCILLATION_MAX_HOLD_BARS"]):
                exit_price, reason = float(row["close"]) * (1 - direction * slip_rate), "max_hold"
            if exit_price is not None:
                gross = direction * (exit_price - position["entry"]) * position["quantity"]
                fee = exit_price * position["quantity"] * fee_rate
                pnl = gross - fee
                cash += pnl
                trades.append({
                    "side": position["side"], "entry_date": position["entry_date"], "exit_date": date_value,
                    "entry_price": round(position["entry"], 4), "exit_price": round(exit_price, 4),
                    "quantity": position["quantity"], "pnl": round(pnl, 2), "exit_reason": reason,
                })
                position = None

        if position is None and index < len(data) - 1 and not pd.isna(row.get("osc_cycle_score")):
            regime = row["osc_cycle_score"] >= float(p["OSCILLATION_MIN_CYCLE_SCORE"]) and row["osc_trend_strength"] <= float(p["OSCILLATION_MAX_TREND_STRENGTH"])
            if regime:
                signal_diagnostics["eligible_bars"] += 1
            if regime and row["osc_z"] <= -float(p["OSCILLATION_ENTRY_Z"]) and row["osc_rsi"] <= float(p["OSCILLATION_RSI_LOW"]):
                pending = "long"
                signal_diagnostics["long_entries"] += 1
            elif regime and row["osc_z"] >= float(p["OSCILLATION_ENTRY_Z"]) and row["osc_rsi"] >= float(p["OSCILLATION_RSI_HIGH"]):
                pending = "short"
                signal_diagnostics["short_entries"] += 1

        marked = cash
        if position:
            marked += position["direction"] * (float(row["close"]) - position["entry"]) * position["quantity"]
        curve.append({"date": date_value, "equity": round(marked, 2)})

    if position and len(data):
        row = data.iloc[-1]
        exit_price = float(row["close"]) * (1 - position["direction"] * slip_rate)
        pnl = position["direction"] * (exit_price - position["entry"]) * position["quantity"] - exit_price * position["quantity"] * fee_rate
        cash += pnl
        trades.append({"side": position["side"], "entry_date": position["entry_date"], "exit_date": str(row.get("timestamp", row.get("date", len(data)-1))), "entry_price": round(position["entry"], 4), "exit_price": round(exit_price, 4), "quantity": position["quantity"], "pnl": round(pnl, 2), "exit_reason": "end_of_test"})
        curve[-1]["equity"] = round(cash, 2)

    benchmark = benchmark_df
    if benchmark is not None and not benchmark.empty:
        benchmark = benchmark.copy()
        if "timestamp" not in benchmark.columns:
            benchmark["timestamp"] = benchmark.get("date", benchmark.index)
        benchmark["timestamp"] = pd.to_datetime(benchmark["timestamp"], utc=True).dt.tz_localize(None)
        # Always align benchmark to the actual strategy bars, not merely the
        # requested range. This prevents a multi-year benchmark from being
        # compared to a shorter strategy test.
        benchmark = benchmark[(benchmark["timestamp"] >= test_start) & (benchmark["timestamp"] <= test_end)]
    metrics = _metrics(initial_equity, cash, trades, curve, benchmark_df=benchmark, monthly_target_pct=float(p.get("MONTHLY_TARGET_PCT", 10.0)))
    metrics.update({
        "test_start": test_start.date().isoformat(), "test_end": test_end.date().isoformat(),
        "bars_tested": len(data),
        "benchmark_start": benchmark["timestamp"].min().date().isoformat() if benchmark is not None and not benchmark.empty else None,
        "benchmark_end": benchmark["timestamp"].max().date().isoformat() if benchmark is not None and not benchmark.empty else None,
    })
    return {"metrics": metrics, "trades": trades, "equity_curve": curve, "params": p, "signal_diagnostics": signal_diagnostics}
