"""
Structure-Based Swing Trade Signal Engine (Long & Short).
Strategy: Trend following with pullback/retest structure detection.
Supports both long and short entries.

Entry logic:
  - Long: Bullish bias (Price > SMA50 or EMA9 > EMA21). Price pulls back to support (SMA50 or recent swing low).
  - Short: Bearish bias (Price < SMA50 or EMA9 < EMA21). Price rallies to resistance (SMA50 or recent swing high).
  - Both require configurable "retest" confirmation (e.g. rejection wick, volume spike).

Exit logic:
  - Trailing stops managed by risk_manager.py
  - Hard structural exit if trend completely reverses (emits "exit_long" or "exit_short").
"""

import logging
from typing import Optional, Dict, Any

import pandas as pd

from core.data_ingestion import get_daily_data
from config import (
    SWING_EMA_FAST, SWING_EMA_SLOW, SWING_SMA_FAST, SWING_SMA_SLOW,
    SWING_STRUCTURE_LOOKBACK, SWING_PROXIMITY_PCT,
    SWING_RETEST_MIN_WICK_PCT_LONG, SWING_RETEST_REQUIRE_CLOSE_ABOVE_SUPPORT,
    SWING_RETEST_REQUIRE_HIGHER_LOW, SWING_RETEST_VOLUME_MULTIPLIER_LONG,
    SWING_RETEST_MIN_WICK_PCT_SHORT, SWING_RETEST_REQUIRE_CLOSE_BELOW_RESISTANCE,
    SWING_RETEST_REQUIRE_LOWER_HIGH, SWING_RETEST_VOLUME_MULTIPLIER_SHORT,
    SWING_RSI_LONG_MAX, SWING_RSI_LONG_MIN, SWING_RSI_SHORT_MAX, SWING_RSI_SHORT_MIN,
    SWING_PROFIT_R_MAX, SWING_STOP_MULTIPLIER, SWING_USE_ADAPTIVE_MA, SWING_SMA_ADAPTIVE_FAST
)

logger = logging.getLogger(__name__)

def generate_signal(symbol: str, daily_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generate a swing trade signal for a symbol based on structure rules.
    """
    result = {
        "symbol": symbol,
        "signal": "hold",
        "reason": "",
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "atr": None,
        "entry_type": None,
        "_daily_df": None,
        "_indicator_values": {},
    }

    if daily_df is None:
        daily_df = get_daily_data(symbol, limit=250)

    min_bars = SWING_STRUCTURE_LOOKBACK + 5
    if daily_df is None or len(daily_df) < min_bars:
        result["reason"] = f"Insufficient daily data (need {min_bars} bars)"
        return result

    # Identify primary SMA columns
    sma_fast_col = f"sma_{SWING_SMA_FAST}"
    sma_slow_col = f"sma_{SWING_SMA_SLOW}"
    
    # Fallback to adaptive if SMA200 missing
    if SWING_USE_ADAPTIVE_MA and (sma_slow_col not in daily_df.columns or pd.isna(daily_df.iloc[-1].get(sma_slow_col))):
        sma_fast_col = f"sma_{SWING_SMA_ADAPTIVE_FAST}"
        sma_slow_col = f"sma_{SWING_SMA_FAST}"

    ema_fast_col = f"ema_{SWING_EMA_FAST}"
    ema_slow_col = f"ema_{SWING_EMA_SLOW}"

    # Check required indicators
    for col in [ema_fast_col, ema_slow_col, sma_fast_col, "atr", "rsi", "volume_avg_20"]:
        if col not in daily_df.columns:
            result["reason"] = f"Missing indicator: {col}"
            return result

    latest = daily_df.iloc[-1]
    prev = daily_df.iloc[-2]

    current_price = float(latest["close"])
    atr = float(latest["atr"]) if not pd.isna(latest["atr"]) else 0.0
    rsi = float(latest["rsi"]) if not pd.isna(latest["rsi"]) else 50.0
    vol = float(latest["volume"])
    vol_avg = float(latest["volume_avg_20"])

    ema_fast = float(latest[ema_fast_col])
    ema_slow = float(latest[ema_slow_col])
    sma_fast = float(latest[sma_fast_col])
    
    # Optional slow SMA
    sma_slow = float(latest[sma_slow_col]) if sma_slow_col in daily_df.columns and not pd.isna(latest[sma_slow_col]) else None

    result["entry_price"] = current_price
    result["atr"] = atr
    result["_daily_df"] = daily_df
    result["_indicator_values"] = {
        "current_price": current_price, "ema_fast": ema_fast, "ema_slow": ema_slow,
        "sma_fast": sma_fast, "sma_slow": sma_slow, "rsi": rsi, "atr": atr
    }

    # ── 1. Determine Bias & Trend ──
    # Bullish if price > SMA50 AND short-term momentum is up (EMA9 > EMA21)
    # We are slightly more forgiving: either price > SMA50 OR (EMA9 > EMA21 and price recovering)
    bullish_bias = (current_price > sma_fast) or (ema_fast > ema_slow)
    bearish_bias = (current_price < sma_fast) or (ema_fast < ema_slow)

    # ── 2. Exits (Structure Broken) ──
    # If the trend completely flips, exit open positions.
    if not bullish_bias and bearish_bias:
        result["signal"] = "exit_long"
        result["reason"] = "Structure broken: Bearish bias detected"
        # We don't return immediately, because a bearish bias means we might want to SHORT right now!
    elif not bearish_bias and bullish_bias:
        result["signal"] = "exit_short"
        result["reason"] = "Structure broken: Bullish bias detected"

    # ── 3. Find Structure Levels ──
    # Look back over recent bars (excluding today) to find recent swing extremes
    recent_history = daily_df.iloc[-SWING_STRUCTURE_LOOKBACK:-1]
    swing_low = float(recent_history["low"].min())
    swing_high = float(recent_history["high"].max())

    # We use the higher of (SMA50, swing_low) as primary support, and lower of (SMA50, swing_high) as resistance
    primary_support = max(sma_fast, swing_low)
    primary_resistance = min(sma_fast, swing_high)

    # ── 4. Check Long Setup ──
    if bullish_bias and SWING_RSI_LONG_MIN <= rsi <= SWING_RSI_LONG_MAX:
        # Are we near support?
        if current_price <= primary_support * (1 + SWING_PROXIMITY_PCT) and current_price >= primary_support * (1 - SWING_PROXIMITY_PCT):
            
            # Retest Protections
            is_valid_retest = True
            rejection_reason = []
            
            # Wick check
            daily_range = latest["high"] - latest["low"]
            lower_wick = min(latest["open"], latest["close"]) - latest["low"]
            wick_pct = (lower_wick / daily_range) if daily_range > 0 else 0
            
            if wick_pct < SWING_RETEST_MIN_WICK_PCT_LONG:
                is_valid_retest = False
                rejection_reason.append(f"Wick too small ({wick_pct*100:.1f}% < {SWING_RETEST_MIN_WICK_PCT_LONG*100:.1f}%)")
                
            # Close check
            if SWING_RETEST_REQUIRE_CLOSE_ABOVE_SUPPORT and current_price < primary_support:
                is_valid_retest = False
                rejection_reason.append(f"Closed below support (${current_price:.2f} < ${primary_support:.2f})")
                
            # Higher low check
            if SWING_RETEST_REQUIRE_HIGHER_LOW and latest["low"] <= prev["low"]:
                is_valid_retest = False
                rejection_reason.append("Did not form a higher low")
                
            # Volume check
            if SWING_RETEST_VOLUME_MULTIPLIER_LONG > 0 and vol < vol_avg * SWING_RETEST_VOLUME_MULTIPLIER_LONG:
                is_valid_retest = False
                rejection_reason.append("Insufficient volume spike")

            if is_valid_retest:
                result["signal"] = "buy"
                result["entry_type"] = "support_retest"
                # Stop loss placed slightly below the swing low / support
                result["stop_loss"] = round(primary_support - (atr * SWING_STOP_MULTIPLIER), 2)
                # Take profit targeting RR 
                risk = abs(current_price - result["stop_loss"])
                result["take_profit"] = round(current_price + (risk * SWING_PROFIT_R_MAX), 2)
                
                result["reason"] = (
                    f"Long Setup: Valid retest at support (${primary_support:.2f}). "
                    f"Wick: {wick_pct*100:.1f}%, RSI: {rsi:.1f}. "
                    f"Targeting {SWING_PROFIT_R_MAX}R."
                )
                logger.info(f"[{symbol}] BUY signal: {result['reason']}")
                return result
            elif result["signal"] in ("hold", "exit_long"):
                result["reason"] = f"Near support, but failed retest filters: {', '.join(rejection_reason)}"


    # ── 5. Check Short Setup ──
    if bearish_bias and SWING_RSI_SHORT_MIN <= rsi <= SWING_RSI_SHORT_MAX:
        # Are we near resistance?
        if current_price >= primary_resistance * (1 - SWING_PROXIMITY_PCT) and current_price <= primary_resistance * (1 + SWING_PROXIMITY_PCT):
            
            # Retest Protections
            is_valid_retest = True
            rejection_reason = []
            
            # Wick check
            daily_range = latest["high"] - latest["low"]
            upper_wick = latest["high"] - max(latest["open"], latest["close"])
            wick_pct = (upper_wick / daily_range) if daily_range > 0 else 0
            
            if wick_pct < SWING_RETEST_MIN_WICK_PCT_SHORT:
                is_valid_retest = False
                rejection_reason.append(f"Upper wick too small ({wick_pct*100:.1f}% < {SWING_RETEST_MIN_WICK_PCT_SHORT*100:.1f}%)")
                
            # Close check
            if SWING_RETEST_REQUIRE_CLOSE_BELOW_RESISTANCE and current_price > primary_resistance:
                is_valid_retest = False
                rejection_reason.append(f"Closed above resistance (${current_price:.2f} > ${primary_resistance:.2f})")
                
            # Lower high check
            if SWING_RETEST_REQUIRE_LOWER_HIGH and latest["high"] >= prev["high"]:
                is_valid_retest = False
                rejection_reason.append("Did not form a lower high")
                
            # Volume check
            if SWING_RETEST_VOLUME_MULTIPLIER_SHORT > 0 and vol < vol_avg * SWING_RETEST_VOLUME_MULTIPLIER_SHORT:
                is_valid_retest = False
                rejection_reason.append("Insufficient volume spike")

            if is_valid_retest:
                result["signal"] = "short"
                result["entry_type"] = "resistance_retest"
                # Stop loss placed slightly above the swing high / resistance
                result["stop_loss"] = round(primary_resistance + (atr * SWING_STOP_MULTIPLIER), 2)
                # Take profit targeting RR 
                risk = abs(result["stop_loss"] - current_price)
                result["take_profit"] = round(current_price - (risk * SWING_PROFIT_R_MAX), 2)
                
                result["reason"] = (
                    f"Short Setup: Valid rejection at resistance (${primary_resistance:.2f}). "
                    f"Upper Wick: {wick_pct*100:.1f}%, RSI: {rsi:.1f}. "
                    f"Targeting {SWING_PROFIT_R_MAX}R."
                )
                logger.info(f"[{symbol}] SHORT signal: {result['reason']}")
                return result
            elif result["signal"] in ("hold", "exit_short"):
                result["reason"] = f"Near resistance, but failed retest filters: {', '.join(rejection_reason)}"


    if result["signal"] in ("hold", "exit_long", "exit_short") and not result["reason"]:
        trend = "Bullish" if bullish_bias else "Bearish" if bearish_bias else "Neutral"
        result["reason"] = f"No setup. Trend: {trend}, RSI: {rsi:.1f}, Price: ${current_price:.2f}"

    return result


def generate_signals_batch(symbols: list,
                           daily_cache: Optional[Dict[str, pd.DataFrame]] = None) -> list:
    signals = []
    for symbol in symbols:
        daily_df = daily_cache.get(symbol) if daily_cache else None
        sig = generate_signal(symbol, daily_df=daily_df)
        signals.append(sig)
    return signals
