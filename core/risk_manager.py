"""
Unified Risk Manager (AGGRESSIVE VARIANT).

Handles:
- ATR-based position sizing (4% risk, 20% max position)
- Portfolio allocation enforcement (60/80 day/swing split)
- Max concurrent position checks
- Stop-loss and take-profit calculations
- Pre-trade validation
- Trailing stop with ratchet mechanism (+20% gain -> tighten from 3x to 2x ATR)

Important margin notes:
    - Day Trading uses Day Trading Buying Power (4x equity in US, requires $25k+ for PDT)
    - Swing Trading uses Reg T margin (2x equity)
    - The risk manager tracks these separately. Day trade orders draw from day trade
      allocation, swing trade orders from swing allocation. One must never consume the other.
"""

import logging
import math
from typing import Optional, Dict, Tuple

from db.database import get_db
from db.models import Trade
from config import (
    MAX_RISK_PER_TRADE, MAX_POSITION_VALUE_PCT, ATR_PERIOD,
    DAY_TRADE_ALLOCATION, SWING_TRADE_ALLOCATION,
    DAY_MAX_POSITIONS, SWING_MAX_POSITIONS,
    DAY_STOP_MULTIPLIER, DAY_PROFIT_MULTIPLIER,
    SWING_STOP_MULTIPLIER, SWING_POSITION_SIZE_REDUCTION,
    SWING_RATCHET_ENABLED, SWING_RATCHET_THRESHOLD,
    SWING_RATCHET_STOP_MULTIPLIER,
    SWING_LOCK_PROFIT_AT_R, SWING_LOCK_PROFIT_TO_R, SWING_PROFIT_R_MAX,
)

logger = logging.getLogger(__name__)


def calculate_position_size(portfolio_equity: float, atr: float,
                            strategy_type: str, entry_price: float = 0.0,
                            allocation_pct: Optional[float] = None,
                            stop_loss: Optional[float] = None,
                            params: Optional[Dict] = None) -> int:
    p = params or {}
    """
    Calculate number of shares to trade based on risk sizing,
    capped by maximum position value (% of equity).

    Formula: shares = risk_amount / risk_per_share
    Where: risk_amount = portfolio_equity * p.get('MAX_RISK_PER_TRADE', MAX_RISK_PER_TRADE) * allocation_pct

    The result is then capped so total position cost never exceeds
    p.get('MAX_POSITION_VALUE_PCT', MAX_POSITION_VALUE_PCT) of portfolio equity.
    """
    if portfolio_equity <= 0:
        logger.warning(f"Invalid inputs: equity={portfolio_equity}")
        return 0

    if allocation_pct is None:
        allocation_pct = p.get('DAY_TRADE_ALLOCATION', DAY_TRADE_ALLOCATION) if strategy_type == "day" else p.get('SWING_TRADE_ALLOCATION', SWING_TRADE_ALLOCATION)

    risk_amount = portfolio_equity * p.get('MAX_RISK_PER_TRADE', MAX_RISK_PER_TRADE) * allocation_pct

    # Determine risk per share
    if stop_loss is not None and entry_price > 0:
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            risk_per_share = 0.01  # Prevent division by zero
    else:
        if atr <= 0:
            logger.warning(f"Invalid inputs: atr={atr} and no stop_loss provided")
            return 0
        stop_multiplier = p.get('DAY_STOP_MULTIPLIER', DAY_STOP_MULTIPLIER) if strategy_type == "day" else p.get('SWING_STOP_MULTIPLIER', SWING_STOP_MULTIPLIER)
        risk_per_share = atr * stop_multiplier

    shares = risk_amount / risk_per_share

    # Swing size reduction (now 0% in aggressive config -- no reduction)
    if strategy_type == "swing":
        shares *= (1 - p.get('SWING_POSITION_SIZE_REDUCTION', SWING_POSITION_SIZE_REDUCTION))

    shares = max(1, math.floor(shares))

    # Cap position value at p.get('MAX_POSITION_VALUE_PCT', MAX_POSITION_VALUE_PCT) of equity
    if entry_price > 0:
        max_position_value = portfolio_equity * p.get('MAX_POSITION_VALUE_PCT', MAX_POSITION_VALUE_PCT)
        max_shares_by_value = math.floor(max_position_value / entry_price)
        if max_shares_by_value < 1:
            max_shares_by_value = 1
        if shares > max_shares_by_value:
            logger.info(
                f"Position capped: {shares} -> {max_shares_by_value} shares "
                f"(max position value ${max_position_value:.2f} at ${entry_price:.2f}/share)"
            )
            shares = max_shares_by_value

    logger.debug(
        f"Position size: {shares} shares (equity=${portfolio_equity:.2f}, "
        f"ATR=${atr:.2f}, strategy={strategy_type}, risk=${risk_amount:.2f})"
    )
    return shares


def calculate_stop_loss(entry_price: float, atr: float, strategy_type: str,
                        side: str = "buy", params: Optional[Dict] = None) -> float:
    """Calculate stop-loss price based on ATR multiplier."""
    p = params or {}
    multiplier = p.get('DAY_STOP_MULTIPLIER', DAY_STOP_MULTIPLIER) if strategy_type == "day" else p.get('SWING_STOP_MULTIPLIER', SWING_STOP_MULTIPLIER)
    if side.lower() == "buy":
        return round(entry_price - (atr * multiplier), 2)
    else:
        return round(entry_price + (atr * multiplier), 2)


def calculate_take_profit(entry_price: float, atr: float, strategy_type: str,
                          side: str = "buy", params: Optional[Dict] = None) -> Optional[float]:
    """Calculate take-profit price. Day trades have fixed TP, swing trades don't."""
    p = params or {}
    if strategy_type == "swing":
        return None  # Swing trades: let winners run

    multiplier = p.get('DAY_PROFIT_MULTIPLIER', DAY_PROFIT_MULTIPLIER)
    if side.lower() == "buy":
        return round(entry_price + (atr * multiplier), 2)
    else:
        return round(entry_price - (atr * multiplier), 2)


def get_open_position_count(strategy_type: str) -> int:
    """Count open positions for a given strategy type."""
    with get_db() as session:
        count = session.query(Trade).filter(
            Trade.strategy_type == strategy_type,
            Trade.status == "open"
        ).count()
    return count


def get_strategy_exposure(strategy_type: str) -> float:
    """Calculate total dollar exposure for a strategy's open positions."""
    with get_db() as session:
        trades = session.query(Trade).filter(
            Trade.strategy_type == strategy_type,
            Trade.status == "open"
        ).all()
        exposure = sum(t.entry_price * t.quantity for t in trades)
    return exposure


def pre_trade_check(symbol: str, strategy_type: str, shares: int,
                    entry_price: float, portfolio_equity: float,
                    buying_power: float, params: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Validate a trade before execution.

    Checks:
        1. Strategy hasn't exceeded max positions
        2. Strategy hasn't exceeded buying power allocation
        3. Single trade risk doesn't exceed max position value
    """
    p = params or {}

    # Check 1: Max positions
    max_positions = p.get('DAY_MAX_POSITIONS', DAY_MAX_POSITIONS) if strategy_type == "day" else p.get('SWING_MAX_POSITIONS', SWING_MAX_POSITIONS)
    current_positions = get_open_position_count(strategy_type)

    if current_positions >= max_positions:
        reason = (
            f"Max {strategy_type} positions reached: {current_positions}/{max_positions}. "
            f"Cannot open new {symbol} position."
        )
        logger.warning(f"[Risk] {reason}")
        return False, reason

    # Check 2: Buying power allocation
    allocation_pct = p.get('DAY_TRADE_ALLOCATION', DAY_TRADE_ALLOCATION) if strategy_type == "day" else p.get('SWING_TRADE_ALLOCATION', SWING_TRADE_ALLOCATION)
    max_allocation = buying_power * allocation_pct
    current_exposure = get_strategy_exposure(strategy_type)
    order_cost = entry_price * shares

    if current_exposure + order_cost > max_allocation:
        reason = (
            f"Allocation exceeded for {strategy_type}: current ${current_exposure:.2f} + "
            f"order ${order_cost:.2f} > max ${max_allocation:.2f} "
            f"({allocation_pct*100:.0f}% of ${buying_power:.2f})"
        )
        logger.warning(f"[Risk] {reason}")
        return False, reason

    # Check 3: Single trade cost vs max position value
    trade_risk = entry_price * shares
    max_position_value = portfolio_equity * p.get('MAX_POSITION_VALUE_PCT', MAX_POSITION_VALUE_PCT)
    if trade_risk > max_position_value:
        reason = (
            f"Single trade cost ${trade_risk:.2f} is too large relative to "
            f"portfolio equity ${portfolio_equity:.2f}"
        )
        logger.warning(f"[Risk] {reason}")
        return False, reason

    logger.info(
        f"[Risk] Pre-trade check PASSED for {symbol} ({strategy_type}): "
        f"{shares} shares @ ${entry_price:.2f}, positions: {current_positions}/{max_positions}"
    )
    return True, "Approved"


def update_trailing_stop(trade: Trade, current_price: float, atr: float, params: Optional[Dict] = None) -> Optional[float]:
    p = params or {}
    """
    Update trailing stop for swing trades with ratchet mechanism and R-based profit locking.
    Moves stop up for longs, and down for shorts.

    Features:
    1. Lock Profit: If price reaches +1.5R, move stop to +1.0R.
    2. ATR Ratchet: Tightens ATR multiplier after 20% gain.

    Returns:
        New stop-loss price if updated, None if no change
    """
    new_stop = None
    stop_mult = p.get('SWING_STOP_MULTIPLIER', SWING_STOP_MULTIPLIER)

    if trade.entry_price > 0:
        # Calculate Risk per share (1R) using the initial take profit
        # (Since TP was set at entry to entry +/- (1R * p.get('SWING_PROFIT_R_MAX', SWING_PROFIT_R_MAX)))
        if trade.take_profit and trade.take_profit != trade.entry_price:
            risk_per_share = abs(trade.take_profit - trade.entry_price) / p.get('SWING_PROFIT_R_MAX', SWING_PROFIT_R_MAX)
        else:
            risk_per_share = 0

        if trade.side == "buy":
            gain_pct = (current_price - trade.entry_price) / trade.entry_price
            current_r = (current_price - trade.entry_price) / risk_per_share if risk_per_share > 0 else 0
        else:
            gain_pct = (trade.entry_price - current_price) / trade.entry_price
            current_r = (trade.entry_price - current_price) / risk_per_share if risk_per_share > 0 else 0

        # 1. ATR Ratchet
        if p.get('SWING_RATCHET_ENABLED', SWING_RATCHET_ENABLED) and gain_pct >= p.get('SWING_RATCHET_THRESHOLD', SWING_RATCHET_THRESHOLD):
            stop_mult = p.get('SWING_RATCHET_STOP_MULTIPLIER', SWING_RATCHET_STOP_MULTIPLIER)

        # Base ATR trailing stop
        if trade.side == "buy":
            atr_stop = round(current_price - (atr * stop_mult), 2)
        else:
            atr_stop = round(current_price + (atr * stop_mult), 2)

        # 2. R-based Profit Lock Stop
        r_lock_stop = None
        if risk_per_share > 0 and current_r >= p.get('SWING_LOCK_PROFIT_AT_R', SWING_LOCK_PROFIT_AT_R):
            if trade.side == "buy":
                r_lock_stop = round(trade.entry_price + (risk_per_share * p.get('SWING_LOCK_PROFIT_TO_R', SWING_LOCK_PROFIT_TO_R)), 2)
            else:
                r_lock_stop = round(trade.entry_price - (risk_per_share * p.get('SWING_LOCK_PROFIT_TO_R', SWING_LOCK_PROFIT_TO_R)), 2)

        # 3. Take the tightest stop (highest for longs, lowest for shorts)
        if trade.side == "buy":
            candidates = [trade.stop_loss, atr_stop]
            if r_lock_stop: candidates.append(r_lock_stop)
            best_stop = max(candidates)

            if best_stop > trade.stop_loss:
                new_stop = best_stop
                reason = "+1R Lock" if best_stop == r_lock_stop else f"ATR ({stop_mult}x)"
                logger.info(f"[{trade.symbol}] Trailing stop updated: ${trade.stop_loss:.2f} -> ${new_stop:.2f} ({reason})")

        else: # short
            candidates = [trade.stop_loss if trade.stop_loss > 0 else float('inf'), atr_stop]
            if r_lock_stop: candidates.append(r_lock_stop)
            best_stop = min(candidates)

            if best_stop < trade.stop_loss or trade.stop_loss == 0:
                new_stop = best_stop
                reason = "+1R Lock" if best_stop == r_lock_stop else f"ATR ({stop_mult}x)"
                logger.info(f"[{trade.symbol}] Trailing stop updated (short): ${trade.stop_loss:.2f} -> ${new_stop:.2f} ({reason})")

    return new_stop
