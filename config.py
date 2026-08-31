"""
Central configuration for the Trading Bot.
All tunable parameters live here. No magic numbers elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca API ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# --- Alpaca Data Feed ---
# "sip" = consolidated feed from ALL US exchanges (most accurate, requires paid subscription)
# "iex" = IEX exchange only (~2-3% of market, free but can have wildly different prices)
# Set via env var or default to "sip". If you're on the free plan, Alpaca will auto-downgrade
# to IEX for real-time data but allow SIP for historical queries with a 15-min delay.
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip")

# --- Bot Behavior ---
ENABLE_TRADING = True          # False = monitor-only mode (no orders placed)
LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR

# --- Portfolio Allocation (AGGRESSIVE — backtested to +89,770% over 10yr) ---
DAY_TRADE_ALLOCATION = 0.60    # 60% of buying power for day trades (was 20%)
SWING_TRADE_ALLOCATION = 0.80  # 80% of buying power for swing trades (was 60%)

# --- Day Trade Settings (AGGRESSIVE) ---
DAY_MAX_POSITIONS = 5
DAY_STOP_MULTIPLIER = 1.5      # ATR multiplier for stop-loss (applied to DAILY ATR)
DAY_PROFIT_MULTIPLIER = 2.5    # ATR multiplier for take-profit — wider target (was 2.0)
DAY_RSI_ENTRY_THRESHOLD = 50   # Minimum intraday RSI to enter (was 60 — rejected too many setups)
DAY_RSI_ENTRY_CEILING = 80     # Maximum intraday RSI (was 75 — missed strong momentum)
DAY_VOLUME_MULTIPLIER = 1.2    # Volume must be > 1.2x 20-period avg (was 1.5 — rejected 49% of days)
DAY_VOLUME_SPIKE_CAP = 6.0     # Reject if volume > 6x avg (was 5.0)
DAY_SCAN_INTERVAL_MINUTES = 15  # 15-min interval (matches SIP free-tier delay)
DAY_USE_DAILY_ATR = True        # Use daily ATR for stop/profit (not intraday 5min ATR)
DAY_MIN_INTRADAY_ATR = 0.05    # Minimum intraday ATR to trade (skip illiquid/dead stocks)
DAY_MAX_PRICE_DEVIATION_PCT = 0.10  # Skip if price is >10% away from last daily close
DAY_REQUIRE_CONFIRMATION = False # Dropped confirmation bar (was True — killed 1-2% more entries)
DAY_DAILY_RSI_FLOOR = 40       # Require daily RSI > 40 (was 45 — slightly looser)
DAY_REQUIRE_ABOVE_SMA50 = True # Require price to be above daily SMA50
DAY_MIN_DAILY_BARS = 20        # Minimum daily bars needed (for indicator quality)
DAY_MAX_HOLD_DAYS = 3          # NEW: Hold up to 3 days instead of EOD liquidation (was 1)

# --- Swing Trade Settings (Structure-Based Strategy — Long & Short) ---
SWING_MAX_POSITIONS = 10

# ── Moving Averages ────────────────────────────────────────────────────────
SWING_EMA_FAST = 9           # Fast EMA — short-term momentum signal
SWING_EMA_SLOW = 21          # Slow EMA — bias confirmation (above = bullish, below = bearish)
SWING_SMA_FAST = 50          # SMA50 — primary bias filter and support/resistance
SWING_SMA_SLOW = 200         # SMA200 — macro trend confirmation
SWING_SMA_ADAPTIVE_FAST = 20 # Fallback fast SMA when SMA200 unavailable
SWING_SMA_ADAPTIVE_SLOW = 50 # Fallback slow SMA when SMA200 unavailable
SWING_USE_ADAPTIVE_MA = True # Use SMA20/50 when SMA200 not available (new stocks)

# ── Risk / Stops ───────────────────────────────────────────────────────────
SWING_STOP_MULTIPLIER = 2.0  # ATR multiplier for structure-based stop placement
SWING_PROFIT_R_MIN = 2.0     # Minimum risk-reward ratio for take-profit (2R)
SWING_PROFIT_R_MAX = 3.0     # Target take-profit at 3R
SWING_POSITION_SIZE_REDUCTION = 0.0  # No size reduction (trailing stop manages risk)

# ── Ratchet & Breakeven Trailing Stop ──────────────────────────────────────
SWING_RATCHET_ENABLED = True          # Tighten stop after significant gain
SWING_RATCHET_THRESHOLD = 0.20        # Gain % to trigger ratchet (20%)
SWING_RATCHET_STOP_MULTIPLIER = 1.5   # Tighter ATR multiplier after ratchet

# Lock profit based on Risk (R-Multiples)
SWING_LOCK_PROFIT_AT_R = 1.5          # When price reaches +1.5R in profit...
SWING_LOCK_PROFIT_TO_R = 1.0          # ...move the stop loss to +1.0R (locking 1R)

# ── Structure Detection ────────────────────────────────────────────────────
# How many recent daily bars to scan when looking for a swing high/low
SWING_STRUCTURE_LOOKBACK = 20

# "Near support/resistance" tolerance: price within this % of the level triggers
# the structure proximity check. 2% means within 2% of the SMA50 or swing level.
SWING_PROXIMITY_PCT = 0.02

# ── Retest Protection — Long Entries ──────────────────────────────────────
# These filters prevent entering a long just because price touched support again
# without showing any evidence of actual buying pressure.
#
# To make entries MORE aggressive (more trades, higher false-positive rate):
#   - Increase SWING_RETEST_MIN_WICK_PCT (accept smaller wicks)
#   - Set SWING_RETEST_REQUIRE_CLOSE_ABOVE = False
#   - Set SWING_RETEST_REQUIRE_HIGHER_LOW = False
#
# To make entries MORE conservative (fewer trades, cleaner setups):
#   - Decrease SWING_RETEST_MIN_WICK_PCT (require larger rejection wicks)
#   - Set SWING_RETEST_REQUIRE_CLOSE_ABOVE = True
#   - Set SWING_RETEST_REQUIRE_HIGHER_LOW = True

# Minimum lower-wick size as a fraction of the daily range.
# A candle with a 30% wick means the low pierced support but closed well above it.
# 0.0 = disabled (accept any close above support, no wick needed)
# 0.25 = require wick to be at least 25% of high-low range
SWING_RETEST_MIN_WICK_PCT_LONG = 0.25

# Require that the close is above the support level (not just a touch)
# True = candle must close above the SMA50/swing low — mandatory for clean longs
SWING_RETEST_REQUIRE_CLOSE_ABOVE_SUPPORT = True

# Require a higher low vs the previous bar (short-term momentum flip)
# True = current low must be above previous bar's low (price is being supported)
SWING_RETEST_REQUIRE_HIGHER_LOW = False

# Require volume to be at least N× the 20-period average on the rejection candle
# 0.0 = disabled. 1.2 = require 20% above avg volume (buying pressure confirmation)
SWING_RETEST_VOLUME_MULTIPLIER_LONG = 0.0

# ── Retest Protection — Short Entries ─────────────────────────────────────
# Mirror of the above for short setups: these filter out simple resistance touches
# that haven't shown actual selling pressure (rejection).

# Minimum upper-wick size as fraction of daily range (sellers pushed price back down)
# 0.0 = disabled. 0.25 = require upper wick >= 25% of high-low range
SWING_RETEST_MIN_WICK_PCT_SHORT = 0.25

# Require close to be BELOW the resistance level (failed to break through)
SWING_RETEST_REQUIRE_CLOSE_BELOW_RESISTANCE = True

# Require a lower high vs the previous bar (short-term rejection pattern)
SWING_RETEST_REQUIRE_LOWER_HIGH = False

# Volume confirmation for short rejection candles
# 0.0 = disabled. 1.2 = require 20% above avg volume on rejection bar
SWING_RETEST_VOLUME_MULTIPLIER_SHORT = 0.0

# ── RSI Filters ────────────────────────────────────────────────────────────
# For longs: RSI must be in a "dip" zone — not overbought, not yet recovering
SWING_RSI_LONG_MAX = 55   # Don't enter long if RSI is already high (>55 = extended)
SWING_RSI_LONG_MIN = 20   # Don't enter if RSI is catastrophically oversold (<20 = crash)
# For shorts: RSI must be in a "peak" zone — not oversold, not yet falling
SWING_RSI_SHORT_MIN = 45  # Don't enter short if RSI is already low (<45 = extended down)
SWING_RSI_SHORT_MAX = 80  # Don't enter short if RSI is wildly overbought (>80 = squeeze risk)

# --- Risk Management (AGGRESSIVE) ---
MAX_RISK_PER_TRADE = 0.04      # 4% of portfolio per trade (was 1% — 4x increase)
MAX_POSITION_VALUE_PCT = 0.20  # Max 20% of equity per single position (was 5% — 4x increase)
ATR_PERIOD = 14
RSI_PERIOD = 14

# --- Watchlists ---
# Day trade watchlist is now built dynamically from all Alpaca tradeable assets.
# These are the FALLBACK symbols used if the dynamic fetch fails.
DAY_TRADE_WATCHLIST_FALLBACK = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMD",
    "META", "AMZN", "GOOGL", "SPY", "QQQ"
]
# The active watchlist is populated at runtime by pre_market_setup.
# Starts with the fallback, then gets replaced by the dynamic universe.
DAY_TRADE_WATCHLIST = list(DAY_TRADE_WATCHLIST_FALLBACK)

SWING_TRADE_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "V", "UNH", "JNJ",
    "PG", "HD", "MA", "DIS", "NFLX",
    "ADBE", "CRM", "PYPL", "INTC", "CSCO"
]

# Commodity exposure available through Alpaca-listed ETF proxies. Native CFD/futures
# symbols depend on the broker and are rejected clearly by the data adapter.
COMMODITY_WATCHLIST = [
    "UNG",  # natural gas
    "USO",  # crude oil
    "GLD",  # gold
    "SLV",  # silver
    "DBA",  # agriculture basket
    "DBB",  # base metals
]
SWING_TRADE_WATCHLIST.extend(COMMODITY_WATCHLIST)

# --- Dynamic Day Trade Universe Filters ---
DAY_UNIVERSE_MIN_PRICE = 5.00          # Skip penny stocks below $5
DAY_UNIVERSE_MAX_PRICE = 10000.00      # No upper limit effectively
DAY_UNIVERSE_MIN_AVG_VOLUME = 500_000  # Minimum 500k avg daily volume
DAY_UNIVERSE_MAX_SYMBOLS = 500         # Cap the universe to top N by volume
DAY_UNIVERSE_EXCHANGES = ["NASDAQ", "NYSE", "ARCA", "AMEX"]
DAY_UNIVERSE_TOP_MOVERS = 50            # Fetch top N movers (by % change) from screener (Alpaca max ~50)
DAY_UNIVERSE_TOP_MOST_ACTIVE = 100      # Fetch top N most active (by volume) from screener
DAY_UNIVERSE_INCLUDE_MOVERS = True      # Merge top movers into universe
DAY_UNIVERSE_INCLUDE_MOST_ACTIVE = True # Merge most active into universe

# --- Scheduling ---
EOD_LIQUIDATION_TIME = "15:50"  # 3:50 PM ET
PRE_MARKET_SETUP_TIME = "09:00"
MARKET_OPEN_BUFFER_MINUTES = 5  # Start scanning at 9:35 AM (5 min after open)
HEALTH_CHECK_INTERVAL_SECONDS = 300
STOP_LOSS_CHECK_INTERVAL_SECONDS = 60

# --- Dashboard ---
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8000
NGROK_ENABLED = os.getenv("NGROK_ENABLED", "false").lower() == "true"
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")
DASHBOARD_POLL_INTERVAL_MS = 5000   # Frontend polls every 5 seconds
CHART_REFRESH_INTERVAL_MS = 60000   # Charts refresh every 60 seconds
HEARTBEAT_LOG_MAX_DISPLAY = 200
TRADE_HISTORY_PAGE_SIZE = 50

# --- Trade Decision Logs ---
TRADE_LOG_DIR = "trade_logs"           # Directory for per-trade JSON log files (trade_logs/<date>/<trade>.json)
TRADE_LOG_REJECTED = True              # Also log trades rejected by risk management (useful for analysis)

# --- Database ---
DATABASE_URL = "sqlite:///trading_bot.db"

# ══════════════════════════════════════════════
# PHASE 2 SETTINGS
# ══════════════════════════════════════════════

# --- Sentiment Analysis ---
ENABLE_SENTIMENT = True                   # Enable/disable sentiment layer
ALPACA_NEWS_LIMIT = 10                     # Max news articles to fetch per symbol
SENTIMENT_WEIGHT = 0.15                    # How much sentiment adjusts the signal score (0-1)
SENTIMENT_BULLISH_THRESHOLD = 0.2          # Score above this is considered bullish
SENTIMENT_BEARISH_THRESHOLD = -0.2         # Score below this is considered bearish
SENTIMENT_CACHE_TTL_SECONDS = 300          # Cache news sentiment for 5 minutes
SENTIMENT_KEYWORDS_POSITIVE = [
    "beats", "exceeds", "upgrade", "buy", "bullish", "growth", "record",
    "partnership", "acquisition", "profit", "revenue beat", "outperform",
    "strong", "surge", "rally", "breakout", "momentum", "upside",
]
SENTIMENT_KEYWORDS_NEGATIVE = [
    "misses", "downgrade", "sell", "bearish", "decline", "loss", "lawsuit",
    "investigation", "recall", "warning", "cut", "layoff", "restructuring",
    "weak", "plunge", "crash", "risk", "overvalued", "downside",
]

# --- Performance Analytics ---
ANALYTICS_LOOKBACK_DAYS = 90               # Default lookback for analytics calculations
RISK_FREE_RATE = 0.05                      # Annual risk-free rate for Sharpe ratio (5%)
MIN_TRADES_FOR_ANALYTICS = 5               # Minimum closed trades before showing analytics

# --- Alerts ---
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
ENABLE_WEBHOOK_ALERTS = os.getenv("ENABLE_WEBHOOK_ALERTS", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # Discord/Slack incoming webhook URL
ALERT_ON_TRADE = True                      # Alert on every trade entry/exit
ALERT_ON_STOP_LOSS = True                  # Alert on stop-loss triggers
ALERT_ON_ERROR = True                      # Alert on consecutive health check failures
ALERT_ON_DAILY_REPORT = True               # Alert with daily P&L summary

# --- Advanced Charting ---
CANDLESTICK_DEFAULT_BARS = 60              # Default number of bars for candlestick charts
CHART_INDICATORS = ["sma_50", "sma_200", "vwap", "rsi", "atr"]  # Indicators to show on charts
# --- Oscillation / Mean-Reversion Strategy ---
OSCILLATION_ENABLED = False
SWING_STRATEGY_MODE = "structure"  # "structure" or "oscillation"
OSCILLATION_LOOKBACK = 50
OSCILLATION_ENTRY_Z = 1.8
OSCILLATION_EXIT_Z = 0.25
OSCILLATION_RSI_PERIOD = 14
OSCILLATION_RSI_LOW = 35
OSCILLATION_RSI_HIGH = 65
OSCILLATION_MIN_CYCLE_SCORE = 0.55
OSCILLATION_MAX_TREND_STRENGTH = 0.35
OSCILLATION_MIN_CROSSINGS = 3
OSCILLATION_MAX_HOLD_BARS = 20
OSCILLATION_STOP_ATR = 2.0
OSCILLATION_TAKE_PROFIT_ATR = 2.5
OSCILLATION_FEE_BPS = 1.0
OSCILLATION_SLIPPAGE_BPS = 2.0
OSCILLATION_POSITION_PCT = 0.10
BACKTEST_MIN_HISTORY_BARS = 100
BACKTEST_MIN_HISTORY_YEARS = 0
BENCHMARK_SYMBOL = "QQQ"
MONTHLY_TARGET_PCT = 10.0
