"""
AI Options Trading Bot — Configuration
Calibrated against live Zerodha account TY3228 (₹39,430 capital, May 2026)
"""

# ─── Capital & Risk ───────────────────────────────────────────────────────────
INITIAL_CAPITAL           = 39_430       # INR — verified from Kite margins
MAX_CAPITAL_PER_TRADE_PCT = 40.0         # 40% → ~₹15,772 budget → 1 lot Nifty 3-OTM
MAX_OPEN_POSITIONS        = 1            # 1 trade at a time (capital constraint)
MAX_DAILY_LOSS_PCT        = 5.0          # hard stop: ₹1,971 daily loss limit
BROKERAGE_PER_ORDER       = 20           # flat ₹20 Zerodha

# ─── Options Instruments ──────────────────────────────────────────────────────
# Lot sizes VERIFIED from live NFO instrument dump (May 2026 SEBI revision):
#   NIFTY = 65,  BANKNIFTY = 30
# Only NIFTY to start — most liquid, tightest spreads.
# BankNifty re-enabled below once capital grows past ₹60,000.
OPTION_UNDERLYINGS = [
    {
        "name":        "NIFTY",
        "index":       "NSE:NIFTY 50",   # spot feed for indicators
        "exchange":    "NFO",
        "lot_size":    65,                # VERIFIED
        "strike_step": 50,
    },
    {
        "name":        "BANKNIFTY",
        "index":       "NSE:NIFTY BANK",
        "exchange":    "NFO",
        "lot_size":    30,                # VERIFIED
        "strike_step": 100,
    },
]

# ─── Strike Selection ─────────────────────────────────────────────────────────
# We BUY options only (defined risk). Never naked sell.
#
# With current capital:
#   3-OTM Nifty ≈ ₹225 × 65 = ₹14,625/lot  (37% of capital)  ← used for normal signals
#   2-OTM Nifty ≈ ₹250 × 65 = ₹16,250/lot  (41% of capital)  ← used for strong signals
#
STRIKE_OFFSET_STRONG  = 2      # strikes OTM when signal score ≥ STRONG_SIGNAL_SCORE
STRIKE_OFFSET_WEAK    = 3      # strikes OTM otherwise
STRONG_SIGNAL_SCORE   = 0.72

# ─── Expiry Rules ─────────────────────────────────────────────────────────────
# Skip to next expiry if ≤ this many days remain (theta decay too aggressive)
SKIP_EXPIRY_DAYS_BEFORE = 2
# Never hold overnight — all positions squared off before SQUARE_OFF_TIME

# ─── Options P&L Management ───────────────────────────────────────────────────
OPTION_TARGET_MULT  = 0.80   # exit when premium +80% of entry (near double)
OPTION_SL_MULT      = 0.40   # exit when premium -40% of entry (tighter)
OPTION_TRAIL_START  = 0.35   # activate trail once premium +35%
OPTION_TRAIL_LOCK   = 0.15   # once trail active, lock in +15% floor
OPTION_MIDDAY_SL    = 0.30   # if down 30% by 13:00, cut loss (no EOD bagholding)

# ─── Time Windows ─────────────────────────────────────────────────────────────
MARKET_OPEN         = "09:15"
MARKET_CLOSE        = "15:30"
SQUARE_OFF_TIME     = "15:10"   # mandatory exit before this
NO_ENTRY_AFTER      = "14:00"   # no new positions after
NO_ENTRY_BEFORE     = "09:20"   # skip first 5-min noise candle
SCAN_INTERVAL_SECS  = 60

# ─── Technical Indicators ─────────────────────────────────────────────────────
EMA_FAST          = 9
EMA_SLOW          = 21
EMA_TREND         = 50
RSI_PERIOD        = 14
RSI_OVERSOLD      = 35
RSI_OVERBOUGHT    = 65
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BB_PERIOD         = 20
BB_STD            = 2.0
ATR_PERIOD        = 14
SUPERTREND_PERIOD = 10
SUPERTREND_MULT   = 3.0
VWAP_BAND_STD     = 1.0

# ─── Strategy Weights ─────────────────────────────────────────────────────────
# Options profit most from momentum — SuperTrend + Breakout weighted highest.
STRATEGY_WEIGHTS = {
    "ema_crossover":  0.30,   # works well on index data (no volume needed)
    "vwap_reversion": 0.15,
    "supertrend":     0.45,   # best for directional index moves
    "breakout":       0.10,   # reduced — index volume data is 0 from Kite
}
SIGNAL_THRESHOLD = 0.28      # 2-strategy confluence × ML(0.85) clears this bar

# ─── ML Model ─────────────────────────────────────────────────────────────────
MIN_TRAINING_SAMPLES      = 20
MODEL_RETRAIN_EVERY       = 5
MODEL_PATH                = "models/rf_model.pkl"
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# ─── Candle Data ──────────────────────────────────────────────────────────────
LIVE_INTERVAL    = "5minute"
HISTORICAL_DAYS  = 30

# ─── Logging / DB ─────────────────────────────────────────────────────────────
DB_PATH   = "data/trades.db"
LOG_PATH  = "logs/bot.log"
LOG_LEVEL = "INFO"

# ─── Capital scaling guide (auto-unlock more underlyings / lots as you grow) ──
# ₹39,430  → 1 lot NIFTY only       (current)
# ₹60,000  → 1 lot NIFTY + BANKNIFTY
# ₹1,00,000 → 2 lots NIFTY / underlying
# ₹5,00,000 → 3 lots, tighter OTM offsets
# ₹10,00,000 → TARGET ACHIEVED
