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
MAX_PREMIUM_PER_LOT       = 500          # skip option if premium × lot_size > budget
                                         # prevents Trade-6 type (₹937×30=₹28k = 71% capital)

# ─── Options Instruments ──────────────────────────────────────────────────────
# Lot sizes VERIFIED from live NFO instrument dump (May 2026 SEBI revision):
#   NIFTY = 65,  BANKNIFTY = 30
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
# Lesson from PDF: agreement count matters more than individual score.
# High agreement (4-5 strategies) → use tighter OTM (2-OTM, more delta)
# Normal agreement (2-3 strategies) → use 3-OTM
#
STRIKE_OFFSET_STRONG    = 2      # strikes OTM when score ≥ STRONG_SIGNAL_SCORE
STRIKE_OFFSET_WEAK      = 3      # strikes OTM otherwise
STRONG_SIGNAL_SCORE     = 0.72
AGREEMENT_STRIKE_THRESH = 4      # agreement count ≥ this → auto upgrade to 2-OTM
                                 # (equivalent to PDF's 4/5 unanimity bonus)

# ─── Expiry Rules ─────────────────────────────────────────────────────────────
SKIP_EXPIRY_DAYS_BEFORE = 2

# ─── Options P&L Management ───────────────────────────────────────────────────
OPTION_TARGET_MULT  = 0.80   # exit when premium +80% of entry
OPTION_SL_MULT      = 0.40   # exit when premium -40% of entry
OPTION_TRAIL_START  = 0.35   # activate trail once premium +35%
OPTION_TRAIL_LOCK   = 0.15   # once trail active, lock in +15% floor
OPTION_MIDDAY_SL    = 0.20   # tightened from 0.30 — cut if down 20% after 12:00
                             # prevents EOD bagholding (Trade-2 lost ₹4,094 holding all day)
MIDDAY_CUT_HOUR     = "12:00"  # was implicit 13:00, now explicit and earlier

# ─── Time Windows ─────────────────────────────────────────────────────────────
MARKET_OPEN         = "09:15"
MARKET_CLOSE        = "15:30"
SQUARE_OFF_TIME     = "15:10"   # mandatory exit before this
NO_ENTRY_AFTER      = "13:00"   # no new positions after 13:00 (was 14:00)
NO_ENTRY_BEFORE     = "09:20"   # skip first 5-min noise candle
COUNTER_SIGNAL_EXIT   = True    # exit CE on bearish flip, exit PE on bullish flip
COUNTER_SIGNAL_THRESH = 0.28    # minimum final score of opposing signal to trigger exit
MIN_ENTRY_SCORE       = 0.32    # raw signal score must clear this before ML scaling
SCAN_INTERVAL_SECS    = 60

# ─── Market Regime Filter ─────────────────────────────────────────────────────
# Only enter when ATR is elevated (trending market) — skip quiet/choppy days.
# Lesson from PDF: strategies designed for trending timeframes; 5m signals on BTC
# had 0% win rate vs 67% on 15m. ATR ratio proxies this for intraday.
ATR_TREND_FILTER    = True
ATR_TREND_MIN_RATIO = 0.80   # current ATR must be ≥ 80% of its 20-bar average
                              # below this = market ranging, signals are noise

# ─── Kronos Deep-Learning Forecasting Filter ──────────────────────────────────
# Kronos (NeoQuasar/Kronos-small) forecasts next N candles from recent OHLCV.
# We use it as a pre-trade gate: if Kronos forecast direction contradicts the
# technical signal, reduce conviction (don't block, just penalise the score).
USE_KRONOS_FILTER      = True
KRONOS_LOOKBACK        = 100    # candles fed to Kronos as context
KRONOS_PRED_LEN        = 5      # candles to forecast (25 min on 5-min bars)
KRONOS_MIN_ALIGNMENT   = 0.55   # Kronos bull-fraction must exceed this to confirm BUY
                                 # (or be below 1-this to confirm SELL)
KRONOS_SCORE_WEIGHT    = 0.15   # contribution of Kronos to final blended score

# ─── Technical Indicators ─────────────────────────────────────────────────────
EMA_FAST          = 9
EMA_SLOW          = 21
EMA_TREND         = 50
RSI_PERIOD        = 14
RSI_OVERSOLD      = 38   # widened from 35 — fires more often in moderate oversold
RSI_OVERBOUGHT    = 62   # widened from 65 — fires more often in moderate overbought
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BB_PERIOD         = 20
BB_STD            = 2.0
ATR_PERIOD        = 14
SUPERTREND_PERIOD = 10
SUPERTREND_MULT   = 3.0
VWAP_BAND_STD     = 1.0
FVG_LOOKBACK      = 30          # bars to scan for fresh Fair Value Gaps
FVG_FRESH_BARS    = 15          # gap older than this gets no score
FVG_PROXIMITY_ATR = 0.50        # price must be within this × ATR of the gap edge

# ─── Strategy Weights ─────────────────────────────────────────────────────────
# Lesson from PDF: unanimity (all agree) is qualitatively different — 67% WR vs 27%.
# Our equivalent: agreement_count ≥ 4 strategies → treat like unanimity.
# FVG added as 5th strategy (like PDF's 5-strategy setup).
STRATEGY_WEIGHTS = {
    "ema_crossover":  0.25,
    "vwap_reversion": 0.10,
    "supertrend":     0.40,   # best for directional index moves
    "breakout":       0.10,
    "fvg":            0.15,   # new: Fair Value Gap (SMC-based)
}
SIGNAL_THRESHOLD = 0.28       # min blended score to fire a trade

# ─── Layer 4: Consensus (5 Voters, need 4/5) ────────────────────────────────
# PDF finding: 5/5 unanimous = 67% WR vs 27% for 4/5.
# Our 5 voters: Trend | Reversion | Breakout | Kronos | ML-Ensemble
CONSENSUS_VOTES_REQUIRED = 3   # TEMP: lowered from 4 → 3 until Kronos is live
                               # Kronos always abstains without PyTorch → max was 3/5
                               # Restore to 4 once Kronos confirms it votes
N_VOTERS = 5

# Legacy compat
MIN_AGREEMENT_COUNT = 2        # min technical strategies agreeing (pre-consensus)

# ─── Layer 5: Market Filter ───────────────────────────────────────────────────
ADX_PERIOD          = 14
ADX_TREND_THRESHOLD = 20       # ADX >= 20 means trending market
VOLUME_FILTER_RATIO = 0.5      # lowered from 1.0 → 0.5 for index data
                               # Kite returns zero volume for spot indices — our
                               # volatility proxy underestimates vs real market volume

# ─── Layer 6: Confidence-Based Position Sizing ───────────────────────────────
# PDF Scenario B: 5/5 unanimous at 20% → +13% in 6 months, 1.4% max drawdown
POSITION_SIZE_3_OF_5 = 15.0   # % of capital when 3/5 agree (conservative — Kronos absent)
POSITION_SIZE_4_OF_5 = 20.0   # % of capital when 4/5 agree
POSITION_SIZE_5_OF_5 = 35.0   # % of capital when all 5 agree (high conviction)

# ─── Layer 7: ATR-Based Risk ─────────────────────────────────────────────────
# Stop loss on underlying index (not fixed % of premium — adapts to volatility)
ATR_STOP_MULT        = 1.5    # stop = entry_spot ± 1.5 × ATR
ATR_TRAIL_TRIGGER    = 0.5    # activate trail once spot gains 0.5 × ATR
ATR_TRAIL_DIST       = 1.5    # trail by 1.5 × ATR from peak

# ─── ML Model ─────────────────────────────────────────────────────────────────
MIN_TRAINING_SAMPLES      = 20
MODEL_RETRAIN_EVERY       = 5
MODEL_PATH                = "models/rf_model.pkl"
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# ─── Candle Data ──────────────────────────────────────────────────────────────
LIVE_INTERVAL    = "5minute"
HISTORICAL_DAYS  = 30

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_PORT = 5001
DASHBOARD_HOST = "127.0.0.1"
STATUS_FILE    = "data/status.json"

# ─── Logging / DB ─────────────────────────────────────────────────────────────
DB_PATH   = "data/trades.db"
LOG_PATH  = "logs/bot.log"
LOG_LEVEL = "INFO"

# ─── Capital scaling guide ────────────────────────────────────────────────────
# ₹39,430  → 1 lot NIFTY only       (current)
# ₹60,000  → 1 lot NIFTY + BANKNIFTY
# ₹1,00,000 → 2 lots NIFTY / underlying
# ₹5,00,000 → 3 lots, tighter OTM offsets
# ₹10,00,000 → TARGET ACHIEVED
