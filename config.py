"""
PureAlpha Bot — Configuration
Dalio-inspired systematic NSE options trading.
Live: Zerodha TY3228 (₹39,430) | Paper: ₹1,00,000 → ₹10,00,000
"""

# ─── Capital & Risk ───────────────────────────────────────────────────────────
INITIAL_CAPITAL           = 39_430       # INR live Zerodha balance
PAPER_CAPITAL             = 1_00_000    # INR paper trading — journey ₹1L → ₹10L
MAX_CAPITAL_PER_TRADE_PCT = 40.0
MAX_OPEN_POSITIONS        = 1
MAX_DAILY_LOSS_PCT        = 5.0
BROKERAGE_PER_ORDER       = 20
MAX_PREMIUM_PER_LOT       = 600          # raised from 500 — ₹1L gives more room

# ─── Data Source ──────────────────────────────────────────────────────────────
# YFinance = primary (free, no token, bot auto-starts without manual login)
# Kite     = fallback for OHLCV + sole source for live option LTP / orders
USE_YFINANCE_PRIMARY = True
YFINANCE_SYMBOLS = {
    "NSE:NIFTY 50":   "^NSEI",
    "NSE:NIFTY BANK": "^NSEBANK",
    "NSE:INDIA VIX":  "^INDIAVIX",
}
# Black-Scholes params for paper-mode option premium estimation
BS_IV_NIFTY      = 0.14    # ~14% baseline implied vol for Nifty
BS_IV_BANKNIFTY  = 0.18    # ~18% for BankNifty
BS_RISK_FREE     = 0.065   # RBI repo rate

# ─── Auto-Scheduler ───────────────────────────────────────────────────────────
# Bot auto-starts at 09:15 and self-terminates at 15:45 on market days.
# Controlled by scheduler.py + macOS launchd (see com.purealpha.bot.plist).
BOT_START_TIME   = "09:15"
BOT_KILL_TIME    = "15:45"

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
# Risk:Reward = 1:2  →  SL = -40% of premium, Target = +80% of premium
OPTION_TARGET_MULT  = 0.80   # full exit when premium +80% of entry (2R)
OPTION_SL_MULT      = 0.40   # exit when premium -40% of entry (1R risk)
OPTION_TRAIL_START  = 0.50   # activate trail once premium +50% (raised from 0.35)
OPTION_TRAIL_LOCK   = 0.20   # once trail active, lock in +20% floor (raised from 0.15)
OPTION_MIDDAY_SL    = 0.20   # cut if down 20% after 12:00
MIDDAY_CUT_HOUR     = "12:00"

# Partial exit: sell 50% of position at 1R profit, then move SL to breakeven.
# This locks in guaranteed profit on half the position and removes downside
# on the rest — mathematically raises expectancy without cutting winners early.
PARTIAL_EXIT_ENABLED    = True
PARTIAL_EXIT_TRIGGER    = 0.40   # take 50% off when premium up 40% (= 1R)
PARTIAL_EXIT_FRACTION   = 0.50   # sell this fraction of remaining lots
PARTIAL_BREAKEVEN_AFTER = True   # after partial exit, move SL to entry premium + small buffer
PARTIAL_BREAKEVEN_BUFF  = 0.05   # move SL to entry * (1 + 0.05) after partial (slight positive)

# ─── Time Windows ─────────────────────────────────────────────────────────────
MARKET_OPEN         = "09:15"
MARKET_CLOSE        = "15:30"
SQUARE_OFF_TIME     = "15:10"   # mandatory exit before this
NO_ENTRY_AFTER      = "13:00"   # no new positions after 13:00 (was 14:00)
NO_ENTRY_BEFORE     = "09:25"   # skip first 10-min noise (was 09:20)

# Dead zone: 11:30–13:00 is typically low-momentum chop on NSE.
# Trades entered here fail significantly more often — block new entries.
SESSION_DEADZONE_START = "11:30"
SESSION_DEADZONE_END   = "13:00"
USE_SESSION_DEADZONE   = True
COUNTER_SIGNAL_EXIT   = True    # exit CE on bearish flip, exit PE on bullish flip
COUNTER_SIGNAL_THRESH = 0.28    # minimum final score of opposing signal to trigger exit
MIN_ENTRY_SCORE       = 0.32    # raw signal score must clear this before ML scaling
SCAN_INTERVAL_SECS    = 60

# ─── Market Regime Filter ─────────────────────────────────────────────────────
# Only enter when ATR is elevated (trending market) — skip quiet/choppy days.
ATR_TREND_FILTER    = True
ATR_TREND_MIN_RATIO = 0.80   # current ATR must be ≥ 80% of its 20-bar average

# ─── India VIX Regime Filter ─────────────────────────────────────────────────
# India VIX = market fear gauge. High VIX = wide spreads, gap risk, mean-reverting.
# VIX zones for option buyers:
#   < 12  : Too low — premiums cheap but directionality weak
#   12-20 : Sweet spot for directional option buying
#   > 20  : Elevated fear — widen SL multiplier, reduce position size
#   > 28  : Extreme fear — block new entries (event risk, gap danger)
USE_VIX_FILTER          = True
VIX_SYMBOL              = "NSE:INDIA VIX"   # Kite feed symbol
VIX_MAX_ENTRY           = 28.0   # block new entries above this
VIX_HIGH_THRESHOLD      = 20.0   # above this — reduce size by 30%
VIX_LOW_THRESHOLD       = 12.0   # below this — reduce size by 20% (low vol)
VIX_SIZE_REDUCTION_HIGH = 0.70   # multiplier when VIX > VIX_HIGH_THRESHOLD
VIX_SIZE_REDUCTION_LOW  = 0.80   # multiplier when VIX < VIX_LOW_THRESHOLD

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

# ─── Layer 4: Consensus (6 Voters, need 4/6) ────────────────────────────────
# PDF finding: 5/5 unanimous = 67% WR vs 27% for 4/5.
# Our 6 voters: Trend | Reversion | Breakout | Kronos | ML | HTF-Alignment
# HTF voter replaces Kronos dependency — always active, no PyTorch needed.
CONSENSUS_VOTES_REQUIRED = 4   # restored: 4/6 minimum for trade entry
N_VOTERS = 6

# Legacy compat
MIN_AGREEMENT_COUNT = 2        # min technical strategies agreeing (pre-consensus)

# ─── Layer 2: HTF (Higher Timeframe) Alignment Voter ─────────────────────────
# 15-minute candles fetched in parallel with 5-min candles.
# HTF voter checks that the 15-min trend agrees before confirming a 5-min signal.
# Always active — does not depend on Kronos/PyTorch.
HTF_INTERVAL        = "15minute"   # higher timeframe for confirmation
HTF_EMA_FAST        = 9            # fast EMA on 15-min
HTF_EMA_SLOW        = 21           # slow EMA on 15-min
HTF_SUPERTREND_MULT = 3.0
HTF_MIN_CONF        = 0.55         # minimum HTF confidence to cast a non-ABSTAIN vote

# ─── Layer 5: Market Filter ───────────────────────────────────────────────────
ADX_PERIOD          = 14
ADX_TREND_THRESHOLD = 20       # ADX >= 20 means trending market
VOLUME_FILTER_RATIO = 0.5      # lowered from 1.0 → 0.5 for index data
                               # Kite returns zero volume for spot indices — our
                               # volatility proxy underestimates vs real market volume

# ─── Layer 6: Confidence-Based Position Sizing ───────────────────────────────
# PDF Scenario B: 5/5 unanimous at 20% → +13% in 6 months, 1.4% max drawdown
# Updated for 6-voter system:
POSITION_SIZE_4_OF_6 = 15.0   # % of capital when 4/6 agree (minimum consensus)
POSITION_SIZE_5_OF_6 = 22.0   # % of capital when 5/6 agree (strong consensus)
POSITION_SIZE_6_OF_6 = 35.0   # % of capital when all 6 agree (maximum conviction)
# Legacy aliases (kept for backward compat)
POSITION_SIZE_3_OF_5 = 15.0
POSITION_SIZE_4_OF_5 = 22.0
POSITION_SIZE_5_OF_5 = 35.0

# ─── Layer 7: ATR-Based Risk ─────────────────────────────────────────────────
# Stop loss on underlying index (not fixed % of premium — adapts to volatility)
ATR_STOP_MULT        = 1.5    # stop = entry_spot ± 1.5 × ATR
ATR_TRAIL_TRIGGER    = 0.5    # activate trail once spot gains 0.5 × ATR
ATR_TRAIL_DIST       = 1.5    # trail by 1.5 × ATR from peak

# ─── ML Model ─────────────────────────────────────────────────────────────────
MIN_TRAINING_SAMPLES      = 20
MODEL_RETRAIN_EVERY       = 5
MODEL_PATH                = "models/purealpha_rf.pkl"
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# ─── Candle Data ──────────────────────────────────────────────────────────────
LIVE_INTERVAL    = "5minute"
HISTORICAL_DAYS  = 30

# ─── Dashboard ────────────────────────────────────────────────────────────────
# Port 5002 — original bot uses 5001, both run in parallel without conflict
DASHBOARD_PORT = 5002
DASHBOARD_HOST = "127.0.0.1"
STATUS_FILE    = "data/purealpha_status.json"

# ─── Logging / DB ─────────────────────────────────────────────────────────────
# Separate paths so both bots' data stay fully isolated
DB_PATH   = "data/purealpha_trades.db"
LOG_PATH  = "logs/purealpha.log"
LOG_LEVEL = "INFO"

# ─── Capital scaling guide ────────────────────────────────────────────────────
# ₹39,430  → 1 lot NIFTY only       (current)
# ₹60,000  → 1 lot NIFTY + BANKNIFTY
# ₹1,00,000 → 2 lots NIFTY / underlying
# ₹5,00,000 → 3 lots, tighter OTM offsets
# ₹10,00,000 → TARGET ACHIEVED
