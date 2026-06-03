"""
Technical indicators — all computed on a pandas DataFrame with columns:
  open, high, low, close, volume
Returns the same DataFrame with extra indicator columns appended.
"""

import numpy as np
import pandas as pd
from config import (
    EMA_FAST, EMA_SLOW, EMA_TREND,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD,
    SUPERTREND_PERIOD, SUPERTREND_MULT, VWAP_BAND_STD,
    FVG_LOOKBACK, FVG_FRESH_BARS, ADX_PERIOD,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


# ─── Individual Indicators ────────────────────────────────────────────────────

def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_fast"]  = _ema(df["close"], EMA_FAST)
    df["ema_slow"]  = _ema(df["close"], EMA_SLOW)
    df["ema_trend"] = _ema(df["close"], EMA_TREND)
    df["ema_cross"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    fast           = _ema(df["close"], MACD_FAST)
    slow           = _ema(df["close"], MACD_SLOW)
    df["macd"]     = fast - slow
    df["macd_sig"] = _ema(df["macd"], MACD_SIGNAL)
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    mid           = _sma(df["close"], BB_PERIOD)
    std           = df["close"].rolling(BB_PERIOD).std()
    df["bb_mid"]  = mid
    df["bb_up"]   = mid + BB_STD * std
    df["bb_lo"]   = mid - BB_STD * std
    df["bb_pct"]  = (df["close"] - df["bb_lo"]) / (df["bb_up"] - df["bb_lo"]).replace(0, np.nan)
    # Squeeze: BB width in bottom 30% of 50-bar range (lesson from PDF's BB Squeeze strategy)
    bb_width      = df["bb_up"] - df["bb_lo"]
    bb_width_min  = bb_width.rolling(50, min_periods=10).min()
    bb_width_max  = bb_width.rolling(50, min_periods=10).max()
    bb_width_pct  = (bb_width - bb_width_min) / (bb_width_max - bb_width_min).replace(0, np.nan)
    df["bb_squeeze"] = (bb_width_pct < 0.30).astype(int)   # 1 = squeeze active
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    hl   = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
    # ATR ratio: current ATR vs 20-bar rolling average — proxy for "trending vs ranging"
    # Lesson from PDF: signals on quiet timeframes (5m BTC) had 0% WR.
    # atr_ratio > 1 = expanding volatility, < 0.8 = quiet/ranging, skip entry.
    atr_avg           = df["atr"].rolling(20, min_periods=5).mean()
    df["atr_ratio"]   = df["atr"] / atr_avg.replace(0, np.nan)
    return df


def add_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    """SuperTrend — direction +1 (bullish) / -1 (bearish)."""
    if "atr" not in df.columns:
        df = add_atr(df)

    hl2   = (df["high"] + df["low"]) / 2
    upper = hl2 + SUPERTREND_MULT * df["atr"]
    lower = hl2 - SUPERTREND_MULT * df["atr"]

    supertrend = pd.Series(np.nan, index=df.index)
    direction  = pd.Series(1,      index=df.index)

    for i in range(1, len(df)):
        prev_st  = supertrend.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        close    = df["close"].iloc[i]

        u = upper.iloc[i]
        l = lower.iloc[i]
        if not np.isnan(prev_st):
            if prev_dir == 1:
                l = max(l, prev_st)
            else:
                u = min(u, prev_st)

        if prev_dir == 1:
            if close < l:
                direction.iloc[i] = -1
                supertrend.iloc[i] = u
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = l
        else:
            if close > u:
                direction.iloc[i] = 1
                supertrend.iloc[i] = l
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = u

    df["supertrend"]     = supertrend
    df["supertrend_dir"] = direction
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP with ±1σ bands."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cumvol  = df["volume"].cumsum()
    cumtpv  = (typical * df["volume"]).cumsum()
    df["vwap"] = cumtpv / cumvol.replace(0, np.nan)

    var   = ((typical - df["vwap"]) ** 2 * df["volume"]).cumsum() / cumvol.replace(0, np.nan)
    sigma = np.sqrt(var)
    df["vwap_up"] = df["vwap"] + VWAP_BAND_STD * sigma
    df["vwap_lo"] = df["vwap"] - VWAP_BAND_STD * sigma
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["vol_ma"]    = _sma(df["volume"], period)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)
    return df


def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df["momentum"] = df["close"].pct_change(period) * 100
    return df


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Directional Index (ADX) — Layer 5 Market Filter.
    ADX >= 20: trending market (good for our strategies)
    ADX < 20: ranging/choppy (skip entry)
    Also computes +DI and -DI for directional bias.
    """
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Where +DM > -DM, keep +DM else 0
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    atr_raw = df["atr"] if "atr" in df.columns else (high - low)
    plus_di  = 100 * (plus_dm.ewm(com=ADX_PERIOD-1, adjust=False).mean() /
                      atr_raw.ewm(com=ADX_PERIOD-1, adjust=False).mean().replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(com=ADX_PERIOD-1, adjust=False).mean() /
                      atr_raw.ewm(com=ADX_PERIOD-1, adjust=False).mean().replace(0, np.nan))
    dx       = (100 * (plus_di - minus_di).abs() /
                (plus_di + minus_di).replace(0, np.nan))
    df["adx"]      = dx.ewm(com=ADX_PERIOD-1, adjust=False).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    return df


def add_fair_value_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gap (SMC / Smart Money Concepts).

    Inspired by PDF Section 3.4:
    • Bullish FVG: candle[i].low > candle[i-2].high  → price gap above candle i-2
    • Bearish FVG: candle[i].high < candle[i-2].low  → price gap below candle i-2
    • Fill detection: close-based (not wick-based) — prevents immediate false fills
    • Freshness: gaps older than FVG_FRESH_BARS get zero weight

    Columns added:
      fvg_bull_gap_top, fvg_bull_gap_bot — most recent unfilled bullish FVG edges
      fvg_bear_gap_top, fvg_bear_gap_bot — most recent unfilled bearish FVG edges
      fvg_bull_fresh, fvg_bear_fresh     — bars since last FVG formed (999 = none)
    """
    n = len(df)
    bull_gap_top = np.zeros(n)
    bull_gap_bot = np.zeros(n)
    bear_gap_top = np.zeros(n)
    bear_gap_bot = np.zeros(n)
    bull_fresh   = np.full(n, 999)
    bear_fresh   = np.full(n, 999)

    # Find FVG formation candles
    bull_fvgs = []  # list of (bar_index, top, bot)
    bear_fvgs = []

    for i in range(2, n):
        # Bullish FVG: candle[i] has low above candle[i-2] high
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            top = df["low"].iloc[i]
            bot = df["high"].iloc[i - 2]
            bull_fvgs.append((i, top, bot, False))   # (idx, top, bot, filled)

        # Bearish FVG: candle[i] has high below candle[i-2] low
        if df["high"].iloc[i] < df["low"].iloc[i - 2]:
            top = df["low"].iloc[i - 2]
            bot = df["high"].iloc[i]
            bear_fvgs.append((i, top, bot, False))

    # For each bar, find the most recent unfilled FVG
    for bar in range(n):
        close = df["close"].iloc[bar]

        # Most recent unfilled bullish FVG
        for fvg_bar, top, bot, filled in reversed(bull_fvgs):
            if fvg_bar >= bar:
                continue
            age = bar - fvg_bar
            # Check if filled (close went below bot = gap filled)
            if close < bot:
                break   # filled — stop looking
            if age <= FVG_FRESH_BARS:
                bull_gap_top[bar] = top
                bull_gap_bot[bar] = bot
                bull_fresh[bar]   = age
                break

        # Most recent unfilled bearish FVG
        for fvg_bar, top, bot, filled in reversed(bear_fvgs):
            if fvg_bar >= bar:
                continue
            age = bar - fvg_bar
            if close > top:
                break   # filled
            if age <= FVG_FRESH_BARS:
                bear_gap_top[bar] = top
                bear_gap_bot[bar] = bot
                bear_fresh[bar]   = age
                break

    df["fvg_bull_top"]   = bull_gap_top
    df["fvg_bull_bot"]   = bull_gap_bot
    df["fvg_bear_top"]   = bear_gap_top
    df["fvg_bear_bot"]   = bear_gap_bot
    df["fvg_bull_fresh"] = bull_fresh
    df["fvg_bear_fresh"] = bear_fresh
    return df


# ─── Master function ──────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator and return enriched DataFrame."""
    df = df.copy()
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_supertrend(df)
    df = add_vwap(df)
    df = add_volume_ma(df)
    df = add_momentum(df)
    df = add_fair_value_gap(df)
    return df


# ─── Feature vector for ML ────────────────────────────────────────────────────
# Expanded from 11 → 16 features.
# New features (lessons from 6 trades + PDF):
#   atr_ratio    — was the market trending or quiet at entry?
#   bb_squeeze   — was volatility contracting before breakout?
#   time_sin/cos — cyclical time encoding (captures morning vs afternoon bias)
#   fvg_signal   — was there a fresh FVG supporting the trade?
FEATURE_COLS = [
    "rsi",           # 0
    "macd_hist",     # 1
    "bb_pct",        # 2
    "ema_cross",     # 3
    "supertrend_dir",# 4
    "vol_ratio",     # 5
    "momentum",      # 6
    "atr",           # 7
    "atr_ratio",     # 8  ← NEW: trending vs ranging filter
    "bb_squeeze",    # 9  ← NEW: volatility state
]

def extract_features(row: pd.Series, extra: dict = None) -> np.ndarray:
    """
    Return a 1-D numpy feature vector from a single indicator row.
    extra: optional dict with keys like 'agreement_count', 'kronos_score'
    """
    feats = []
    for col in FEATURE_COLS:
        feats.append(float(row.get(col, 0.0) or 0.0))

    # Engineered ratios (price vs indicators, normalised by ATR)
    atr = max(float(row.get("atr", 1) or 1), 1e-6)
    feats.append(float((row.get("close", 0) - row.get("vwap", 0)) / atr))         # 10: vwap_dist
    feats.append(float((row.get("close", 0) - row.get("ema_fast", 0)) / atr))     # 11: close_vs_ema_fast
    feats.append(float((row.get("ema_fast", 0) - row.get("ema_slow", 0)) / atr))  # 12: ema_fast_vs_slow

    # Cyclical time-of-day encoding (captures morning momentum vs afternoon chop)
    # Maps market hours 09:15–15:30 → 0..1, then sin/cos for cyclical continuity
    import datetime
    now = datetime.datetime.now()
    market_minutes = (now.hour * 60 + now.minute) - (9 * 60 + 15)
    session_length = 375.0   # 09:15 to 15:30
    t = max(0.0, min(1.0, market_minutes / session_length))
    feats.append(float(np.sin(2 * np.pi * t)))   # 13: time_sin
    feats.append(float(np.cos(2 * np.pi * t)))   # 14: time_cos

    # FVG presence: +1 = fresh bullish FVG near price, -1 = fresh bearish FVG
    fvg_signal = 0.0
    if row.get("fvg_bull_fresh", 999) < 999 and row.get("fvg_bull_top", 0) > 0:
        fvg_signal = 1.0
    elif row.get("fvg_bear_fresh", 999) < 999 and row.get("fvg_bear_top", 0) > 0:
        fvg_signal = -1.0
    feats.append(fvg_signal)   # 15: fvg_signal

    # Extra context passed in from bot (agreement_count, kronos_score)
    if extra:
        feats.append(float(extra.get("agreement_count", 0)))   # 16
        feats.append(float(extra.get("kronos_score", 0.5)))    # 17

    return np.array(feats, dtype=np.float32)
