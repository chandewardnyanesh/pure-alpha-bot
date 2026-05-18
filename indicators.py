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
    fast          = _ema(df["close"], MACD_FAST)
    slow          = _ema(df["close"], MACD_SLOW)
    df["macd"]    = fast - slow
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
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    hl   = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
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

        # Update bands — never widen against trend
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
    """Session VWAP with ±1σ bands. Assumes df is one trading session."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cumvol  = df["volume"].cumsum()
    cumtpv  = (typical * df["volume"]).cumsum()
    df["vwap"] = cumtpv / cumvol.replace(0, np.nan)

    # Rolling std of typical price for bands
    var = ((typical - df["vwap"]) ** 2 * df["volume"]).cumsum() / cumvol.replace(0, np.nan)
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


# ─── Master function ──────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator and return enriched DataFrame."""
    df = df.copy()
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_supertrend(df)
    df = add_vwap(df)
    df = add_volume_ma(df)
    df = add_momentum(df)
    return df


# ─── Feature vector for ML ────────────────────────────────────────────────────

FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_cross",
    "supertrend_dir", "vol_ratio", "momentum",
    "atr",
    # Derived ratios
]

def extract_features(row: pd.Series) -> np.ndarray:
    """Return a 1-D numpy feature vector from a single indicator row."""
    feats = []
    for col in FEATURE_COLS:
        feats.append(float(row.get(col, 0.0) or 0.0))

    # Additional engineered features
    feats.append(float((row.get("close", 0) - row.get("vwap", 0)) / max(row.get("atr", 1), 1e-6)))
    feats.append(float((row.get("close", 0) - row.get("ema_fast", 0)) / max(row.get("atr", 1), 1e-6)))
    feats.append(float((row.get("ema_fast", 0) - row.get("ema_slow", 0)) / max(row.get("atr", 1), 1e-6)))
    return np.array(feats, dtype=np.float32)
