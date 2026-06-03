"""
Layer 1-3: 5 Independent Signal Voters
=======================================
Architecture:
  Layer 1 — Indicators (SuperTrend, RSI, MACD, Volume, ATR)
  Layer 2 — Forecasting (Kronos)
  Layer 3 — ML Ensemble (XGBoost + RandomForest + LightGBM)

Each voter independently evaluates the market and returns:
  vote       : "BUY" | "SELL" | "ABSTAIN"
  confidence : 0.0 – 1.0
  reason     : human-readable explanation

Consensus layer (Layer 4) needs 4/5 voters to agree on direction.

PDF lesson: 5/5 unanimous = 67% WR; 4/5 = 27% WR.
Agreement quality matters more than individual score magnitude.
"""

import numpy as np
import pandas as pd
import logging
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    FVG_PROXIMITY_ATR,
)

logger = logging.getLogger(__name__)

VOTE_BUY     = "BUY"
VOTE_SELL    = "SELL"
VOTE_ABSTAIN = "ABSTAIN"


def _vote(action: str, confidence: float, reason: str) -> dict:
    return {"vote": action, "confidence": round(confidence, 3), "reason": reason}


# ─── Voter 1: Trend (SuperTrend + EMA + MACD) ────────────────────────────────

def trend_voter(df: pd.DataFrame) -> dict:
    """
    Evaluates macro trend direction using 3 sub-indicators.
    Each sub-indicator gets 1 point. Score = points / 3.
    Vote if ≥ 2/3 sub-indicators agree.

    Layer 1 indicators: SuperTrend, EMA alignment, MACD histogram.
    """
    if len(df) < 3:
        return _vote(VOTE_ABSTAIN, 0.0, "Trend: insufficient data")

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    bull_points = 0
    bear_points = 0
    notes = []

    # SuperTrend direction
    st_dir = row.get("supertrend_dir", 0)
    if st_dir == 1:
        bull_points += 1
        # Flip = extra conviction
        if prev.get("supertrend_dir", 1) == -1:
            bull_points += 0.5
            notes.append("ST_flip↑")
        else:
            notes.append("ST↑")
    elif st_dir == -1:
        bear_points += 1
        if prev.get("supertrend_dir", -1) == 1:
            bear_points += 0.5
            notes.append("ST_flip↓")
        else:
            notes.append("ST↓")

    # EMA alignment (fast > slow = bullish trend)
    if row.get("ema_fast", 0) > row.get("ema_slow", 0):
        bull_points += 1
        notes.append("EMA↑")
    else:
        bear_points += 1
        notes.append("EMA↓")

    # MACD histogram direction + acceleration
    hist      = row.get("macd_hist", 0)
    prev_hist = prev.get("macd_hist", 0)
    if hist > 0:
        bull_points += 1
        if hist > prev_hist:   # accelerating
            bull_points += 0.25
        notes.append("MACD↑")
    elif hist < 0:
        bear_points += 1
        if hist < prev_hist:
            bear_points += 0.25
        notes.append("MACD↓")

    max_score = 3.75   # max possible (with flip + acceleration bonus)
    bull_conf = bull_points / max_score
    bear_conf = bear_points / max_score

    if bull_conf >= 0.50 and bull_conf > bear_conf:
        return _vote(VOTE_BUY,  bull_conf, f"Trend BUY [{','.join(notes)}]")
    if bear_conf >= 0.50 and bear_conf > bull_conf:
        return _vote(VOTE_SELL, bear_conf, f"Trend SELL [{','.join(notes)}]")
    return _vote(VOTE_ABSTAIN, 0.0, f"Trend mixed [{','.join(notes)}]")


# ─── Voter 2: Mean Reversion (RSI + VWAP) ───────────────────────────────────

def reversion_voter(df: pd.DataFrame) -> dict:
    """
    Evaluates oversold/overbought extremes for counter-trend entries.
    Uses RSI extremes + VWAP band deviation.

    PDF: RSI Mean Reversion — deliberately conflicts with trend strategies.
    When this agrees with trend → much stronger signal.
    """
    if len(df) < 5:
        return _vote(VOTE_ABSTAIN, 0.0, "Reversion: insufficient data")

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    rsi    = row.get("rsi", 50)
    price  = row["close"]
    vwap   = row.get("vwap", price)
    vwap_lo = row.get("vwap_lo", vwap)
    vwap_up = row.get("vwap_up", vwap)
    band_w  = max(vwap_up - vwap_lo, 1e-6)
    deviation = (price - vwap) / (band_w / 2)

    bull_pts = 0
    bear_pts = 0

    # RSI extreme — uses widened 38/62 thresholds (was 35/65, too extreme, rarely fired)
    if rsi < RSI_OVERSOLD:
        rsi_score = 1.0 + (RSI_OVERSOLD - rsi) / max(RSI_OVERSOLD, 1) * 0.5
        bull_pts += rsi_score
        # RSI turning up = extra conviction
        if rsi > prev.get("rsi", rsi):
            bull_pts += 0.5
    elif rsi > RSI_OVERBOUGHT:
        rsi_score = 1.0 + (rsi - RSI_OVERBOUGHT) / max(100 - RSI_OVERBOUGHT, 1) * 0.5
        bear_pts += rsi_score
        if rsi < prev.get("rsi", rsi):
            bear_pts += 0.5

    # VWAP deviation
    if deviation < -1.0:
        bull_pts += min(abs(deviation) * 0.4, 1.0)
    elif deviation > 1.0:
        bear_pts += min(abs(deviation) * 0.4, 1.0)

    # Candle body confirmation
    if row["close"] > row["open"]:
        bull_pts += 0.3
    elif row["close"] < row["open"]:
        bear_pts += 0.3

    max_score = 3.3
    bull_conf = min(bull_pts / max_score, 1.0)
    bear_conf = min(bear_pts / max_score, 1.0)

    if bull_conf >= 0.45 and bull_conf > bear_conf:
        return _vote(VOTE_BUY,  bull_conf, f"Reversion BUY [RSI={rsi:.0f} dev={deviation:.2f}]")
    if bear_conf >= 0.45 and bear_conf > bull_conf:
        return _vote(VOTE_SELL, bear_conf, f"Reversion SELL [RSI={rsi:.0f} dev={deviation:.2f}]")
    return _vote(VOTE_ABSTAIN, 0.0, f"Reversion neutral [RSI={rsi:.0f}]")


# ─── Voter 3: Breakout (N-bar breakout + FVG + BB Squeeze) ──────────────────

def breakout_voter(df: pd.DataFrame) -> dict:
    """
    Evaluates momentum breakouts and institutional order flow.

    Sub-indicators:
    • N-bar price breakout with volume surge
    • Fair Value Gap (SMC imbalance) near price
    • Bollinger Band squeeze → expansion
    """
    lookback = 10
    if len(df) < lookback + 2:
        return _vote(VOTE_ABSTAIN, 0.0, "Breakout: insufficient data")

    window = df.iloc[-(lookback + 1):-1]
    row    = df.iloc[-1]
    price  = row["close"]
    atr    = max(float(row.get("atr", 1) or 1), 1.0)

    bull_pts = 0.0
    bear_pts = 0.0
    notes = []

    # ── Price breakout ───────────────────────────────────────────────────────
    pivot_high  = window["high"].max()
    pivot_low   = window["low"].min()
    vol_surge   = row.get("vol_ratio", 1.0) >= 1.5
    vol_bonus   = min((row.get("vol_ratio", 1.0) - 1.5) * 0.3, 0.5) if vol_surge else 0

    if price > pivot_high:
        bull_pts += 1.5 + vol_bonus
        notes.append(f"Break↑{pivot_high:.0f}")
    if price < pivot_low:
        bear_pts += 1.5 + vol_bonus
        notes.append(f"Break↓{pivot_low:.0f}")

    # ── Fair Value Gap ────────────────────────────────────────────────────────
    bull_fresh = float(row.get("fvg_bull_fresh", 999))
    bull_top   = float(row.get("fvg_bull_top", 0))
    bull_bot   = float(row.get("fvg_bull_bot", 0))
    if bull_fresh < 999 and bull_top > 0:
        gap_dist = min(abs(price - bull_bot), abs(price - bull_top))
        if gap_dist <= FVG_PROXIMITY_ATR * atr:
            fvg_score = 0.8 * (1 - gap_dist / (FVG_PROXIMITY_ATR * atr + 1e-6))
            fvg_score *= (1.0 if bull_fresh <= 5 else 0.6)
            bull_pts += fvg_score
            notes.append(f"FVG↑age={bull_fresh:.0f}")

    bear_fresh = float(row.get("fvg_bear_fresh", 999))
    bear_top   = float(row.get("fvg_bear_top", 0))
    bear_bot   = float(row.get("fvg_bear_bot", 0))
    if bear_fresh < 999 and bear_top > 0:
        gap_dist = min(abs(price - bear_bot), abs(price - bear_top))
        if gap_dist <= FVG_PROXIMITY_ATR * atr:
            fvg_score = 0.8 * (1 - gap_dist / (FVG_PROXIMITY_ATR * atr + 1e-6))
            fvg_score *= (1.0 if bear_fresh <= 5 else 0.6)
            bear_pts += fvg_score
            notes.append(f"FVG↓age={bear_fresh:.0f}")

    # ── BB Squeeze expansion ─────────────────────────────────────────────────
    prev_squeeze = df.iloc[-3].get("bb_squeeze", 0) if len(df) >= 3 else 0
    curr_squeeze = row.get("bb_squeeze", 0)
    if prev_squeeze == 1 and curr_squeeze == 0:   # squeeze just released
        if price > row.get("bb_up", price):
            bull_pts += 0.8
            notes.append("BB_squeeze↑")
        elif price < row.get("bb_lo", price):
            bear_pts += 0.8
            notes.append("BB_squeeze↓")

    # ── Normalise ─────────────────────────────────────────────────────────────
    max_score = 3.8
    bull_conf = min(bull_pts / max_score, 1.0)
    bear_conf = min(bear_pts / max_score, 1.0)

    if bull_conf >= 0.40 and bull_conf > bear_conf:
        return _vote(VOTE_BUY,  bull_conf, f"Breakout BUY [{','.join(notes)}]")
    if bear_conf >= 0.40 and bear_conf > bull_conf:
        return _vote(VOTE_SELL, bear_conf, f"Breakout SELL [{','.join(notes)}]")
    return _vote(VOTE_ABSTAIN, 0.0, f"Breakout neutral [{','.join(notes) or 'no break'}]")


# ─── Voter 4: Kronos (DL Price Forecast) ─────────────────────────────────────

def kronos_voter(df: pd.DataFrame, kronos_filter=None,
                 lookback: int = 100) -> dict:
    """
    Uses Kronos to forecast next 5 candles.
    Votes based on forecast direction alignment.

    Layer 2 — Forecasting.
    """
    if kronos_filter is None or not kronos_filter.is_available:
        return _vote(VOTE_ABSTAIN, 0.5, "Kronos: unavailable (fallback neutral)")

    try:
        # Kronos returns alignment score 0-1 for BUY direction
        # >0.55 = bullish, <0.45 = bearish, 0.45-0.55 = neutral
        raw = kronos_filter.get_alignment_score(df, "BUY", lookback)

        if raw >= 0.60:
            return _vote(VOTE_BUY,  raw,       f"Kronos BUY [forecast={raw:.2f}]")
        if raw <= 0.40:
            return _vote(VOTE_SELL, 1.0 - raw, f"Kronos SELL [forecast={raw:.2f}]")
        return _vote(VOTE_ABSTAIN, 0.5, f"Kronos neutral [forecast={raw:.2f}]")

    except Exception as e:
        logger.debug(f"Kronos voter error: {e}")
        return _vote(VOTE_ABSTAIN, 0.5, f"Kronos error: {e}")


# ─── Voter 5: ML Ensemble (XGBoost + RandomForest + LightGBM) ───────────────

def ml_voter(proposed_action: str, ml_ensemble=None,
             features: np.ndarray = None) -> dict:
    """
    Uses the trained ML ensemble to predict trade success probability.
    Votes to confirm (or abstain from) the proposed direction.

    Layer 3 — ML.
    """
    if ml_ensemble is None or features is None:
        return _vote(VOTE_ABSTAIN, 0.5, "ML: not initialised")

    try:
        proba = ml_ensemble.predict_success_proba(features)

        if proba >= 0.65:
            # ML says this trade is likely profitable → vote confirms direction
            return _vote(proposed_action, proba,
                         f"ML confirms {proposed_action} [proba={proba:.2f}]")
        if proba <= 0.35:
            # ML says likely unprofitable → vote against
            opp = VOTE_SELL if proposed_action == VOTE_BUY else VOTE_BUY
            return _vote(opp, 1 - proba,
                         f"ML rejects {proposed_action} [proba={proba:.2f}]")

        return _vote(VOTE_ABSTAIN, proba, f"ML uncertain [proba={proba:.2f}]")

    except Exception as e:
        logger.debug(f"ML voter error: {e}")
        return _vote(VOTE_ABSTAIN, 0.5, f"ML error: {e}")


# ─── Master: run all 5 voters ─────────────────────────────────────────────────

def run_all_voters(df: pd.DataFrame,
                   proposed_action: str,
                   kronos_filter=None,
                   ml_ensemble=None,
                   features: np.ndarray = None) -> dict:
    """
    Run all 5 voters and return their individual results.

    Returns:
      voters      : list of 5 voter result dicts
      buy_votes   : count of BUY votes
      sell_votes  : count of SELL votes
      abstentions : count of ABSTAIN votes
    """
    v1 = trend_voter(df)
    v2 = reversion_voter(df)
    v3 = breakout_voter(df)
    v4 = kronos_voter(df, kronos_filter)
    v5 = ml_voter(proposed_action, ml_ensemble, features)

    voters = [
        {"name": "Trend",     **v1},
        {"name": "Reversion", **v2},
        {"name": "Breakout",  **v3},
        {"name": "Kronos",    **v4},
        {"name": "ML",        **v5},
    ]

    buy_votes  = sum(1 for v in voters if v["vote"] == VOTE_BUY)
    sell_votes = sum(1 for v in voters if v["vote"] == VOTE_SELL)
    abstain    = sum(1 for v in voters if v["vote"] == VOTE_ABSTAIN)

    return {
        "voters":      voters,
        "buy_votes":   buy_votes,
        "sell_votes":  sell_votes,
        "abstentions": abstain,
    }
