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
    HTF_EMA_FAST, HTF_EMA_SLOW, HTF_SUPERTREND_MULT, HTF_MIN_CONF,
)

logger = logging.getLogger(__name__)

VOTE_BUY     = "BUY"
VOTE_SELL    = "SELL"
VOTE_ABSTAIN = "ABSTAIN"


def _vote(action: str, confidence: float, reason: str) -> dict:
    return {"vote": action, "confidence": round(confidence, 3), "reason": reason}


# ─── Voter 1: Trend (SuperTrend + EMA + MACD) ────────────────────────────────

def _detect_macd_divergence(df: pd.DataFrame, lookback: int = 10) -> str:
    """
    Detect bullish/bearish MACD divergence over the last `lookback` bars.

    Bullish divergence: price making lower lows while MACD histogram making
    higher lows  → momentum turning before price confirms.
    Bearish divergence: price making higher highs while MACD making lower highs.

    Returns: "bullish", "bearish", or "none"
    """
    if len(df) < lookback + 2:
        return "none"

    window = df.iloc[-lookback:]
    closes = window["close"].values
    hists  = window["macd_hist"].values

    price_ll = closes[-1] < closes[:-1].min()          # current close is new low
    price_hh = closes[-1] > closes[:-1].max()          # current close is new high
    macd_ll  = hists[-1]  < hists[:-1].min()
    macd_hh  = hists[-1]  > hists[:-1].max()

    # Bullish divergence: price at new low but MACD NOT at new low
    if price_ll and not macd_ll and hists[-1] > hists[-2]:
        return "bullish"

    # Bearish divergence: price at new high but MACD NOT at new high
    if price_hh and not macd_hh and hists[-1] < hists[-2]:
        return "bearish"

    return "none"


def trend_voter(df: pd.DataFrame) -> dict:
    """
    Evaluates macro trend direction using 3 sub-indicators + MACD divergence.

    Sub-checks (each worth 1 point):
    • SuperTrend direction / flip bonus
    • EMA-9 vs EMA-21 alignment
    • MACD histogram sign + acceleration bonus
    • MACD divergence bonus (adds 0.5 for confirmation, subtracts 0.5 for warning)

    Layer 1 indicators: SuperTrend, EMA alignment, MACD histogram, divergence.
    """
    if len(df) < 3:
        return _vote(VOTE_ABSTAIN, 0.0, "Trend: insufficient data")

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    bull_points = 0.0
    bear_points = 0.0
    notes = []

    # SuperTrend direction
    st_dir = row.get("supertrend_dir", 0)
    if st_dir == 1:
        bull_points += 1
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
        if hist > prev_hist:
            bull_points += 0.25
        notes.append("MACD↑")
    elif hist < 0:
        bear_points += 1
        if hist < prev_hist:
            bear_points += 0.25
        notes.append("MACD↓")

    # MACD divergence check — adds conviction when divergence agrees, warns when it conflicts
    divergence = _detect_macd_divergence(df)
    if divergence == "bullish":
        bull_points += 0.5    # momentum turning up before price = strong early signal
        notes.append("DIV↑")
    elif divergence == "bearish":
        bear_points += 0.5
        notes.append("DIV↓")

    max_score = 4.25   # updated max (3.75 + 0.5 divergence bonus)
    bull_conf = bull_points / max_score
    bear_conf = bear_points / max_score

    if bull_conf >= 0.48 and bull_conf > bear_conf:
        return _vote(VOTE_BUY,  bull_conf, f"Trend BUY [{','.join(notes)}]")
    if bear_conf >= 0.48 and bear_conf > bull_conf:
        return _vote(VOTE_SELL, bear_conf, f"Trend SELL [{','.join(notes)}]")
    return _vote(VOTE_ABSTAIN, 0.0, f"Trend mixed [{','.join(notes)}]")


# ─── Voter 2: Mean Reversion (RSI + VWAP) ───────────────────────────────────

def reversion_voter(df: pd.DataFrame) -> dict:
    """
    Evaluates oversold/overbought extremes for counter-trend entries.
    Uses RSI extremes + VWAP band deviation + candle confirmation.

    Thresholds tightened back to 35/65: the 38/62 setting generated too many
    false reversion signals in trending markets, conflicting with trend voters
    and reducing consensus quality. Better to fire rarely but accurately.

    When this voter agrees with the trend voters → much stronger signal.
    When it conflicts → it contributes to blocking bad trend-following entries.
    """
    if len(df) < 5:
        return _vote(VOTE_ABSTAIN, 0.0, "Reversion: insufficient data")

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    rsi     = row.get("rsi", 50)
    price   = row["close"]
    vwap    = row.get("vwap", price)
    vwap_lo = row.get("vwap_lo", vwap)
    vwap_up = row.get("vwap_up", vwap)
    band_w  = max(vwap_up - vwap_lo, 1e-6)
    deviation = (price - vwap) / (band_w / 2)

    bull_pts = 0.0
    bear_pts = 0.0

    # RSI extreme — tightened to 35/65 for higher precision, lower noise
    RSI_EXTREME_BULL = 35.0
    RSI_EXTREME_BEAR = 65.0
    if rsi < RSI_EXTREME_BULL:
        rsi_score = 1.0 + (RSI_EXTREME_BULL - rsi) / RSI_EXTREME_BULL * 0.5
        bull_pts += rsi_score
        if rsi > prev.get("rsi", rsi):   # RSI turning up
            bull_pts += 0.5
    elif rsi > RSI_EXTREME_BEAR:
        rsi_score = 1.0 + (rsi - RSI_EXTREME_BEAR) / (100 - RSI_EXTREME_BEAR) * 0.5
        bear_pts += rsi_score
        if rsi < prev.get("rsi", rsi):   # RSI turning down
            bear_pts += 0.5

    # VWAP deviation — requires at least 1.2σ from VWAP (tightened from 1.0)
    if deviation < -1.2:
        bull_pts += min(abs(deviation) * 0.4, 1.0)
    elif deviation > 1.2:
        bear_pts += min(abs(deviation) * 0.4, 1.0)

    # Candle body confirmation — must be a meaningful candle, not a doji
    body_size = abs(row["close"] - row["open"])
    atr       = max(float(row.get("atr", 1) or 1), 1.0)
    if body_size > atr * 0.15:   # candle body at least 15% of ATR
        if row["close"] > row["open"]:
            bull_pts += 0.4
        else:
            bear_pts += 0.4

    max_score = 3.4
    bull_conf = min(bull_pts / max_score, 1.0)
    bear_conf = min(bear_pts / max_score, 1.0)

    if bull_conf >= 0.50 and bull_conf > bear_conf:
        return _vote(VOTE_BUY,  bull_conf, f"Reversion BUY [RSI={rsi:.0f} dev={deviation:.2f}]")
    if bear_conf >= 0.50 and bear_conf > bull_conf:
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


# ─── Voter 6: HTF (15-minute Higher Timeframe Alignment) ─────────────────────

def htf_voter(df_15m: pd.DataFrame | None) -> dict:
    """
    Evaluates the 15-minute timeframe for trend direction.
    Requires 15-min OHLCV DataFrame (same columns as 5-min df).
    Returns ABSTAIN if df_15m is None or has insufficient data.

    Sub-checks:
    • EMA-9 vs EMA-21 alignment on 15-min
    • SuperTrend direction on 15-min
    • MACD histogram sign on 15-min

    Rationale: 5-min signals against a 15-min downtrend fail far more often.
    HTF agreement is the single biggest win-rate lever in multi-TF systems.
    """
    if df_15m is None or len(df_15m) < 30:
        return _vote(VOTE_ABSTAIN, 0.5, "HTF: no 15m data")

    try:
        # Compute fast/slow EMA on 15-min
        ema_f = df_15m["close"].ewm(span=HTF_EMA_FAST,  adjust=False).mean()
        ema_s = df_15m["close"].ewm(span=HTF_EMA_SLOW,  adjust=False).mean()

        # SuperTrend direction on 15-min
        hl2  = (df_15m["high"] + df_15m["low"]) / 2
        hl   = df_15m["high"] - df_15m["low"]
        hc   = (df_15m["high"] - df_15m["close"].shift()).abs()
        lc   = (df_15m["low"]  - df_15m["close"].shift()).abs()
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr  = tr.ewm(com=13, adjust=False).mean()
        upper = hl2 + HTF_SUPERTREND_MULT * atr
        lower = hl2 - HTF_SUPERTREND_MULT * atr

        st_dir = pd.Series(1, index=df_15m.index)
        st_val = pd.Series(np.nan, index=df_15m.index)
        for i in range(1, len(df_15m)):
            pv = st_val.iloc[i - 1]
            pd_ = st_dir.iloc[i - 1]
            c  = df_15m["close"].iloc[i]
            u  = upper.iloc[i]
            l  = lower.iloc[i]
            if not np.isnan(pv):
                l = max(l, pv) if pd_ == 1 else l
                u = min(u, pv) if pd_ == -1 else u
            if pd_ == 1:
                st_dir.iloc[i] = -1 if c < l else 1
                st_val.iloc[i] = u   if c < l else l
            else:
                st_dir.iloc[i] = 1  if c > u else -1
                st_val.iloc[i] = l  if c > u else u

        row  = df_15m.iloc[-1]
        macd_fast = df_15m["close"].ewm(span=12, adjust=False).mean()
        macd_slow = df_15m["close"].ewm(span=26, adjust=False).mean()
        macd_hist = (macd_fast - macd_slow).iloc[-1]

        bull_pts = 0.0
        bear_pts = 0.0

        if ema_f.iloc[-1] > ema_s.iloc[-1]:
            bull_pts += 1.0
        else:
            bear_pts += 1.0

        if st_dir.iloc[-1] == 1:
            bull_pts += 1.0
        else:
            bear_pts += 1.0

        if macd_hist > 0:
            bull_pts += 0.5
        else:
            bear_pts += 0.5

        total = bull_pts + bear_pts
        bull_conf = bull_pts / total if total > 0 else 0.5
        bear_conf = bear_pts / total if total > 0 else 0.5

        if bull_conf >= HTF_MIN_CONF:
            return _vote(VOTE_BUY,  bull_conf, f"HTF BUY [15m ema={'↑' if ema_f.iloc[-1]>ema_s.iloc[-1] else '↓'} st={st_dir.iloc[-1]} macd={'↑' if macd_hist>0 else '↓'}]")
        if bear_conf >= HTF_MIN_CONF:
            return _vote(VOTE_SELL, bear_conf, f"HTF SELL [15m ema={'↑' if ema_f.iloc[-1]>ema_s.iloc[-1] else '↓'} st={st_dir.iloc[-1]} macd={'↑' if macd_hist>0 else '↓'}]")

        return _vote(VOTE_ABSTAIN, 0.5, "HTF: mixed 15m signals")

    except Exception as e:
        logger.debug(f"HTF voter error: {e}")
        return _vote(VOTE_ABSTAIN, 0.5, f"HTF error: {e}")


# ─── Master: run all 6 voters ─────────────────────────────────────────────────

def run_all_voters(df: pd.DataFrame,
                   proposed_action: str,
                   kronos_filter=None,
                   ml_ensemble=None,
                   features: np.ndarray = None,
                   df_htf: pd.DataFrame = None) -> dict:
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
    v6 = htf_voter(df_htf)

    voters = [
        {"name": "Trend",     **v1},
        {"name": "Reversion", **v2},
        {"name": "Breakout",  **v3},
        {"name": "Kronos",    **v4},
        {"name": "ML",        **v5},
        {"name": "HTF",       **v6},
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
