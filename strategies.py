"""
Directional strategies for options trading.
We generate a BUY (bullish → buy CE) or SELL (bearish → buy PE) signal
on the underlying index using technical indicators on its spot/futures candles.

All strategies score 0-1. Higher = more conviction.

Lesson from PDF model (Trading_Model_Documented.pdf):
  • Agreement count matters MORE than individual score thresholds.
  • 5/5 unanimous = 67% WR vs 27% for 4/5 (their biggest finding).
  • We replicate this: blend_signals() returns agreement_count alongside score.
  • FVG (Fair Value Gap) added as 5th strategy — mirrors PDF's 5-strategy setup.
"""

import pandas as pd
import numpy as np
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    STRATEGY_WEIGHTS, FVG_PROXIMITY_ATR,
)


# ─── 1. EMA Crossover + RSI ───────────────────────────────────────────────────

def ema_crossover_strategy(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data", "is_flip": False}

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_cross = (prev["ema_fast"] <= prev["ema_slow"] and
                     row["ema_fast"]   >  row["ema_slow"])
    bearish_cross = (prev["ema_fast"] >= prev["ema_slow"] and
                     row["ema_fast"]   <  row["ema_slow"])

    above_trend = row["close"] > row["ema_trend"]
    macd_bull   = row["macd_hist"] > 0
    macd_bear   = row["macd_hist"] < 0

    if bullish_cross and above_trend and row["rsi"] < RSI_OVERBOUGHT:
        score = 0.50 + (0.25 if macd_bull else 0) + (0.15 if row.get("vol_ratio", 1) > 1.2 else 0)
        return {"action": "BUY", "score": min(score, 0.95), "reason": "EMA bullish cross", "is_flip": True}

    if bearish_cross and not above_trend and row["rsi"] > RSI_OVERSOLD:
        score = 0.50 + (0.25 if macd_bear else 0) + (0.15 if row.get("vol_ratio", 1) > 1.2 else 0)
        return {"action": "SELL", "score": min(score, 0.95), "reason": "EMA bearish cross", "is_flip": True}

    return {"action": "HOLD", "score": 0.0, "reason": "no EMA cross", "is_flip": False}


# ─── 2. VWAP Extreme Reversion ────────────────────────────────────────────────

def vwap_reversion_strategy(df: pd.DataFrame) -> dict:
    """Only act on extreme VWAP deviations (>2 std) — moderate = noise."""
    if len(df) < 5:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data", "is_flip": False}

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    price    = row["close"]
    vwap     = row.get("vwap", price)
    vwap_lo  = row.get("vwap_lo", vwap)
    vwap_up  = row.get("vwap_up", vwap)

    band_width = vwap_up - vwap_lo
    if band_width <= 0:
        return {"action": "HOLD", "score": 0.0, "reason": "VWAP bands zero", "is_flip": False}

    deviation  = (price - vwap) / (band_width / 2)
    recovering = price > prev["close"]
    fading     = price < prev["close"]
    bull_candle = row["close"] > row["open"]
    bear_candle = row["close"] < row["open"]

    if deviation < -1.0 and row["rsi"] < RSI_OVERSOLD and bull_candle and recovering:
        score = min(0.40 + abs(deviation) * 0.1, 0.80)
        return {"action": "BUY", "score": score, "reason": "VWAP extreme low bounce", "is_flip": True}

    if deviation > 1.0 and row["rsi"] > RSI_OVERBOUGHT and bear_candle and fading:
        score = min(0.40 + abs(deviation) * 0.1, 0.80)
        return {"action": "SELL", "score": score, "reason": "VWAP extreme high fade", "is_flip": True}

    return {"action": "HOLD", "score": 0.0, "reason": "VWAP within normal range", "is_flip": False}


# ─── 3. SuperTrend ────────────────────────────────────────────────────────────

def supertrend_strategy(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data", "is_flip": False}

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    flipped_bull = (prev["supertrend_dir"] == -1 and row["supertrend_dir"] == 1)
    flipped_bear = (prev["supertrend_dir"] ==  1 and row["supertrend_dir"] == -1)
    high_vol     = row.get("vol_ratio", 1.0) > 1.0
    rsi_ok_bull  = row["rsi"] < 70
    rsi_ok_bear  = row["rsi"] > 30

    if flipped_bull and rsi_ok_bull:
        score = 0.65 + (0.20 if high_vol else 0)
        return {"action": "BUY",  "score": min(score, 0.95), "reason": "SuperTrend bullish flip", "is_flip": True}

    if flipped_bear and rsi_ok_bear:
        score = 0.65 + (0.20 if high_vol else 0)
        return {"action": "SELL", "score": min(score, 0.95), "reason": "SuperTrend bearish flip", "is_flip": True}

    # Continuation — only when RSI is not extended AND volume is elevated
    # Lesson from our trades: continuation entries at 11:51 on quiet RSI = loss
    cont_bull = (row["supertrend_dir"] == 1 and row["close"] > row["ema_fast"]
                 and rsi_ok_bull and row["rsi"] < 58 and high_vol)
    cont_bear = (row["supertrend_dir"] == -1 and row["close"] < row["ema_fast"]
                 and rsi_ok_bear and row["rsi"] > 42 and high_vol)

    if cont_bull:
        return {"action": "BUY",  "score": 0.42, "reason": "SuperTrend continuation (bull)", "is_flip": False}
    if cont_bear:
        return {"action": "SELL", "score": 0.42, "reason": "SuperTrend continuation (bear)", "is_flip": False}

    return {"action": "HOLD", "score": 0.0, "reason": "SuperTrend no signal", "is_flip": False}


# ─── 4. Momentum Breakout ─────────────────────────────────────────────────────

def breakout_strategy(df: pd.DataFrame) -> dict:
    """Price breaks N-bar high/low with strong volume surge."""
    lookback = 10
    if len(df) < lookback + 2:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data", "is_flip": False}

    window    = df.iloc[-(lookback + 1):-1]
    row       = df.iloc[-1]
    price     = row["close"]
    vol_surge = row.get("vol_ratio", 1.0) >= 1.5

    pivot_high = window["high"].max()
    pivot_low  = window["low"].min()
    macd_bull  = row["macd_hist"] > 0
    macd_bear  = row["macd_hist"] < 0

    vol_bonus = min(row.get("vol_ratio", 1.0) - 1.5, 0.5) * 0.2 if vol_surge else 0

    if price > pivot_high and vol_surge and row["rsi"] < 75:
        score = 0.60 + (0.20 if macd_bull else 0) + vol_bonus
        return {"action": "BUY",  "score": min(score, 0.95), "reason": f"Breakout above {pivot_high:.0f}", "is_flip": True}

    if price < pivot_low and vol_surge and row["rsi"] > 25:
        score = 0.60 + (0.20 if macd_bear else 0) + vol_bonus
        return {"action": "SELL", "score": min(score, 0.95), "reason": f"Breakdown below {pivot_low:.0f}", "is_flip": True}

    return {"action": "HOLD", "score": 0.0, "reason": "no breakout", "is_flip": False}


# ─── 5. Opening Range Breakout (ORB) ─────────────────────────────────────────

def orb_strategy(df: pd.DataFrame) -> dict:
    """First 15 minutes = Opening Range. Trade breakout of that range (09:30–10:30)."""
    from datetime import datetime
    now_h = datetime.now().hour
    now_m = datetime.now().minute

    if not (9 * 60 + 30 <= now_h * 60 + now_m <= 10 * 60 + 30):
        return {"action": "HOLD", "score": 0.0, "reason": "ORB: outside time window", "is_flip": False}

    if len(df) < 4:
        return {"action": "HOLD", "score": 0.0, "reason": "ORB: insufficient data", "is_flip": False}

    opening_range = df.iloc[:3]
    orb_high = opening_range["high"].max()
    orb_low  = opening_range["low"].min()
    row      = df.iloc[-1]
    price    = row["close"]
    high_vol = row.get("vol_ratio", 1.0) > 1.3

    if price > orb_high and high_vol:
        return {"action": "BUY",  "score": 0.75, "reason": f"ORB breakout above {orb_high:.0f}", "is_flip": True}

    if price < orb_low and high_vol:
        return {"action": "SELL", "score": 0.75, "reason": f"ORB breakdown below {orb_low:.0f}", "is_flip": True}

    return {"action": "HOLD", "score": 0.0, "reason": "ORB: price within range", "is_flip": False}


# ─── 6. Fair Value Gap (SMC) ──────────────────────────────────────────────────

def fvg_strategy(df: pd.DataFrame) -> dict:
    """
    Fair Value Gap — Smart Money Concepts imbalance pattern.

    Directly ported from PDF Section 3.4 lessons:
    • Uses close-based fill detection (not wick-based) → prevents false fills
    • Requires fresh gap (< FVG_FRESH_BARS) and price near the gap edge
    • HTF filter: EMA trend must agree with FVG direction

    Scoring (0–100 points → normalised to 0–1):
      Proximity  (35): price within 0.5 ATR of gap edge
      Freshness  (25): gap formed within FVG_FRESH_BARS
      HTF filter (25): EMA trend agrees with gap direction
      Candle     (15): candle body confirms direction
    """
    if len(df) < 5:
        return {"action": "HOLD", "score": 0.0, "reason": "FVG: insufficient data", "is_flip": False}

    row = df.iloc[-1]
    atr = max(float(row.get("atr", 1) or 1), 1.0)

    # ── Bullish FVG ──────────────────────────────────────────────────────────
    bull_fresh = float(row.get("fvg_bull_fresh", 999))
    bull_top   = float(row.get("fvg_bull_top", 0))
    bull_bot   = float(row.get("fvg_bull_bot", 0))

    if bull_fresh < 999 and bull_top > 0:
        price = row["close"]
        # Price returning into the bullish FVG from above = support bounce
        proximity_to_gap = min(abs(price - bull_bot), abs(price - bull_top))
        proximity_score  = max(0, 35 * (1 - proximity_to_gap / (FVG_PROXIMITY_ATR * atr)))
        freshness_score  = 25 if bull_fresh <= 15 else max(0, 25 * (1 - (bull_fresh - 15) / 15))
        htf_score        = 25 if row.get("ema_trend", 0) > 0 and row["close"] > row.get("ema_trend", 0) else 0
        candle_score     = 15 if row["close"] > row["open"] else 0

        total = proximity_score + freshness_score + htf_score + candle_score
        if total >= 40 and row["rsi"] < RSI_OVERBOUGHT:
            score = min(total / 100, 0.90)
            return {"action": "BUY", "score": score,
                    "reason": f"FVG bullish gap [{bull_bot:.0f}–{bull_top:.0f}] age={bull_fresh:.0f}b",
                    "is_flip": True}

    # ── Bearish FVG ──────────────────────────────────────────────────────────
    bear_fresh = float(row.get("fvg_bear_fresh", 999))
    bear_top   = float(row.get("fvg_bear_top", 0))
    bear_bot   = float(row.get("fvg_bear_bot", 0))

    if bear_fresh < 999 and bear_top > 0:
        price = row["close"]
        proximity_to_gap = min(abs(price - bear_bot), abs(price - bear_top))
        proximity_score  = max(0, 35 * (1 - proximity_to_gap / (FVG_PROXIMITY_ATR * atr)))
        freshness_score  = 25 if bear_fresh <= 15 else max(0, 25 * (1 - (bear_fresh - 15) / 15))
        htf_score        = 25 if row.get("ema_trend", 0) > 0 and row["close"] < row.get("ema_trend", 0) else 0
        candle_score     = 15 if row["close"] < row["open"] else 0

        total = proximity_score + freshness_score + htf_score + candle_score
        if total >= 40 and row["rsi"] > RSI_OVERSOLD:
            score = min(total / 100, 0.90)
            return {"action": "SELL", "score": score,
                    "reason": f"FVG bearish gap [{bear_bot:.0f}–{bear_top:.0f}] age={bear_fresh:.0f}b",
                    "is_flip": True}

    return {"action": "HOLD", "score": 0.0, "reason": "FVG: no fresh gap near price", "is_flip": False}


# ─── Ensemble ─────────────────────────────────────────────────────────────────

STRATEGY_FNS = {
    "ema_crossover":  ema_crossover_strategy,
    "vwap_reversion": vwap_reversion_strategy,
    "supertrend":     supertrend_strategy,
    "breakout":       breakout_strategy,
    "orb":            orb_strategy,
    "fvg":            fvg_strategy,        # NEW: 6th strategy
}

_ORB_WEIGHT = 0.15


def blend_signals(df: pd.DataFrame, weights: dict = None) -> dict:
    """
    Ensemble blender.

    Returns:
      action          : BUY / SELL / HOLD
      score           : weighted blended score
      reasons         : list of contributing strategy strings
      has_flip        : True if ≥1 contributing strategy fired a flip signal
      agreement_count : number of strategies that fired in the winning direction
                        (directly mirrors PDF's N/5 counting — key metric)
    """
    if weights is None:
        weights = STRATEGY_WEIGHTS

    buy_score       = 0.0
    sell_score      = 0.0
    reasons         = []
    has_flip        = False
    buy_agreement   = 0    # # strategies that fired BUY
    sell_agreement  = 0    # # strategies that fired SELL

    for name, fn in STRATEGY_FNS.items():
        w      = _ORB_WEIGHT if name == "orb" else weights.get(name, 0.0)
        result = fn(df)

        if result["action"] == "BUY":
            buy_score      += w * result["score"]
            buy_agreement  += 1
            reasons.append(f"{name}:BUY({result['score']:.2f})")
            if result.get("is_flip", False):
                has_flip = True

        elif result["action"] == "SELL":
            sell_score     += w * result["score"]
            sell_agreement += 1
            reasons.append(f"{name}:SELL({result['score']:.2f})")
            if result.get("is_flip", False):
                has_flip = True

    if buy_score > sell_score:
        return {
            "action":          "BUY",
            "score":           buy_score,
            "reasons":         reasons,
            "has_flip":        has_flip,
            "agreement_count": buy_agreement,
        }
    elif sell_score > buy_score:
        return {
            "action":          "SELL",
            "score":           sell_score,
            "reasons":         reasons,
            "has_flip":        has_flip,
            "agreement_count": sell_agreement,
        }
    else:
        return {
            "action":          "HOLD",
            "score":           0.0,
            "reasons":         reasons,
            "has_flip":        False,
            "agreement_count": 0,
        }
