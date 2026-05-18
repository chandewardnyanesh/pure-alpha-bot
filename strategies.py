"""
Directional strategies for options trading.
We generate a BUY (bullish → buy CE) or SELL (bearish → buy PE) signal
on the underlying index using technical indicators on its spot/futures candles.

All strategies score 0-1. Higher = more conviction.
"""

import pandas as pd
import numpy as np
from config import (
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    STRATEGY_WEIGHTS,
)


# ─── 1. EMA Crossover + RSI ───────────────────────────────────────────────────

def ema_crossover_strategy(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data"}

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
        return {"action": "BUY", "score": min(score, 0.95), "reason": "EMA bullish cross"}

    if bearish_cross and not above_trend and row["rsi"] > RSI_OVERSOLD:
        score = 0.50 + (0.25 if macd_bear else 0) + (0.15 if row.get("vol_ratio", 1) > 1.2 else 0)
        return {"action": "SELL", "score": min(score, 0.95), "reason": "EMA bearish cross"}

    return {"action": "HOLD", "score": 0.0, "reason": "no EMA cross"}


# ─── 2. VWAP Extreme Reversion ────────────────────────────────────────────────

def vwap_reversion_strategy(df: pd.DataFrame) -> dict:
    """
    For options, only take VWAP signals when the move is extreme (>2 std).
    Moderate VWAP bands = noise for options.
    """
    if len(df) < 5:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data"}

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    price = row["close"]
    vwap  = row.get("vwap", price)
    vwap_lo = row.get("vwap_lo", vwap)
    vwap_up = row.get("vwap_up", vwap)

    band_width = vwap_up - vwap_lo
    if band_width <= 0:
        return {"action": "HOLD", "score": 0.0, "reason": "VWAP bands zero"}

    deviation = (price - vwap) / (band_width / 2)   # -1 = lower band, +1 = upper band

    recovering = price > prev["close"]
    fading     = price < prev["close"]
    bull_candle = row["close"] > row["open"]
    bear_candle = row["close"] < row["open"]

    # Only act on extreme deviations (price beyond 1σ band)
    if deviation < -1.0 and row["rsi"] < RSI_OVERSOLD and bull_candle and recovering:
        score = min(0.40 + abs(deviation) * 0.1, 0.80)
        return {"action": "BUY", "score": score, "reason": "VWAP extreme low bounce"}

    if deviation > 1.0 and row["rsi"] > RSI_OVERBOUGHT and bear_candle and fading:
        score = min(0.40 + abs(deviation) * 0.1, 0.80)
        return {"action": "SELL", "score": score, "reason": "VWAP extreme high fade"}

    return {"action": "HOLD", "score": 0.0, "reason": "VWAP within normal range"}


# ─── 3. SuperTrend ────────────────────────────────────────────────────────────

def supertrend_strategy(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data"}

    row  = df.iloc[-1]
    prev = df.iloc[-2]

    flipped_bull = (prev["supertrend_dir"] == -1 and row["supertrend_dir"] == 1)
    flipped_bear = (prev["supertrend_dir"] ==  1 and row["supertrend_dir"] == -1)
    high_vol     = row.get("vol_ratio", 1.0) > 1.0
    rsi_ok_bull  = row["rsi"] < 70
    rsi_ok_bear  = row["rsi"] > 30

    if flipped_bull and rsi_ok_bull:
        score = 0.65 + (0.20 if high_vol else 0)
        return {"action": "BUY",  "score": min(score, 0.95), "reason": "SuperTrend bullish flip"}

    if flipped_bear and rsi_ok_bear:
        score = 0.65 + (0.20 if high_vol else 0)
        return {"action": "SELL", "score": min(score, 0.95), "reason": "SuperTrend bearish flip"}

    # Continuation — lower conviction
    if row["supertrend_dir"] == 1 and row["close"] > row["ema_fast"] and rsi_ok_bull:
        return {"action": "BUY",  "score": 0.42, "reason": "SuperTrend continuation (bull)"}

    if row["supertrend_dir"] == -1 and row["close"] < row["ema_fast"] and rsi_ok_bear:
        return {"action": "SELL", "score": 0.42, "reason": "SuperTrend continuation (bear)"}

    return {"action": "HOLD", "score": 0.0, "reason": "SuperTrend no signal"}


# ─── 4. Momentum Breakout ─────────────────────────────────────────────────────

def breakout_strategy(df: pd.DataFrame) -> dict:
    """
    Price breaks N-bar high/low with strong volume surge.
    This is the best options entry signal — momentum move triggers large premium moves.
    """
    lookback = 10
    if len(df) < lookback + 2:
        return {"action": "HOLD", "score": 0.0, "reason": "insufficient data"}

    window    = df.iloc[-(lookback + 1):-1]
    row       = df.iloc[-1]
    price     = row["close"]
    vol_surge = row.get("vol_ratio", 1.0) >= 1.5   # 1.5× avg volume

    pivot_high = window["high"].max()
    pivot_low  = window["low"].min()
    macd_bull  = row["macd_hist"] > 0
    macd_bear  = row["macd_hist"] < 0

    # Stronger bonus for bigger volume surges
    vol_bonus = min(row.get("vol_ratio", 1.0) - 1.5, 0.5) * 0.2 if vol_surge else 0

    if price > pivot_high and vol_surge and row["rsi"] < 75:
        score = 0.60 + (0.20 if macd_bull else 0) + vol_bonus
        return {"action": "BUY",  "score": min(score, 0.95), "reason": f"Breakout above {pivot_high:.0f}"}

    if price < pivot_low and vol_surge and row["rsi"] > 25:
        score = 0.60 + (0.20 if macd_bear else 0) + vol_bonus
        return {"action": "SELL", "score": min(score, 0.95), "reason": f"Breakdown below {pivot_low:.0f}"}

    return {"action": "HOLD", "score": 0.0, "reason": "no breakout"}


# ─── 5. Opening Range Breakout (ORB) ─────────────────────────────────────────

def orb_strategy(df: pd.DataFrame) -> dict:
    """
    First 15 minutes = Opening Range. Trade breakout of that range.
    Very reliable for Nifty/BankNifty option buys.
    Only fires between 09:30 and 10:30.
    """
    from datetime import datetime
    now_h = datetime.now().hour
    now_m = datetime.now().minute

    # Only active between 09:30 and 10:30
    if not (9 * 60 + 30 <= now_h * 60 + now_m <= 10 * 60 + 30):
        return {"action": "HOLD", "score": 0.0, "reason": "ORB: outside time window"}

    if len(df) < 4:
        return {"action": "HOLD", "score": 0.0, "reason": "ORB: insufficient data"}

    # Opening range = first 3 candles (15 min if 5-min candles)
    opening_range = df.iloc[:3]
    orb_high = opening_range["high"].max()
    orb_low  = opening_range["low"].min()
    row      = df.iloc[-1]
    price    = row["close"]
    high_vol = row.get("vol_ratio", 1.0) > 1.3

    if price > orb_high and high_vol:
        return {"action": "BUY",  "score": 0.75, "reason": f"ORB breakout above {orb_high:.0f}"}

    if price < orb_low and high_vol:
        return {"action": "SELL", "score": 0.75, "reason": f"ORB breakdown below {orb_low:.0f}"}

    return {"action": "HOLD", "score": 0.0, "reason": "ORB: price within range"}


# ─── Ensemble ─────────────────────────────────────────────────────────────────

STRATEGY_FNS = {
    "ema_crossover":  ema_crossover_strategy,
    "vwap_reversion": vwap_reversion_strategy,
    "supertrend":     supertrend_strategy,
    "breakout":       breakout_strategy,
    "orb":            orb_strategy,
}

# ORB gets its own weight (not in config to keep config clean)
_ORB_WEIGHT = 0.15


def blend_signals(df: pd.DataFrame, weights: dict = None) -> dict:
    if weights is None:
        weights = STRATEGY_WEIGHTS

    # Use weights directly (un-normalised) so a single strong strategy
    # (e.g. SuperTrend flip weight=0.45, score=0.85 → 0.38) can clear the
    # SIGNAL_THRESHOLD on its own; weak signals still stay below it.
    buy_score  = 0.0
    sell_score = 0.0
    reasons    = []

    for name, fn in STRATEGY_FNS.items():
        w      = _ORB_WEIGHT if name == "orb" else weights.get(name, 0.0)
        result = fn(df)

        if result["action"] == "BUY":
            buy_score  += w * result["score"]
            reasons.append(f"{name}:BUY({result['score']:.2f})")
        elif result["action"] == "SELL":
            sell_score += w * result["score"]
            reasons.append(f"{name}:SELL({result['score']:.2f})")

    if buy_score > sell_score:
        return {"action": "BUY",  "score": buy_score,  "reasons": reasons}
    elif sell_score > buy_score:
        return {"action": "SELL", "score": sell_score, "reasons": reasons}
    else:
        return {"action": "HOLD", "score": 0.0,        "reasons": reasons}
