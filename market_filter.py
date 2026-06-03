"""
Layer 5: Market Filter — ATR + ADX + Volume
============================================
Only allows entries when the market is in a trending, liquid state.

PDF lesson: strategies designed for trending markets produce 0% WR on noisy
timeframes. We proxy this with:
  • ATR ratio: current ATR vs 20-bar avg — measures volatility expansion
  • ADX: Average Directional Index — measures trend strength (not direction)
  • Volume: above-average volume confirms institutional participation

All three must pass for the filter to approve entry.
"""

import logging
from config import (
    ATR_TREND_FILTER, ATR_TREND_MIN_RATIO,
    ADX_TREND_THRESHOLD, VOLUME_FILTER_RATIO,
)

logger = logging.getLogger(__name__)


def check_market_filter(row) -> tuple[bool, str]:
    """
    Run all Layer 5 checks on the current bar.

    Returns:
      (allowed: bool, reason: str)
    """
    if not ATR_TREND_FILTER:
        return True, "filter_disabled"

    reasons_fail = []
    reasons_pass = []

    # ── ATR ratio: is volatility expanding? ──────────────────────────────────
    atr_ratio = float(row.get("atr_ratio", 1.0) or 1.0)
    if atr_ratio < ATR_TREND_MIN_RATIO:
        reasons_fail.append(f"ATR_ratio={atr_ratio:.2f}<{ATR_TREND_MIN_RATIO} (ranging)")
    else:
        reasons_pass.append(f"ATR_ratio={atr_ratio:.2f}✓")

    # ── ADX: is the market trending? ─────────────────────────────────────────
    adx = float(row.get("adx", 25.0) or 25.0)   # default to pass if no ADX computed
    if adx < ADX_TREND_THRESHOLD:
        reasons_fail.append(f"ADX={adx:.1f}<{ADX_TREND_THRESHOLD} (choppy)")
    else:
        reasons_pass.append(f"ADX={adx:.1f}✓")

    # ── Volume: above average? ────────────────────────────────────────────────
    vol_ratio = float(row.get("vol_ratio", 1.0) or 1.0)
    if vol_ratio < VOLUME_FILTER_RATIO:
        reasons_fail.append(f"vol_ratio={vol_ratio:.2f}<{VOLUME_FILTER_RATIO} (thin)")
    else:
        reasons_pass.append(f"vol_ratio={vol_ratio:.2f}✓")

    if reasons_fail:
        reason = "BLOCKED: " + ", ".join(reasons_fail)
        logger.info(f"[FILTER] {reason}")
        return False, reason

    reason = "PASS: " + ", ".join(reasons_pass)
    return True, reason


def filter_status(row) -> dict:
    """Return Layer 5 status dict for dashboard display."""
    atr_ratio = float(row.get("atr_ratio", 1.0) or 1.0)
    adx       = float(row.get("adx", 0.0) or 0.0)
    vol_ratio = float(row.get("vol_ratio", 1.0) or 1.0)
    allowed, reason = check_market_filter(row)
    return {
        "allowed":   allowed,
        "reason":    reason,
        "atr_ratio": round(atr_ratio, 3),
        "adx":       round(adx, 1),
        "vol_ratio": round(vol_ratio, 3),
    }
