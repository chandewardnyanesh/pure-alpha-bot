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
from datetime import datetime
from config import (
    ATR_TREND_FILTER, ATR_TREND_MIN_RATIO,
    ADX_TREND_THRESHOLD, VOLUME_FILTER_RATIO,
    USE_SESSION_DEADZONE, SESSION_DEADZONE_START, SESSION_DEADZONE_END,
    USE_VIX_FILTER, VIX_MAX_ENTRY, VIX_HIGH_THRESHOLD, VIX_LOW_THRESHOLD,
    VIX_SIZE_REDUCTION_HIGH, VIX_SIZE_REDUCTION_LOW,
)

logger = logging.getLogger(__name__)


_current_vix: float = 0.0   # module-level cache, updated by bot each scan


def update_vix(vix_value: float):
    """Called by bot after fetching India VIX LTP each scan cycle."""
    global _current_vix
    _current_vix = vix_value


def get_vix_size_multiplier() -> float:
    """Return position-size multiplier based on current VIX regime."""
    if _current_vix <= 0:
        return 1.0   # VIX unavailable — no adjustment
    if _current_vix > VIX_HIGH_THRESHOLD:
        return VIX_SIZE_REDUCTION_HIGH
    if _current_vix < VIX_LOW_THRESHOLD:
        return VIX_SIZE_REDUCTION_LOW
    return 1.0


def check_session_filter() -> tuple[bool, str]:
    """Block entries during the 11:30–13:00 midday dead zone."""
    if not USE_SESSION_DEADZONE:
        return True, "session_filter_disabled"
    now = datetime.now().strftime("%H:%M")
    if SESSION_DEADZONE_START <= now <= SESSION_DEADZONE_END:
        return False, f"session_deadzone ({SESSION_DEADZONE_START}–{SESSION_DEADZONE_END})"
    return True, f"session_ok ({now})"


def check_vix_filter() -> tuple[bool, str]:
    """Block entries when India VIX is in extreme fear territory."""
    if not USE_VIX_FILTER or _current_vix <= 0:
        return True, f"vix_unavail"
    if _current_vix > VIX_MAX_ENTRY:
        return False, f"VIX={_current_vix:.1f}>{VIX_MAX_ENTRY} (extreme_fear)"
    return True, f"VIX={_current_vix:.1f}✓"


def check_market_filter(row) -> tuple[bool, str]:
    """
    Run all Layer 5 checks on the current bar.

    Returns:
      (allowed: bool, reason: str)
    """
    # ── Session dead zone ────────────────────────────────────────────────────
    sess_ok, sess_reason = check_session_filter()
    if not sess_ok:
        logger.info(f"[FILTER] BLOCKED: {sess_reason}")
        return False, f"BLOCKED: {sess_reason}"

    # ── VIX extreme fear ─────────────────────────────────────────────────────
    vix_ok, vix_reason = check_vix_filter()
    if not vix_ok:
        logger.info(f"[FILTER] BLOCKED: {vix_reason}")
        return False, f"BLOCKED: {vix_reason}"

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
    adx = float(row.get("adx", 25.0) or 25.0)
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

    reason = "PASS: " + ", ".join(reasons_pass) + f" | {vix_reason}"
    return True, reason


def filter_status(row) -> dict:
    """Return Layer 5 status dict for dashboard display."""
    atr_ratio = float(row.get("atr_ratio", 1.0) or 1.0)
    adx       = float(row.get("adx", 0.0) or 0.0)
    vol_ratio = float(row.get("vol_ratio", 1.0) or 1.0)
    allowed, reason = check_market_filter(row)
    return {
        "allowed":    allowed,
        "reason":     reason,
        "atr_ratio":  round(atr_ratio, 3),
        "adx":        round(adx, 1),
        "vol_ratio":  round(vol_ratio, 3),
        "vix":        round(_current_vix, 2),
        "vix_mult":   get_vix_size_multiplier(),
        "session_ok": check_session_filter()[0],
    }
