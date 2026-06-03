"""
Layer 4: Consensus — 4/5 Voters Must Agree
============================================
Direct implementation of the PDF's biggest finding:
  5/5 unanimous = 67% WR vs 27% for 4/5

Each of 5 voters casts: BUY / SELL / ABSTAIN
If 4 or more vote the same direction → consensus reached.

Layer 6 (Position Sizing) is also handled here:
  4/5 agree → 20% of capital
  5/5 agree → 35% of capital
  (PDF Scenario B: 5/5 at 20% → +13% in 6 months)
"""

import logging
from config import (
    CONSENSUS_VOTES_REQUIRED, N_VOTERS,
    POSITION_SIZE_3_OF_5, POSITION_SIZE_4_OF_5, POSITION_SIZE_5_OF_5,
)

logger = logging.getLogger(__name__)


def run_consensus(voter_results: dict, proposed_action: str) -> dict:
    """
    Count votes for the proposed direction and return consensus decision.

    Args:
      voter_results : output of signal_layers.run_all_voters()
      proposed_action : "BUY" or "SELL" (from technical signal)

    Returns:
      approved       : bool — proceed with trade
      direction      : "BUY" | "SELL" | "HOLD"
      votes_for      : int — votes in winning direction
      votes_against  : int — votes opposing
      abstentions    : int
      position_pct   : float — % of capital to deploy (Layer 6)
      confidence     : float — avg confidence of agreeing voters
      unanimity      : bool — all 5 agree
      summary        : str — human-readable
    """
    voters = voter_results.get("voters", [])
    if not voters:
        return _no_consensus("No voter data")

    votes_for     = sum(1 for v in voters if v["vote"] == proposed_action)
    votes_against = sum(1 for v in voters
                       if v["vote"] != proposed_action and v["vote"] != "ABSTAIN")
    abstentions   = sum(1 for v in voters if v["vote"] == "ABSTAIN")

    # Log each voter result
    for v in voters:
        icon = "✅" if v["vote"] == proposed_action else ("❌" if v["vote"] != "ABSTAIN" else "⏸️")
        logger.info(
            f"  {icon} {v['name']:10s} → {v['vote']:7s} "
            f"conf={v['confidence']:.2f}  {v['reason']}"
        )

    unanimity    = (votes_for == N_VOTERS)
    approved     = (votes_for >= CONSENSUS_VOTES_REQUIRED)

    # Average confidence of agreeing voters
    agreeing_confs = [v["confidence"] for v in voters if v["vote"] == proposed_action]
    avg_confidence = float(sum(agreeing_confs) / len(agreeing_confs)) if agreeing_confs else 0.0

    # Layer 6: Position sizing based on agreement level
    if unanimity:
        position_pct = POSITION_SIZE_5_OF_5    # 35%
        sizing_note  = f"5/5 unanimous → {position_pct:.0f}% capital"
    elif votes_for >= 4:
        position_pct = POSITION_SIZE_4_OF_5    # 20%
        sizing_note  = f"4/5 agree → {position_pct:.0f}% capital"
    elif votes_for >= CONSENSUS_VOTES_REQUIRED:
        position_pct = POSITION_SIZE_3_OF_5    # 15% (conservative while Kronos offline)
        sizing_note  = f"3/5 agree → {position_pct:.0f}% capital (Kronos offline)"
    else:
        position_pct = 0.0
        sizing_note  = f"only {votes_for}/5 — no trade"

    summary = (
        f"{proposed_action} | votes={votes_for}/{N_VOTERS} "
        f"({'UNANIMOUS' if unanimity else 'APPROVED' if approved else 'REJECTED'}) "
        f"| avg_conf={avg_confidence:.2f} | {sizing_note}"
    )

    logger.info(f"[CONSENSUS] {summary}")

    return {
        "approved":     approved,
        "direction":    proposed_action if approved else "HOLD",
        "votes_for":    votes_for,
        "votes_against":votes_against,
        "abstentions":  abstentions,
        "position_pct": position_pct,
        "confidence":   avg_confidence,
        "unanimity":    unanimity,
        "summary":      summary,
        "voters":       voters,
    }


def _no_consensus(reason: str) -> dict:
    return {
        "approved":     False,
        "direction":    "HOLD",
        "votes_for":    0,
        "votes_against":0,
        "abstentions":  0,
        "position_pct": 0.0,
        "confidence":   0.0,
        "unanimity":    False,
        "summary":      f"No consensus: {reason}",
        "voters":       [],
    }
