"""
Kronos Deep-Learning Pre-Trade Filter
======================================
Uses NeoQuasar/Kronos to forecast the next N candles on the index and
computes a direction-alignment score vs the proposed trade signal.

Score returned:
  +1.0  → Kronos strongly confirms the signal direction
   0.5  → Kronos neutral / uncertain
   0.0  → Kronos strongly contradicts the signal direction

The bot uses this score as an additional gate — it doesn't override
the technical signal, but it reduces (or boosts) the final blended score.

Usage:
  from kronos_filter import KronosFilter
  kf = KronosFilter()
  score = kf.get_alignment_score(df, signal_action="BUY")

Integration note:
  Model is loaded once and cached as a class-level singleton to avoid
  repeated HuggingFace downloads (same pattern as the MCP server).
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

# ─── Kronos repo path (mirrors KRONOS_REPO_PATH env var used by MCP server) ──
_KRONOS_PATHS = [
    os.environ.get("KRONOS_REPO_PATH", ""),
    os.path.expanduser("~/Kronos"),
    os.path.expanduser("~/kronos_repo"),
]


def _find_kronos_repo() -> str | None:
    for p in _KRONOS_PATHS:
        if p and os.path.isfile(os.path.join(p, "model", "kronos.py")):
            return p
    return None


class KronosFilter:
    """Singleton wrapper around KronosPredictor for pre-trade direction check."""

    _predictor  = None   # cached across instances
    _available  = None   # None = not tried yet; False = failed; True = ready

    def __init__(self,
                 model_name: str  = "NeoQuasar/Kronos-mini",
                 tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
                 max_context: int = 512,
                 pred_len: int    = 5):
        self.model_name     = model_name
        self.tokenizer_name = tokenizer_name
        self.max_context    = max_context
        self.pred_len       = pred_len
        self._init_model()

    # ─── Lazy initialisation ──────────────────────────────────────────────────

    def _init_model(self):
        if KronosFilter._available is not None:
            return

        repo = _find_kronos_repo()
        if repo is None:
            logger.warning("Kronos repo not found — pre-trade filter disabled. "
                           "Set KRONOS_REPO_PATH env var or clone to ~/Kronos")
            KronosFilter._available = False
            return

        try:
            if repo not in sys.path:
                sys.path.insert(0, repo)

            from model import Kronos, KronosTokenizer, KronosPredictor

            logger.info(f"Loading Kronos model {self.model_name} ...")
            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            model     = Kronos.from_pretrained(self.model_name)
            KronosFilter._predictor = KronosPredictor(
                model, tokenizer, max_context=self.max_context
            )
            KronosFilter._available = True
            logger.info("Kronos pre-trade filter READY ✅")

        except Exception as e:
            logger.warning(f"Kronos init failed ({e}) — pre-trade filter disabled")
            KronosFilter._available = False

    # ─── Core forecast ────────────────────────────────────────────────────────

    def get_alignment_score(self, df: pd.DataFrame, signal_action: str,
                            lookback: int = 100) -> float:
        """
        Forecast next `pred_len` candles and compute alignment with signal.

        Returns:
          float [0.0, 1.0] — proportion of forecast candles where
          close > open  (for BUY) or close < open (for SELL).
          0.5 = neutral.  >0.6 = confirms.  <0.4 = contradicts.
        """
        if not KronosFilter._available or KronosFilter._predictor is None:
            return 0.5   # neutral fallback — no filter applied

        try:
            # Prepare context window
            ctx = df.tail(lookback).copy()
            if len(ctx) < 20:
                return 0.5

            # Kronos expects: open, high, low, close, volume, amount
            if "amount" not in ctx.columns:
                ctx["amount"] = ctx["close"] * ctx["volume"]

            x_df = ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)

            # Build timestamps — use the df index if it has a DatetimeIndex,
            # otherwise synthesise 5-minute timestamps
            if isinstance(ctx.index, pd.DatetimeIndex):
                x_ts = ctx.index.to_series().reset_index(drop=True)
                last_ts = ctx.index[-1]
            else:
                last_ts = pd.Timestamp.now().replace(second=0, microsecond=0)
                x_ts = pd.Series([
                    last_ts - timedelta(minutes=5 * (len(ctx) - i))
                    for i in range(len(ctx))
                ])

            y_ts = pd.Series([
                last_ts + timedelta(minutes=5 * (i + 1))
                for i in range(self.pred_len)
            ])

            # Run Kronos forecast
            pred_df = KronosFilter._predictor.predict(
                df          = x_df,
                x_timestamp = x_ts,
                y_timestamp = y_ts,
                pred_len    = self.pred_len,
                T           = 1.0,
                top_p       = 0.9,
                sample_count= 1,
                verbose     = False,
            )

            if pred_df is None or pred_df.empty:
                return 0.5

            # Compute direction alignment
            bullish_bars = (pred_df["close"] > pred_df["open"]).sum()
            alignment    = bullish_bars / len(pred_df)   # 0..1, >0.5 = up trend

            if signal_action == "BUY":
                score = alignment        # high alignment = confirms BUY
            else:
                score = 1.0 - alignment  # low alignment (bearish forecast) = confirms SELL

            logger.info(
                f"Kronos: signal={signal_action} | "
                f"forecast {self.pred_len} bars: "
                f"{bullish_bars} bull / {len(pred_df)-bullish_bars} bear → "
                f"alignment={score:.2f}"
            )
            return float(score)

        except Exception as e:
            logger.warning(f"Kronos forecast error: {e}")
            return 0.5   # neutral on error

    @property
    def is_available(self) -> bool:
        return bool(KronosFilter._available)

    @classmethod
    def reset(cls):
        """Force re-initialisation (useful after model update)."""
        cls._predictor = None
        cls._available = None
