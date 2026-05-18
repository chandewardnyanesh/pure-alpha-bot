"""
Self-learning ML layer.

Flow:
  1. On startup: load saved model if it exists, else start with heuristic weights.
  2. After each closed trade: add (features_at_entry, profitable?) to dataset.
  3. Every MODEL_RETRAIN_EVERY trades (once MIN_TRAINING_SAMPLES reached):
     retrain RandomForest, update strategy weights from feature importances.
  4. The model predicts trade success probability → used to scale signal scores.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    MIN_TRAINING_SAMPLES, MODEL_RETRAIN_EVERY,
    MODEL_PATH, STRATEGY_WEIGHTS,
    FEATURE_IMPORTANCE_THRESHOLD,
)
from indicators import FEATURE_COLS, extract_features

logger = logging.getLogger(__name__)


class TradingModel:
    def __init__(self):
        self.model          = None
        self.strategy_weights = dict(STRATEGY_WEIGHTS)   # mutable copy
        self.training_data  = []    # list of (features, label)
        self.closed_since_retrain = 0
        self.total_trades   = 0
        self.win_rate       = 0.5   # running estimate
        self._load()

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                self.model             = saved.get("model")
                self.strategy_weights  = saved.get("weights", dict(STRATEGY_WEIGHTS))
                self.training_data     = saved.get("data", [])
                self.win_rate          = saved.get("win_rate", 0.5)
                logger.info(f"Model loaded — {len(self.training_data)} training samples, win_rate={self.win_rate:.2%}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":    self.model,
                "weights":  self.strategy_weights,
                "data":     self.training_data,
                "win_rate": self.win_rate,
            }, f)
        logger.info("Model saved.")

    # ─── Online learning ──────────────────────────────────────────────────────

    def record_trade_result(self, features: np.ndarray, profitable: bool):
        """Call this after every closed trade."""
        self.training_data.append((features, int(profitable)))
        self.total_trades += 1
        self.closed_since_retrain += 1

        wins = sum(label for _, label in self.training_data)
        self.win_rate = wins / len(self.training_data)

        logger.info(f"Trade recorded — profitable={profitable}, "
                    f"win_rate={self.win_rate:.2%}, total={self.total_trades}")

        if (len(self.training_data) >= MIN_TRAINING_SAMPLES and
                self.closed_since_retrain >= MODEL_RETRAIN_EVERY):
            self._retrain()
            self._save()

    def _retrain(self):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        X = np.array([f for f, _ in self.training_data], dtype=np.float32)
        y = np.array([l for _, l in self.training_data], dtype=np.int8)

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Try RF and GB; pick the one with better CV accuracy
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42
        )

        cv = min(5, len(y))  # don't fold more than we have
        try:
            rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="accuracy").mean()
            gb_cv = cross_val_score(gb, X, y, cv=cv, scoring="accuracy").mean()
        except Exception:
            rf_cv, gb_cv = 0.5, 0.4

        best = rf if rf_cv >= gb_cv else gb
        best.fit(X, y)
        self.model = best
        self.closed_since_retrain = 0

        logger.info(f"Model retrained ({type(best).__name__}), "
                    f"RF_cv={rf_cv:.3f}, GB_cv={gb_cv:.3f}")

        # ── Update strategy weights from feature importances ──────────────────
        if hasattr(best, "feature_importances_"):
            self._update_strategy_weights(best.feature_importances_)

    def _update_strategy_weights(self, importances: np.ndarray):
        """
        Map feature importances back to strategy weights.
        Features are in the same order as FEATURE_COLS + 3 engineered.
        We use domain knowledge to assign each feature to a strategy.
        """
        n = len(FEATURE_COLS) + 3       # total features returned by extract_features
        imp = importances[:n] if len(importances) >= n else importances

        # Strategy-to-feature affinity (indices in extract_features output)
        # rsi(0), macd_hist(1), bb_pct(2), ema_cross(3), supertrend_dir(4),
        # vol_ratio(5), momentum(6), atr(7),
        # vwap_dist(8), close_vs_ema_fast(9), ema_fast_vs_slow(10)
        affinity = {
            "ema_crossover":  [3, 9, 10],
            "vwap_reversion": [0, 2, 8],
            "supertrend":     [4, 7],
            "breakout":       [5, 6, 1],
        }

        raw = {}
        for strat, idxs in affinity.items():
            valid = [i for i in idxs if i < len(imp)]
            raw[strat] = float(imp[valid].sum()) if valid else 0.0

        total = sum(raw.values()) or 1.0
        for strat in raw:
            raw[strat] /= total

        # Blend 70% learned + 30% original (avoid catastrophic forgetting)
        orig = STRATEGY_WEIGHTS
        for strat in self.strategy_weights:
            self.strategy_weights[strat] = (
                0.7 * raw.get(strat, 0.0) + 0.3 * orig.get(strat, 0.25)
            )

        logger.info(f"Strategy weights updated: {self.strategy_weights}")

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict_success_proba(self, features: np.ndarray) -> float:
        """
        Returns probability [0, 1] that a trade will be profitable.
        Falls back to 0.85 when untrained (don't penalise early signals),
        then transitions to learned win_rate once we have real data.
        """
        if self.model is None:
            # No training data yet → use optimistic prior so signals can fire
            return 0.85 if len(self.training_data) == 0 else max(self.win_rate, 0.55)

        feats = np.nan_to_num(features, nan=0.0).reshape(1, -1)
        try:
            proba = self.model.predict_proba(feats)[0]
            # class index 1 = profitable
            classes = list(self.model.classes_)
            return float(proba[classes.index(1)]) if 1 in classes else 0.5
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return self.win_rate

    def get_weights(self) -> dict:
        return dict(self.strategy_weights)

    def summary(self) -> dict:
        return {
            "model_type":     type(self.model).__name__ if self.model else "None",
            "training_samples": len(self.training_data),
            "win_rate":       round(self.win_rate, 4),
            "total_trades":   self.total_trades,
            "weights":        self.strategy_weights,
        }
