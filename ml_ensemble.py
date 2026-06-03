"""
Layer 3: ML Ensemble — XGBoost + RandomForest + LightGBM
=========================================================
Replaces the single RandomForest in ml_model.py with a 3-model ensemble.
Each model independently predicts trade success probability.
Final probability = weighted average (or majority vote).

Self-learning flow (unchanged from ml_model.py):
  1. Every closed trade → add (features, profitable) to training set
  2. Every MODEL_RETRAIN_EVERY trades (once MIN_TRAINING_SAMPLES reached):
     retrain all 3 models, pick best by cross-val, update strategy weights
  3. predict_success_proba() → ensemble average → ML voter

PDF lesson: the ML layer acts as a FILTER only — it rejects bad signals,
not generates new ones. Same principle as their proposed LLM meta-filter.
"""

import os
import pickle
import logging
import numpy as np

from config import (
    MIN_TRAINING_SAMPLES, MODEL_RETRAIN_EVERY,
    MODEL_PATH, STRATEGY_WEIGHTS,
    FEATURE_IMPORTANCE_THRESHOLD,
)
from indicators import extract_features

logger = logging.getLogger(__name__)

MODEL_PATH_ENS = MODEL_PATH.replace(".pkl", "_ensemble.pkl")


class MLEnsemble:
    def __init__(self):
        self.models           = {}      # {"rf": model, "xgb": model, "lgb": model}
        self.strategy_weights = dict(STRATEGY_WEIGHTS)
        self.training_data    = []      # list of (features_array, label)
        self.closed_since_retrain = 0
        self.total_trades     = 0
        self.win_rate         = 0.5
        self._load()

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        path = MODEL_PATH_ENS
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    saved = pickle.load(f)
                self.models           = saved.get("models", {})
                self.strategy_weights = saved.get("weights", dict(STRATEGY_WEIGHTS))
                self.training_data    = saved.get("data", [])
                self.win_rate         = saved.get("win_rate", 0.5)
                self.total_trades     = saved.get("total_trades", 0)
                n_models = len(self.models)
                logger.info(
                    f"ML Ensemble loaded — {n_models} model(s), "
                    f"{len(self.training_data)} samples, "
                    f"win_rate={self.win_rate:.2%}"
                )
            except Exception as e:
                logger.warning(f"Could not load ensemble: {e}")
        else:
            # Try to migrate from old single-model file
            if os.path.exists(MODEL_PATH):
                try:
                    with open(MODEL_PATH, "rb") as f:
                        old = pickle.load(f)
                    if old.get("model"):
                        self.models["rf"] = old["model"]
                    self.training_data    = old.get("data", [])
                    self.strategy_weights = old.get("weights", dict(STRATEGY_WEIGHTS))
                    self.win_rate         = old.get("win_rate", 0.5)
                    logger.info(f"Migrated from single RF model — "
                                f"{len(self.training_data)} historical samples kept")
                except Exception:
                    pass

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH_ENS), exist_ok=True)
        with open(MODEL_PATH_ENS, "wb") as f:
            pickle.dump({
                "models":       self.models,
                "weights":      self.strategy_weights,
                "data":         self.training_data,
                "win_rate":     self.win_rate,
                "total_trades": self.total_trades,
            }, f)
        logger.info("ML Ensemble saved.")

    # ─── Online learning ──────────────────────────────────────────────────────

    def record_trade_result(self, features: np.ndarray, profitable: bool):
        self.training_data.append((features, int(profitable)))
        self.total_trades += 1
        self.closed_since_retrain += 1

        wins = sum(label for _, label in self.training_data)
        self.win_rate = wins / len(self.training_data)

        logger.info(
            f"Trade recorded — profitable={profitable} | "
            f"win_rate={self.win_rate:.2%} | total={self.total_trades}"
        )

        if (len(self.training_data) >= MIN_TRAINING_SAMPLES and
                self.closed_since_retrain >= MODEL_RETRAIN_EVERY):
            self._retrain()
            self._save()

    def _retrain(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X = np.array([f for f, _ in self.training_data], dtype=np.float32)
        y = np.array([l for _, l in self.training_data], dtype=np.int8)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        cv = min(5, len(y))
        results = {}

        # ── RandomForest ─────────────────────────────────────────────────────
        try:
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="accuracy").mean()
            rf.fit(X, y)
            self.models["rf"] = rf
            results["rf"] = rf_cv
            logger.info(f"RF retrained — cv={rf_cv:.3f}")
        except Exception as e:
            logger.warning(f"RF training failed: {e}")

        # ── XGBoost ──────────────────────────────────────────────────────────
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1
            )
            xgb_cv = cross_val_score(xgb_model, X, y, cv=cv, scoring="accuracy").mean()
            xgb_model.fit(X, y)
            self.models["xgb"] = xgb_model
            results["xgb"] = xgb_cv
            logger.info(f"XGBoost retrained — cv={xgb_cv:.3f}")
        except ImportError:
            logger.info("XGBoost not installed — skipping (pip3 install xgboost)")
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")

        # ── LightGBM ─────────────────────────────────────────────────────────
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                class_weight="balanced", random_state=42, n_jobs=-1,
                verbose=-1
            )
            lgb_cv = cross_val_score(lgb_model, X, y, cv=cv, scoring="accuracy").mean()
            lgb_model.fit(X, y)
            self.models["lgb"] = lgb_model
            results["lgb"] = lgb_cv
            logger.info(f"LightGBM retrained — cv={lgb_cv:.3f}")
        except ImportError:
            logger.info("LightGBM not installed — skipping (pip3 install lightgbm)")
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}")

        self.closed_since_retrain = 0
        logger.info(f"Ensemble retrained — models={list(results.keys())} scores={results}")

        # Update strategy weights from best model's feature importances
        best_name = max(results, key=results.get) if results else None
        if best_name and hasattr(self.models.get(best_name), "feature_importances_"):
            self._update_strategy_weights(self.models[best_name].feature_importances_)

    def _update_strategy_weights(self, importances: np.ndarray):
        """Map feature importances to strategy weights (18-feature vector)."""
        n = 18
        imp = importances[:n] if len(importances) >= n else importances

        affinity = {
            "ema_crossover":  [3, 11, 12],
            "vwap_reversion": [0, 2, 10],
            "supertrend":     [4, 7, 8],
            "breakout":       [5, 6, 1],
            "fvg":            [15, 9],
        }

        raw = {}
        for strat, idxs in affinity.items():
            valid = [i for i in idxs if i < len(imp)]
            raw[strat] = float(imp[valid].sum()) if valid else 0.0

        total = sum(raw.values()) or 1.0
        orig  = STRATEGY_WEIGHTS
        for strat in raw:
            raw[strat] /= total
        for strat in self.strategy_weights:
            self.strategy_weights[strat] = (
                0.7 * raw.get(strat, 0.0) + 0.3 * orig.get(strat, 0.2)
            )
        logger.info(f"Strategy weights updated: {self.strategy_weights}")

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict_success_proba(self, features: np.ndarray) -> float:
        """
        Ensemble prediction: average probability across all trained models.
        Falls back to optimistic prior when untrained (0.85 = don't block early signals).
        """
        if not self.models:
            return 0.85 if len(self.training_data) == 0 else max(self.win_rate, 0.55)

        feats = np.nan_to_num(features, nan=0.0).reshape(1, -1)
        probas = []

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(feats)[0]
                classes = list(model.classes_)
                p = float(proba[classes.index(1)]) if 1 in classes else 0.5
                probas.append(p)
            except Exception as e:
                logger.debug(f"Model {name} predict error: {e}")

        return float(np.mean(probas)) if probas else max(self.win_rate, 0.55)

    def get_weights(self) -> dict:
        return dict(self.strategy_weights)

    def summary(self) -> dict:
        return {
            "models":           list(self.models.keys()),
            "training_samples": len(self.training_data),
            "win_rate":         round(self.win_rate, 4),
            "total_trades":     self.total_trades,
            "weights":          self.strategy_weights,
        }
