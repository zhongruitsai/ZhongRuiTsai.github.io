"""
Random Forest trading signal generator.

Labels
------
  BUY  ( 1) : forward return over `horizon` days >  +threshold
  SELL (-1) : forward return over `horizon` days < -threshold
  HOLD ( 0) : otherwise

The RF receives all technical / GARCH features PLUS the LSTM outputs
(direction_prob, return_pred) as additional meta-features.
"""

import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "rsi", "macd", "macd_hist",
    "bb_pct", "atr_pct",
    "golden_cross",
    "vol_ratio",
    "roc_5", "roc_10", "roc_20",
    "stoch_k", "stoch_d",
    "willr", "cci",
    "log_ret",
    "garch_vol",
    # LSTM meta-features (may be absent before LSTM stage)
    "direction_prob",
    "return_pred",
]

SIGNAL_MAP = {0: "HOLD", 1: "BUY", -1: "SELL"}


class RFSignalGenerator:
    """Scikit-learn RandomForest wrapper for 3-class signal generation."""

    def __init__(self, cfg=None, model_dir: str = "saved_models"):
        self.n_estimators  = getattr(cfg, "n_estimators",     300) if cfg else 300
        self.max_depth     = getattr(cfg, "max_depth",          8) if cfg else   8
        self.min_samples   = getattr(cfg, "min_samples_leaf",  20) if cfg else  20
        self.horizon       = getattr(cfg, "signal_horizon",     5) if cfg else   5
        self.threshold     = getattr(cfg, "signal_threshold", 0.01) if cfg else 0.01

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.rf:     Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler]        = None

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, symbol: str = "model") -> str:
        """Train on *df* and return a classification report string."""
        X, y = self._build_dataset(df)

        n_train = int(len(X) * 0.8)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.rf.fit(X_train_s, y_train)

        report = classification_report(
            y_test, self.rf.predict(X_test_s),
            target_names=["SELL", "HOLD", "BUY"],
            labels=[-1, 0, 1],
        )
        logger.info("RF signal report for %s:\n%s", symbol, report)
        self._save(symbol)
        return report

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns [signal, signal_prob_buy, signal_prob_sell]."""
        if self.rf is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[cols].values
        X_s = self.scaler.transform(X)
        proba  = self.rf.predict_proba(X_s)
        labels = self.rf.classes_

        buy_idx  = list(labels).index(1)  if 1  in labels else None
        sell_idx = list(labels).index(-1) if -1 in labels else None

        signals = self.rf.predict(X_s)
        out = pd.DataFrame(index=df.index)
        out["signal"] = signals
        out["signal_label"] = out["signal"].map(SIGNAL_MAP)
        out["signal_prob_buy"]  = proba[:, buy_idx]  if buy_idx  is not None else 0.0
        out["signal_prob_sell"] = proba[:, sell_idx] if sell_idx is not None else 0.0
        return out

    def feature_importance(self, df: pd.DataFrame) -> pd.Series:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        return pd.Series(self.rf.feature_importances_, index=cols).sort_values(ascending=False)

    def save(self, symbol: str):
        self._save(symbol)

    def load(self, symbol: str):
        path = os.path.join(self.model_dir, f"rf_{symbol}.pkl")
        bundle = joblib.load(path)
        self.rf     = bundle["rf"]
        self.scaler = bundle["scaler"]
        logger.info("Loaded RF model from %s", path)

    # ------------------------------------------------------------------

    def _build_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        forward_ret = df["close"].pct_change(self.horizon).shift(-self.horizon)

        mask = forward_ret.notna()
        X = df.loc[mask, cols].values
        r = forward_ret[mask].values
        y = np.where(r > self.threshold, 1, np.where(r < -self.threshold, -1, 0))
        return X, y

    def _save(self, symbol: str):
        path = os.path.join(self.model_dir, f"rf_{symbol}.pkl")
        joblib.dump({"rf": self.rf, "scaler": self.scaler}, path)
        logger.info("Saved RF model to %s", path)
