"""
LSTM trend predictor.

Predicts the *direction* of the next-day close (1 = up, 0 = down) and the
expected *return magnitude* (regression head) from a sliding window of
technical / GARCH features.

Architecture
------------
  Input  →  [BatchNorm]  →  LSTM (stacked)  →  Dropout
         →  Linear(1) [direction logit]
         →  Linear(1) [return regression]
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "atr_pct",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20",
    "golden_cross",
    "vol_ratio", "obv",
    "roc_5", "roc_10", "roc_20",
    "stoch_k", "stoch_d",
    "willr", "cci",
    "log_ret",
    "garch_vol",
]


class _SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray):
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_cls[idx], self.y_reg[idx]


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.drop       = nn.Dropout(dropout)
        self.cls_head   = nn.Linear(hidden, 1)   # direction logit
        self.reg_head   = nn.Linear(hidden, 1)   # return magnitude

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])             # last time-step
        return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)


class LSTMPredictor:
    """Train / infer wrapper around _LSTMNet."""

    def __init__(self, cfg=None, model_dir: str = "saved_models"):
        self.seq_len    = getattr(cfg, "seq_len",       60)  if cfg else 60
        self.hidden     = getattr(cfg, "hidden_size",  128)  if cfg else 128
        self.layers     = getattr(cfg, "num_layers",     2)  if cfg else  2
        self.dropout    = getattr(cfg, "dropout",      0.2)  if cfg else 0.2
        self.batch_size = getattr(cfg, "batch_size",    64)  if cfg else 64
        self.epochs     = getattr(cfg, "epochs",        50)  if cfg else 50
        self.lr         = getattr(cfg, "learning_rate", 1e-3) if cfg else 1e-3
        self.train_r    = getattr(cfg, "train_ratio",  0.70) if cfg else 0.70
        self.val_r      = getattr(cfg, "val_ratio",    0.15) if cfg else 0.15

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.scaler: Optional[StandardScaler] = None
        self.model:  Optional[_LSTMNet]       = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, symbol: str = "model") -> dict:
        """Train on *df*. Returns a dict of training metrics."""
        X_seq, y_cls, y_reg = self._build_sequences(df, fit_scaler=True)

        n = len(X_seq)
        n_train = int(n * self.train_r)
        n_val   = int(n * self.val_r)

        splits = {
            "train": (X_seq[:n_train],     y_cls[:n_train],     y_reg[:n_train]),
            "val":   (X_seq[n_train:n_train+n_val],
                      y_cls[n_train:n_train+n_val],
                      y_reg[n_train:n_train+n_val]),
        }

        loaders = {
            k: DataLoader(_SequenceDataset(*v), batch_size=self.batch_size,
                          shuffle=(k == "train"))
            for k, v in splits.items()
        }

        input_size = X_seq.shape[2]
        self.model = _LSTMNet(input_size, self.hidden, self.layers, self.dropout).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        cls_loss_fn = nn.BCEWithLogitsLoss()
        reg_loss_fn = nn.MSELoss()

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for xb, yc, yr in loaders["train"]:
                xb, yc, yr = xb.to(self.device), yc.to(self.device), yr.to(self.device)
                optimizer.zero_grad()
                logit, ret_pred = self.model(xb)
                loss = cls_loss_fn(logit, yc) + 0.5 * reg_loss_fn(ret_pred, yr)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            val_loss, val_acc = self._evaluate(loaders["val"], cls_loss_fn, reg_loss_fn)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss / len(loaders["train"]))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val:
                best_val = val_loss
                self._save(symbol)

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %3d | train=%.4f | val=%.4f | acc=%.3f",
                    epoch, history["train_loss"][-1], val_loss, val_acc
                )

        self._load(symbol)   # restore best checkpoint
        return history

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns [direction_prob, return_pred]."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_seq, _, _ = self._build_sequences(df, fit_scaler=False)
        self.model.eval()
        all_prob, all_ret = [], []
        with torch.no_grad():
            for i in range(0, len(X_seq), self.batch_size):
                xb = torch.tensor(X_seq[i:i+self.batch_size], dtype=torch.float32).to(self.device)
                logit, ret_pred = self.model(xb)
                all_prob.append(torch.sigmoid(logit).cpu().numpy())
                all_ret.append(ret_pred.cpu().numpy())
        prob = np.concatenate(all_prob)
        ret  = np.concatenate(all_ret)
        idx  = df.index[self.seq_len:]
        return pd.DataFrame({"direction_prob": prob, "return_pred": ret}, index=idx)

    def save(self, symbol: str):
        self._save(symbol)

    def load(self, symbol: str, input_size: int):
        self.model = _LSTMNet(input_size, self.hidden, self.layers, self.dropout).to(self.device)
        self._load(symbol)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_sequences(
        self, df: pd.DataFrame, fit_scaler: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        raw = df[cols].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            raw = self.scaler.fit_transform(raw)
        else:
            raw = self.scaler.transform(raw)

        future_ret = df["log_ret"].shift(-1).fillna(0).values

        X, y_cls, y_reg = [], [], []
        for i in range(self.seq_len, len(raw)):
            X.append(raw[i - self.seq_len : i])
            y_cls.append(1.0 if future_ret[i] > 0 else 0.0)
            y_reg.append(float(future_ret[i]))

        return np.array(X), np.array(y_cls), np.array(y_reg)

    def _evaluate(self, loader, cls_fn, reg_fn) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yc, yr in loader:
                xb, yc, yr = xb.to(self.device), yc.to(self.device), yr.to(self.device)
                logit, ret_pred = self.model(xb)
                loss = cls_fn(logit, yc) + 0.5 * reg_fn(ret_pred, yr)
                total_loss += loss.item()
                pred = (torch.sigmoid(logit) > 0.5).float()
                correct += (pred == yc).sum().item()
                total   += len(yc)
        return total_loss / len(loader), correct / total if total else 0.0

    def _save(self, symbol: str):
        path = os.path.join(self.model_dir, f"lstm_{symbol}.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler":      self.scaler,
            "config": {
                "hidden":  self.hidden,
                "layers":  self.layers,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
            },
        }, path)

    def _load(self, symbol: str):
        path = os.path.join(self.model_dir, f"lstm_{symbol}.pt")
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt["config"]
        # infer input size from saved weight
        first_layer_w = ckpt["model_state"]["lstm.weight_ih_l0"]
        input_size = first_layer_w.shape[1]
        if self.model is None:
            self.model = _LSTMNet(
                input_size, cfg["hidden"], cfg["layers"], cfg["dropout"]
            ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.scaler = ckpt["scaler"]
        logger.info("Loaded LSTM checkpoint from %s", path)
