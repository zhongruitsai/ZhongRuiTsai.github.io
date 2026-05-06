"""
Feature engineering – classical technical indicators.

All features are added in-place to the DataFrame that already contains
['open', 'high', 'low', 'close', 'volume'] columns.
"""

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Append technical-indicator columns to *df* and return it."""
    df = df.copy()

    rsi_p    = getattr(cfg, "rsi_period",       14) if cfg else 14
    macd_f   = getattr(cfg, "macd_fast",        12) if cfg else 12
    macd_s   = getattr(cfg, "macd_slow",        26) if cfg else 26
    macd_sig = getattr(cfg, "macd_signal",       9) if cfg else  9
    bb_p     = getattr(cfg, "bb_period",        20) if cfg else 20
    bb_std   = getattr(cfg, "bb_std",          2.0) if cfg else 2.0
    atr_p    = getattr(cfg, "atr_period",       14) if cfg else 14
    vol_p    = getattr(cfg, "volume_ma_period", 20) if cfg else 20

    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # --- RSI -----------------------------------------------------------
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=rsi_p - 1, min_periods=rsi_p).mean()
    avg_loss = loss.ewm(com=rsi_p - 1, min_periods=rsi_p).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # --- MACD ----------------------------------------------------------
    ema_fast   = c.ewm(span=macd_f, adjust=False).mean()
    ema_slow   = c.ewm(span=macd_s, adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=macd_sig, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # --- Bollinger Bands -----------------------------------------------
    bb_mid = c.rolling(bb_p).mean()
    bb_sd  = c.rolling(bb_p).std()
    df["bb_upper"] = bb_mid + bb_std * bb_sd
    df["bb_lower"] = bb_mid - bb_std * bb_sd
    df["bb_mid"]   = bb_mid
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # --- ATR (Average True Range) --------------------------------------
    hl  = h - lo
    hpc = (h - c.shift()).abs()
    lpc = (lo - c.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=atr_p, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c  # normalised

    # --- Moving Averages -----------------------------------------------
    for p in [5, 10, 20, 50, 200]:
        df[f"sma_{p}"] = c.rolling(p).mean()
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()

    # --- Golden / Death Cross flag ------------------------------------
    df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # --- Volume features -----------------------------------------------
    df["vol_ma"]     = v.rolling(vol_p).mean()
    df["vol_ratio"]  = v / df["vol_ma"].replace(0, np.nan)
    df["obv"]        = (np.sign(c.diff()) * v).fillna(0).cumsum()

    # --- Momentum / Rate-of-change ------------------------------------
    for p in [5, 10, 20]:
        df[f"roc_{p}"] = c.pct_change(p)

    # --- Stochastic Oscillator (%K / %D) ------------------------------
    lo_min = lo.rolling(14).min()
    hi_max = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - lo_min) / (hi_max - lo_min + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- Williams %R ---------------------------------------------------
    df["willr"] = -100 * (hi_max - c) / (hi_max - lo_min + 1e-9)

    # --- Commodity Channel Index (CCI) --------------------------------
    tp = (h + lo + c) / 3
    tp_ma = tp.rolling(20).mean()
    tp_sd = tp.rolling(20).std()
    df["cci"] = (tp - tp_ma) / (0.015 * tp_sd + 1e-9)

    # --- Log-return (label helper) ------------------------------------
    df["log_ret"] = np.log(c / c.shift(1))

    df.dropna(inplace=True)
    return df
