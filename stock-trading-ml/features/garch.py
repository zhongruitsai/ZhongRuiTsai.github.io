"""
GARCH(1,1) rolling volatility feature.

For each row we fit a GARCH(1,1) model on the preceding `lookback` days of
log-returns and record the one-step-ahead conditional volatility forecast.
This is computationally expensive so results are cached to disk.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_garch_volatility(
    df: pd.DataFrame,
    lookback: int = 252,
    p: int = 1,
    q: int = 1,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Append ``garch_vol`` (annualised conditional std) to *df*.

    Parameters
    ----------
    df          : DataFrame with a ``log_ret`` column.
    lookback    : number of historical observations used for each GARCH fit.
    p, q        : GARCH order.
    cache_path  : if provided, load from / save to this parquet file.
    """
    try:
        from arch import arch_model  # type: ignore
    except ImportError:
        logger.warning("arch package not installed; skipping GARCH feature")
        df["garch_vol"] = np.nan
        return df

    df = df.copy()

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if "garch_vol" in cached.columns and cached.index.equals(df.index):
            df["garch_vol"] = cached["garch_vol"]
            logger.info("Loaded GARCH volatility from cache")
            return df

    log_ret = df["log_ret"].values * 100  # scale to % for numerical stability
    n = len(log_ret)
    garch_vol = np.full(n, np.nan)

    logger.info("Fitting rolling GARCH(%d,%d) on %d rows (lookback=%d)…", p, q, n, lookback)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(lookback, n):
            window = log_ret[i - lookback : i]
            try:
                am = arch_model(window, vol="Garch", p=p, q=q, rescale=False)
                res = am.fit(disp="off", show_warning=False)
                forecast = res.forecast(horizon=1, reindex=False)
                # annualise: sqrt(252) * daily std
                daily_var = forecast.variance.values[-1, 0]
                garch_vol[i] = np.sqrt(daily_var * 252) / 100  # back to decimal
            except Exception:
                garch_vol[i] = np.nan

    df["garch_vol"] = garch_vol

    # forward-fill the initial NaN window with realised vol as a fallback
    rv = df["log_ret"].rolling(lookback).std() * np.sqrt(252)
    df["garch_vol"] = df["garch_vol"].fillna(rv)

    if cache_path:
        df[["garch_vol"]].to_parquet(cache_path)
        logger.info("Saved GARCH volatility to %s", cache_path)

    return df
