"""
Backtrader-based backtesting engine.

Strategy logic
--------------
  • Uses pre-computed signals from the RF model.
  • BUY  signal  → enter long (if not already in position).
  • SELL signal  → close long position.
  • Position sizing is based on GARCH volatility targeting:
    target_weight = base_risk / garch_vol
    where base_risk is a configurable daily volatility budget (e.g. 0.01 = 1 %).
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import backtrader as bt
    BT_AVAILABLE = True
except ImportError:
    BT_AVAILABLE = False
    logger.warning("backtrader not installed; BacktestEngine will use a fallback vectorised engine")


# ======================================================================
# Backtrader strategy
# ======================================================================

if BT_AVAILABLE:
    class MLSignalStrategy(bt.Strategy):
        """Buy/sell driven by pre-computed signal column in the data feed."""

        params = (
            ("signal_col",   "signal"),
            ("garch_vol_col","garch_vol"),
            ("base_risk",    0.01),     # daily vol budget
            ("commission",   0.001425),
            ("tax",          0.003),
            ("max_weight",   0.25),     # max single-position weight
            ("verbose",      False),
        )

        def __init__(self):
            self.signals   = {d: d._dataname for d in self.datas}
            self.order_map: Dict = {}

        def next(self):
            for data in self.datas:
                self._process(data)

        def _process(self, data):
            symbol = data._name
            try:
                sig      = data.signal[0]
                gvol     = data.garch_vol[0]
            except (AttributeError, IndexError):
                return

            pos = self.getposition(data).size
            portfolio_val = self.broker.getvalue()

            if sig == 1 and pos == 0:
                # Volatility-targeting position size
                vol = max(gvol, 0.005)   # floor at 0.5 % to avoid huge bets
                weight = min(self.p.base_risk / vol, self.p.max_weight)
                size = int(portfolio_val * weight / data.close[0])
                if size > 0:
                    self.buy(data=data, size=size)
                    if self.p.verbose:
                        logger.debug("BUY  %s  size=%d  w=%.2f", symbol, size, weight)

            elif sig == -1 and pos > 0:
                self.sell(data=data, size=pos)
                if self.p.verbose:
                    logger.debug("SELL %s  size=%d", symbol, pos)

        def notify_order(self, order):
            if order.status in (order.Completed,):
                logger.debug(
                    "Order %s %s @ %.2f  cost=%.2f  comm=%.2f",
                    "BUY" if order.isbuy() else "SELL",
                    order.data._name,
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm,
                )


    class _SignalDataFeed(bt.feeds.PandasData):
        """Extended PandasData that carries signal and garch_vol lines."""
        lines = ("signal", "garch_vol")
        params = (
            ("signal",    -1),
            ("garch_vol", -1),
        )


# ======================================================================
# Engine wrapper
# ======================================================================

class BacktestEngine:

    def __init__(self, cfg=None, results_dir: str = "results"):
        self.initial_cash = getattr(cfg, "initial_cash", 1_000_000.0) if cfg else 1_000_000.0
        self.commission   = getattr(cfg, "commission",   0.001425)     if cfg else 0.001425
        self.tax          = getattr(cfg, "tax",          0.003)        if cfg else 0.003
        self.slippage     = getattr(cfg, "slippage",     0.001)        if cfg else 0.001
        self.results_dir  = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.DataFrame],
        plot: bool = False,
    ) -> dict:
        """
        Run the backtest.

        Parameters
        ----------
        data_dict    : {symbol: OHLCV+features DataFrame}
        signals_dict : {symbol: DataFrame with 'signal' column (from RF)}

        Returns a metrics dictionary.
        """
        if BT_AVAILABLE:
            return self._run_backtrader(data_dict, signals_dict, plot)
        else:
            return self._run_vectorised(data_dict, signals_dict)

    # ------------------------------------------------------------------
    # Backtrader path
    # ------------------------------------------------------------------

    def _run_backtrader(
        self, data_dict, signals_dict, plot: bool
    ) -> dict:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.broker.set_slippage_perc(self.slippage)

        for symbol, df in data_dict.items():
            if symbol not in signals_dict:
                continue
            merged = df.join(signals_dict[symbol][["signal"]], how="left")
            if "garch_vol" not in merged.columns:
                merged["garch_vol"] = 0.15  # fallback 15 % annualised vol
            merged["signal"] = merged["signal"].fillna(0)

            feed = _SignalDataFeed(
                dataname=merged,
                datetime=None,
                open="open", high="high", low="low",
                close="close", volume="volume",
                openinterest=-1,
                signal="signal",
                garch_vol="garch_vol",
            )
            cerebro.adddata(feed, name=symbol)

        cerebro.addstrategy(
            MLSignalStrategy,
            commission=self.commission,
            tax=self.tax,
            verbose=True,
        )
        cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe",    riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown,     _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns,      _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer,_name="trades")

        results = cerebro.run()
        strat = results[0]

        final_val = cerebro.broker.getvalue()
        metrics = self._extract_metrics(strat, final_val)
        logger.info("Backtest complete | final_value=%.2f | metrics=%s", final_val, metrics)

        if plot:
            try:
                fig = cerebro.plot(style="candlestick")[0][0]
                fig.savefig(os.path.join(self.results_dir, "backtest_plot.png"), dpi=150)
            except Exception as e:
                logger.warning("Plot failed: %s", e)

        return metrics

    def _extract_metrics(self, strat, final_val: float) -> dict:
        metrics: dict = {"final_value": final_val}

        try:
            sr = strat.analyzers.sharpe.get_analysis()
            metrics["sharpe_ratio"] = sr.get("sharperatio", None)
        except Exception:
            pass

        try:
            dd = strat.analyzers.drawdown.get_analysis()
            metrics["max_drawdown_pct"] = dd.get("max", {}).get("drawdown", None)
        except Exception:
            pass

        try:
            ret = strat.analyzers.returns.get_analysis()
            metrics["total_return_pct"] = ret.get("rtot", None)
            metrics["annual_return_pct"] = ret.get("rnorm100", None)
        except Exception:
            pass

        try:
            tr = strat.analyzers.trades.get_analysis()
            metrics["total_trades"]   = tr.get("total", {}).get("total", 0)
            metrics["won_trades"]     = tr.get("won",   {}).get("total", 0)
            metrics["lost_trades"]    = tr.get("lost",  {}).get("total", 0)
            w = metrics["won_trades"]
            t = metrics["total_trades"]
            metrics["win_rate"] = w / t if t else None
        except Exception:
            pass

        return metrics

    # ------------------------------------------------------------------
    # Vectorised fallback (no backtrader)
    # ------------------------------------------------------------------

    def _run_vectorised(self, data_dict, signals_dict) -> dict:
        logger.info("Running vectorised backtest (backtrader not installed)")
        equity = self.initial_cash
        all_daily_rets = []

        for symbol, df in data_dict.items():
            if symbol not in signals_dict:
                continue
            sig_df = signals_dict[symbol][["signal"]].reindex(df.index).fillna(0)
            sig = sig_df["signal"].values
            close = df["close"].values
            daily_rets = []
            position = 0
            entry_price = 0.0

            for i in range(1, len(close)):
                if sig[i - 1] == 1 and position == 0:
                    position = 1
                    entry_price = close[i] * (1 + self.slippage)
                elif sig[i - 1] == -1 and position == 1:
                    ret = (close[i] * (1 - self.slippage) - entry_price) / entry_price
                    ret -= self.commission * 2 + self.tax
                    daily_rets.append(ret)
                    position = 0

            all_daily_rets.extend(daily_rets)

        if not all_daily_rets:
            return {"final_value": equity, "total_return_pct": 0.0}

        r = np.array(all_daily_rets)
        total_ret = r.sum()
        sharpe = (r.mean() / (r.std() + 1e-9)) * np.sqrt(252)
        cum = (1 + r).cumprod()
        max_dd = float(((cum.cummax() - cum) / cum.cummax()).max())

        return {
            "final_value":       equity * (1 + total_ret),
            "total_return_pct":  total_ret,
            "sharpe_ratio":      sharpe,
            "max_drawdown_pct":  max_dd,
            "total_trades":      len(r),
        }
