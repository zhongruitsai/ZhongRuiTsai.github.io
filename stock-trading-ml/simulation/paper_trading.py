"""
Paper trading (live simulation) engine.

Workflow per tick (called once per trading day):
  1. Fetch latest OHLCV from yfinance.
  2. Compute technical indicators + GARCH volatility.
  3. Run LSTM to get direction probability.
  4. Run RF to generate signal.
  5. Execute virtual orders against last close price.
  6. Log portfolio state to a JSON ledger.

Usage
-----
  from simulation import PaperTrader
  from config import Config

  cfg = Config()
  trader = PaperTrader(cfg)
  trader.run_once(symbols=["2330.TW", "AAPL"])
"""

import json
import logging
import os
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Position:
    def __init__(self, symbol: str, qty: int, avg_price: float):
        self.symbol    = symbol
        self.qty       = qty
        self.avg_price = avg_price

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealised_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_price)

    def to_dict(self, price: float) -> dict:
        return {
            "symbol":        self.symbol,
            "qty":           self.qty,
            "avg_price":     self.avg_price,
            "current_price": price,
            "market_value":  self.market_value(price),
            "pnl":           self.unrealised_pnl(price),
        }


class PaperTrader:

    def __init__(self, cfg=None, ledger_path: str = "results/paper_trading_ledger.json"):
        from config import Config
        _cfg = cfg if cfg is not None else Config()

        self.initial_cash    = _cfg.simulation.initial_cash
        self.position_size   = _cfg.simulation.position_size
        self.max_positions   = _cfg.simulation.max_positions
        self.commission      = _cfg.backtest.commission
        self.tax             = _cfg.backtest.tax
        self.data_cfg        = _cfg.data
        self.feature_cfg     = _cfg.features
        self.lstm_cfg        = _cfg.lstm
        self.rf_cfg          = _cfg.rf
        self.model_dir       = _cfg.model_dir
        self.data_dir        = _cfg.data_dir

        self.ledger_path = ledger_path
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

        self.cash:      float               = self.initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[dict]          = []

        self._load_ledger()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_once(self, symbols: Optional[List[str]] = None) -> dict:
        """Run one simulation tick. Returns a portfolio snapshot dict."""
        if symbols is None:
            symbols = self.data_cfg.tw_symbols + self.data_cfg.us_symbols

        from data.collector import DataCollector
        from features.technical import add_technical_indicators
        from features.garch import add_garch_volatility
        from models.lstm_model import LSTMPredictor
        from models.rf_signal import RFSignalGenerator

        collector = DataCollector(self.data_dir)

        end   = datetime.today().strftime("%Y-%m-%d")
        # fetch 2 years of history to have enough data for indicators
        start = (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

        raw_data = collector.fetch(symbols, start=start, end=end, use_cache=False)

        orders_executed = []
        prices: Dict[str, float] = {}

        for symbol, df in raw_data.items():
            if df.empty or len(df) < 100:
                continue

            # --- Feature engineering -----------------------------------
            df = add_technical_indicators(df, self.feature_cfg)
            garch_cache = os.path.join(self.data_dir, f"garch_{symbol.replace('.','_')}.parquet")
            df = add_garch_volatility(
                df,
                lookback=self.feature_cfg.garch_lookback,
                cache_path=garch_cache,
            )

            last_price = float(df["close"].iloc[-1])
            prices[symbol] = last_price

            # --- LSTM prediction ---------------------------------------
            lstm = LSTMPredictor(self.lstm_cfg, self.model_dir)
            try:
                lstm.load(symbol, input_size=1)   # input_size inferred from checkpoint
                lstm_preds = lstm.predict(df)
                df = df.join(lstm_preds, how="left")
            except (FileNotFoundError, RuntimeError):
                logger.warning("No LSTM model for %s; skipping LSTM features", symbol)

            # --- RF signal ---------------------------------------------
            rf = RFSignalGenerator(self.rf_cfg, self.model_dir)
            try:
                rf.load(symbol)
                sig_df = rf.predict(df)
                signal = int(sig_df["signal"].iloc[-1])
            except (FileNotFoundError, RuntimeError):
                logger.warning("No RF model for %s; skipping signal generation", symbol)
                continue

            # --- Execute virtual order ---------------------------------
            order = self._execute(symbol, signal, last_price)
            if order:
                orders_executed.append(order)

        snapshot = self._portfolio_snapshot(prices)
        snapshot["orders"] = orders_executed
        snapshot["timestamp"] = datetime.now().isoformat()

        self.trade_log.append(snapshot)
        self._save_ledger()

        logger.info(
            "Paper-trade tick | cash=%.2f | equity=%.2f | positions=%s",
            self.cash, snapshot["equity"], list(self.positions.keys())
        )
        return snapshot

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute(self, symbol: str, signal: int, price: float) -> Optional[dict]:
        pos = self.positions.get(symbol)

        if signal == 1 and pos is None:
            if len(self.positions) >= self.max_positions:
                return None
            budget = self.cash * self.position_size
            qty = int(budget / price)
            if qty <= 0:
                return None
            cost = qty * price * (1 + self.commission)
            if cost > self.cash:
                qty = int(self.cash / (price * (1 + self.commission)))
                cost = qty * price * (1 + self.commission)
            if qty <= 0:
                return None
            self.cash -= cost
            self.positions[symbol] = Position(symbol, qty, price)
            return {"action": "BUY", "symbol": symbol, "qty": qty,
                    "price": price, "cost": cost, "date": str(date.today())}

        elif signal == -1 and pos is not None:
            proceeds = pos.qty * price * (1 - self.commission - self.tax)
            self.cash += proceeds
            del self.positions[symbol]
            return {"action": "SELL", "symbol": symbol, "qty": pos.qty,
                    "price": price, "proceeds": proceeds, "date": str(date.today())}

        return None

    # ------------------------------------------------------------------
    # Portfolio snapshot
    # ------------------------------------------------------------------

    def _portfolio_snapshot(self, prices: Dict[str, float]) -> dict:
        pos_value = sum(
            p.market_value(prices.get(s, p.avg_price))
            for s, p in self.positions.items()
        )
        equity = self.cash + pos_value
        ret_pct = (equity - self.initial_cash) / self.initial_cash * 100

        return {
            "cash":             self.cash,
            "positions_value":  pos_value,
            "equity":           equity,
            "return_pct":       ret_pct,
            "num_positions":    len(self.positions),
            "positions": [
                p.to_dict(prices.get(s, p.avg_price))
                for s, p in self.positions.items()
            ],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_ledger(self):
        with open(self.ledger_path, "w") as f:
            json.dump(
                {
                    "initial_cash": self.initial_cash,
                    "cash":         self.cash,
                    "positions":    {
                        s: {"qty": p.qty, "avg_price": p.avg_price}
                        for s, p in self.positions.items()
                    },
                    "trade_log":    self.trade_log,
                },
                f, indent=2, default=str
            )

    def _load_ledger(self):
        if not os.path.exists(self.ledger_path):
            return
        try:
            with open(self.ledger_path) as f:
                data = json.load(f)
            self.cash = data.get("cash", self.initial_cash)
            for s, p in data.get("positions", {}).items():
                self.positions[s] = Position(s, p["qty"], p["avg_price"])
            self.trade_log = data.get("trade_log", [])
            logger.info("Loaded paper trading ledger with %d log entries", len(self.trade_log))
        except Exception as e:
            logger.warning("Could not load ledger: %s", e)
