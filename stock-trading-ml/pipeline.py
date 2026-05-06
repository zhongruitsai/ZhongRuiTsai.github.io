"""
Main pipeline orchestrator.

Stages
------
  1. collect   – download OHLCV data (yfinance / TWSE)
  2. features  – technical indicators + GARCH volatility
  3. lstm      – train / load LSTM trend predictor
  4. rf        – train / load RF signal generator
  5. backtest  – run Backtrader simulation
  6. simulate  – one paper-trading tick (live mode)

Quick start
-----------
  python pipeline.py --mode train  --symbols 2330.TW AAPL
  python pipeline.py --mode backtest
  python pipeline.py --mode simulate
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import pandas as pd

from config import Config
from data.collector import DataCollector
from features.technical import add_technical_indicators
from features.garch import add_garch_volatility
from models.lstm_model import LSTMPredictor
from models.rf_signal import RFSignalGenerator
from backtest.engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def build_features(
    raw: Dict[str, pd.DataFrame],
    cfg: Config,
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for symbol, df in raw.items():
        logger.info("[%s] Computing technical indicators", symbol)
        df = add_technical_indicators(df, cfg.features)

        cache_path = os.path.join(
            cfg.data_dir, f"garch_{symbol.replace('.', '_')}.parquet"
        )
        logger.info("[%s] Fitting GARCH volatility", symbol)
        df = add_garch_volatility(
            df,
            lookback=cfg.features.garch_lookback,
            p=cfg.features.garch_p,
            q=cfg.features.garch_q,
            cache_path=cache_path,
        )
        result[symbol] = df
    return result


def train_lstm(
    feature_data: Dict[str, pd.DataFrame],
    cfg: Config,
) -> Dict[str, LSTMPredictor]:
    predictors: Dict[str, LSTMPredictor] = {}
    for symbol, df in feature_data.items():
        logger.info("[%s] Training LSTM", symbol)
        lstm = LSTMPredictor(cfg.lstm, cfg.model_dir)
        history = lstm.fit(df, symbol=symbol)
        logger.info(
            "[%s] LSTM final val_acc=%.3f", symbol, history["val_acc"][-1]
        )
        predictors[symbol] = lstm
    return predictors


def run_lstm_inference(
    feature_data: Dict[str, pd.DataFrame],
    predictors: Dict[str, LSTMPredictor],
) -> Dict[str, pd.DataFrame]:
    enriched: Dict[str, pd.DataFrame] = {}
    for symbol, df in feature_data.items():
        if symbol not in predictors:
            enriched[symbol] = df
            continue
        logger.info("[%s] LSTM inference", symbol)
        preds = predictors[symbol].predict(df)
        enriched[symbol] = df.join(preds, how="left")
    return enriched


def train_rf(
    enriched_data: Dict[str, pd.DataFrame],
    cfg: Config,
) -> Dict[str, RFSignalGenerator]:
    generators: Dict[str, RFSignalGenerator] = {}
    for symbol, df in enriched_data.items():
        logger.info("[%s] Training RF signal generator", symbol)
        rf = RFSignalGenerator(cfg.rf, cfg.model_dir)
        report = rf.fit(df, symbol=symbol)
        logger.info("[%s] RF report:\n%s", symbol, report)
        generators[symbol] = rf
    return generators


def generate_signals(
    enriched_data: Dict[str, pd.DataFrame],
    generators: Dict[str, RFSignalGenerator],
) -> Dict[str, pd.DataFrame]:
    signals: Dict[str, pd.DataFrame] = {}
    for symbol, df in enriched_data.items():
        if symbol not in generators:
            continue
        logger.info("[%s] Generating RF signals", symbol)
        signals[symbol] = generators[symbol].predict(df)
    return signals


def run_backtest(
    feature_data: Dict[str, pd.DataFrame],
    signals: Dict[str, pd.DataFrame],
    cfg: Config,
) -> dict:
    engine = BacktestEngine(cfg.backtest, cfg.results_dir)
    metrics = engine.run(feature_data, signals, plot=False)
    logger.info("Backtest metrics: %s", metrics)
    return metrics


def run_simulate(cfg: Config, symbols: List[str]):
    from simulation.paper_trading import PaperTrader
    trader = PaperTrader(
        cfg,
        ledger_path=os.path.join(cfg.results_dir, "paper_trading_ledger.json"),
    )
    snapshot = trader.run_once(symbols=symbols)
    logger.info("Paper-trade snapshot: %s", snapshot)
    return snapshot


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stock Trading ML Pipeline")
    p.add_argument(
        "--mode",
        choices=["train", "backtest", "simulate", "full"],
        default="full",
        help="Pipeline mode",
    )
    p.add_argument(
        "--symbols", nargs="*",
        default=None,
        help="Symbols to process (e.g. 2330.TW AAPL). Defaults to config.",
    )
    p.add_argument("--start",  default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",    default=None, help="End date   YYYY-MM-DD")
    p.add_argument("--no-cache", action="store_true", help="Skip data cache")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    if args.start:
        cfg.data.start_date = args.start
    if args.end:
        cfg.data.end_date = args.end

    symbols = args.symbols or (cfg.data.tw_symbols + cfg.data.us_symbols)

    os.makedirs(cfg.model_dir,   exist_ok=True)
    os.makedirs(cfg.data_dir,    exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 – Data collection
    # ------------------------------------------------------------------
    logger.info("=== Stage 1: Data collection ===")
    collector = DataCollector(cfg.data_dir)
    raw_data = collector.fetch(
        symbols,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        use_cache=not args.no_cache,
    )
    if not raw_data:
        logger.error("No data fetched. Check symbols / network.")
        sys.exit(1)

    if args.mode == "simulate":
        run_simulate(cfg, symbols)
        return

    # ------------------------------------------------------------------
    # Stage 2 – Feature engineering
    # ------------------------------------------------------------------
    logger.info("=== Stage 2: Feature engineering ===")
    feature_data = build_features(raw_data, cfg)

    if args.mode == "backtest":
        # Load pre-trained models
        predictors = {}
        for sym in feature_data:
            lstm = LSTMPredictor(cfg.lstm, cfg.model_dir)
            try:
                lstm.load(sym, input_size=1)
                predictors[sym] = lstm
            except FileNotFoundError:
                logger.warning("No saved LSTM for %s – skipping", sym)

        enriched = run_lstm_inference(feature_data, predictors)

        generators = {}
        for sym in enriched:
            rf = RFSignalGenerator(cfg.rf, cfg.model_dir)
            try:
                rf.load(sym)
                generators[sym] = rf
            except FileNotFoundError:
                logger.warning("No saved RF for %s – skipping", sym)

        signals = generate_signals(enriched, generators)
        run_backtest(feature_data, signals, cfg)
        return

    # ------------------------------------------------------------------
    # Stage 3 – LSTM training
    # ------------------------------------------------------------------
    logger.info("=== Stage 3: LSTM training ===")
    predictors = train_lstm(feature_data, cfg)

    # ------------------------------------------------------------------
    # Stage 4 – LSTM inference (enrich features)
    # ------------------------------------------------------------------
    logger.info("=== Stage 4: LSTM inference ===")
    enriched = run_lstm_inference(feature_data, predictors)

    # ------------------------------------------------------------------
    # Stage 5 – RF training
    # ------------------------------------------------------------------
    logger.info("=== Stage 5: RF signal training ===")
    generators = train_rf(enriched, cfg)

    # ------------------------------------------------------------------
    # Stage 6 – Generate signals
    # ------------------------------------------------------------------
    logger.info("=== Stage 6: Signal generation ===")
    signals = generate_signals(enriched, generators)

    if args.mode in ("train", "full"):
        # ------------------------------------------------------------------
        # Stage 7 – Backtest
        # ------------------------------------------------------------------
        logger.info("=== Stage 7: Backtest ===")
        metrics = run_backtest(feature_data, signals, cfg)

        # Save metrics
        import json
        metrics_path = os.path.join(cfg.results_dir, "backtest_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Saved backtest metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
