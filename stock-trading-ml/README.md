# Stock Trading ML System

End-to-end ML pipeline for quantitative stock trading — supports Taiwan (TWSE) and US markets.

## Pipeline

```
資料收集  →  特徵工程  →  LSTM 走勢預測  →  RF 交易信號  →  回測  →  模擬交易
yfinance     技術指標      方向機率          BUY/SELL/HOLD   Backtrader   Paper trading
TWSE API     GARCH 波動率  報酬率預測        信號機率
```

## Project Structure

```
stock-trading-ml/
├── config.py               # All hyperparameters (dataclasses)
├── pipeline.py             # CLI orchestrator
├── requirements.txt
├── data/
│   └── collector.py        # yfinance + TWSE OpenAPI
├── features/
│   ├── technical.py        # RSI, MACD, BB, ATR, Stochastic, CCI, OBV …
│   └── garch.py            # Rolling GARCH(1,1) conditional volatility
├── models/
│   ├── lstm_model.py       # Stacked LSTM (direction + return heads)
│   └── rf_signal.py        # Random Forest 3-class signal (BUY/HOLD/SELL)
├── backtest/
│   └── engine.py           # Backtrader strategy + vectorised fallback
└── simulation/
    └── paper_trading.py    # Live paper trading with JSON ledger
```

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline: collect → features → train LSTM → train RF → backtest
python pipeline.py --mode full --symbols 2330.TW 2317.TW AAPL

# Train only
python pipeline.py --mode train --symbols 2330.TW --start 2018-01-01 --end 2023-12-31

# Backtest with pre-trained models
python pipeline.py --mode backtest --symbols 2330.TW

# One paper-trading tick (requires trained models)
python pipeline.py --mode simulate
```

## Features

### Technical Indicators
| Feature | Description |
|---------|-------------|
| RSI(14) | Relative Strength Index |
| MACD(12,26,9) | Moving Average Convergence Divergence + histogram |
| BB(20,2) | Bollinger Bands %B |
| ATR(14) | Average True Range (normalised) |
| SMA/EMA | 5/10/20/50/200-day moving averages |
| Stochastic %K/%D | Momentum oscillator |
| Williams %R | Momentum oscillator |
| CCI(20) | Commodity Channel Index |
| OBV | On-Balance Volume |
| ROC | Rate of Change (5/10/20-day) |

### GARCH Volatility
Rolling GARCH(1,1) fitted on 252-day window of log-returns.
One-step-ahead conditional volatility used for:
- Volatility-targeting position sizing in backtest
- Feature input to LSTM and RF

### LSTM Architecture
- Input: 60-day sequence of scaled features
- 2-layer stacked LSTM (hidden=128, dropout=0.2)
- Dual output heads:
  - Classification: next-day direction probability
  - Regression: next-day log-return magnitude
- Loss: BCE + 0.5 × MSE

### RF Signal Generator
- RandomForest (300 trees, max_depth=8, balanced classes)
- Labels: forward 5-day return > 1% → BUY, < -1% → SELL, else HOLD
- Meta-features: LSTM direction_prob + return_pred stacked with all technical features

### Backtesting
- Backtrader with TWSE commission (0.1425%) + sell tax (0.3%)
- Volatility-targeting position sizing: weight = base_risk / garch_vol
- Analyzers: Sharpe ratio, max drawdown, trade win-rate

### Paper Trading
- Fetches latest data on each call
- Full ML inference pipeline per tick
- Positions and trade log persisted to `results/paper_trading_ledger.json`

## Configuration

Edit `config.py` to adjust:
- `DataConfig`: symbols, date range
- `FeatureConfig`: indicator periods, GARCH order
- `LSTMConfig`: architecture, training hyperparameters
- `RFConfig`: forest size, signal horizon, return threshold
- `BacktestConfig`: capital, commission, slippage
- `SimulationConfig`: position sizing, max positions

## TWSE API

The `DataCollector` also supports the official TWSE OpenAPI:

```python
from data.collector import DataCollector
c = DataCollector()
df = c.fetch_twse_daily("20240101")          # full market daily
df = c.fetch_twse_institutional("2330")      # 三大法人 for TSMC
df = c.fetch_twse_margin("2330")             # 融資融券
```
