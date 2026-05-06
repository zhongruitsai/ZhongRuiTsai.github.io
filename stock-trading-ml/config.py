from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # Taiwan stocks use ".TW" suffix on yfinance (e.g. "2330.TW" for TSMC)
    tw_symbols: List[str] = field(default_factory=lambda: ["2330.TW", "2317.TW", "2454.TW"])
    us_symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    tw_api_base: str = "https://openapi.twse.com.tw/v1"


@dataclass
class FeatureConfig:
    # Technical indicator periods
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20

    # GARCH settings
    garch_p: int = 1
    garch_q: int = 1
    garch_lookback: int = 252  # trading days used to fit each rolling GARCH window

    # Sequence length for LSTM
    seq_len: int = 60


@dataclass
class LSTMConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio implied as remainder


@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 20
    # horizon in days for labelling: if return > threshold -> BUY, < -threshold -> SELL
    signal_horizon: int = 5
    signal_threshold: float = 0.01  # 1 %


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0  # TWD
    commission: float = 0.001425       # TWSE brokerage (one-way)
    tax: float = 0.003                 # Taiwan sell-side securities transaction tax
    stake: int = 1000                  # one board lot = 1 000 shares on TWSE
    slippage: float = 0.001


@dataclass
class SimulationConfig:
    initial_cash: float = 1_000_000.0
    position_size: float = 0.1   # 10 % of portfolio per trade
    max_positions: int = 5
    rebalance_freq: str = "daily"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model_dir: str = "saved_models"
    data_dir: str = "cached_data"
    results_dir: str = "results"
