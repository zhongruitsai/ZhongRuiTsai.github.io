"""
美股歷史資料回測腳本（內嵌資料版）
策略：雙均線交叉 (MA20 黃金交叉 MA60 => 買入, 死亡交叉 => 賣出)

資料說明：使用 2022-2024 年各股近似收盤價，以真實走勢關鍵節點插值生成。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 參數設定 ──────────────────────────────────────────────
INIT_CASH = 100_000
SHORT_MA  = 20
LONG_MA   = 60
# ─────────────────────────────────────────────────────────


def generate_price_series(milestones: list[tuple], trading_days: pd.DatetimeIndex,
                           seed: int = 42) -> pd.Series:
    """用關鍵節點插值 + 微小雜訊生成仿真股價序列。"""
    rng = np.random.default_rng(seed)
    dates  = [pd.Timestamp(d) for d, _ in milestones]
    prices = [p for _, p in milestones]

    all_prices = []
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        p0, p1 = prices[i], prices[i + 1]
        mask   = (trading_days >= d0) & (trading_days < d1)
        n      = mask.sum()
        if n == 0:
            continue
        t     = np.linspace(0, 1, n)
        trend = p0 + (p1 - p0) * t
        noise = rng.normal(0, p0 * 0.008, n)   # 0.8% 日波動
        chunk = np.maximum(trend + noise, 1.0)
        all_prices.extend(zip(trading_days[mask], chunk))

    s = pd.Series({d: p for d, p in all_prices}, name="Close")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def build_all_data() -> dict[str, pd.Series]:
    """定義各股 2022-2024 關鍵價格節點（近似真實收盤價）。"""
    trading_days = pd.bdate_range("2022-01-03", "2024-12-31")

    stocks = {
        "AAPL": [
            ("2022-01-03", 182), ("2022-06-16", 130), ("2022-08-16", 174),
            ("2022-10-14", 134), ("2023-01-03", 126), ("2023-07-19", 193),
            ("2023-12-29", 193), ("2024-06-28", 210), ("2024-12-31", 244),
        ],
        "MSFT": [
            ("2022-01-03", 311), ("2022-06-16", 242), ("2022-10-14", 215),
            ("2023-01-03", 224), ("2023-07-19", 350), ("2023-12-29", 375),
            ("2024-06-28", 446), ("2024-12-31", 421),
        ],
        "NVDA": [
            ("2022-01-03", 294), ("2022-06-16", 160), ("2022-10-14", 112),
            ("2023-01-03", 146), ("2023-05-25", 389), ("2023-12-29", 495),
            ("2024-06-10", 1208), ("2024-11-21", 1149), ("2024-12-31", 1350),
        ],
        "TSLA": [
            ("2022-01-03", 399), ("2022-08-05", 303), ("2022-11-04", 207),
            ("2022-12-28", 123), ("2023-01-03", 108), ("2023-07-19", 290),
            ("2023-12-29", 253), ("2024-07-05", 248), ("2024-12-31", 404),
        ],
        "SPY": [
            ("2022-01-03", 476), ("2022-06-16", 362), ("2022-10-14", 355),
            ("2023-01-03", 383), ("2023-07-19", 455), ("2023-10-27", 411),
            ("2023-12-29", 475), ("2024-07-05", 549), ("2024-12-31", 585),
        ],
    }

    result = {}
    for ticker, milestones in stocks.items():
        result[ticker] = generate_price_series(
            milestones, trading_days, seed=hash(ticker) % 10000
        )
    return result


def compute_signals(series: pd.Series) -> pd.DataFrame:
    df         = series.to_frame()
    df["MA_S"] = df["Close"].rolling(SHORT_MA).mean()
    df["MA_L"] = df["Close"].rolling(LONG_MA).mean()
    df["Signal"]   = 0
    df.loc[df["MA_S"] > df["MA_L"], "Signal"] = 1
    df["Position"] = df["Signal"].diff()
    return df.dropna()


def backtest(df: pd.DataFrame, ticker: str) -> dict:
    cash, shares, trades = INIT_CASH, 0, []

    for date, row in df.iterrows():
        price = float(row["Close"])
        pos   = float(row["Position"])
        if pos == 1 and cash > 0:
            shares = cash / price
            cash   = 0
            trades.append(("BUY", date, price))
        elif pos == -1 and shares > 0:
            cash   = shares * price
            shares = 0
            trades.append(("SELL", date, price))

    final_price  = float(df["Close"].iloc[-1])
    final_value  = cash + shares * final_price
    total_return = (final_value - INIT_CASH) / INIT_CASH * 100
    bh_return    = (final_price / float(df["Close"].iloc[0]) - 1) * 100
    years        = (df.index[-1] - df.index[0]).days / 365.25
    ann_return   = ((final_value / INIT_CASH) ** (1 / years) - 1) * 100 if years > 0 else 0

    return dict(ticker=ticker, trades=trades, final_value=final_value,
                total_return=total_return, bh_return=bh_return,
                ann_return=ann_return, n_trades=len(trades), df=df)


def plot_results(results: list[dict], save_path: str = "backtest_result.png"):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"US Stock MA Crossover Backtest  2022-01-01 ~ 2024-12-31\n"
        f"MA{SHORT_MA} x MA{LONG_MA}  |  Initial Capital ${INIT_CASH:,}",
        fontsize=13, fontweight="bold"
    )

    for ax, res in zip(axes, results):
        df, ticker = res["df"], res["ticker"]
        ax.plot(df.index, df["Close"], label="Close",         color="#555",       lw=1)
        ax.plot(df.index, df["MA_S"],  label=f"MA{SHORT_MA}", color="royalblue",  lw=1.2)
        ax.plot(df.index, df["MA_L"],  label=f"MA{LONG_MA}",  color="tomato",     lw=1.2)

        buys  = [(d, p) for a, d, p in res["trades"] if a == "BUY"]
        sells = [(d, p) for a, d, p in res["trades"] if a == "SELL"]
        if buys:
            bd, bp = zip(*buys)
            ax.scatter(bd, bp, marker="^", color="green", s=80, zorder=5, label="Buy")
        if sells:
            sd, sp = zip(*sells)
            ax.scatter(sd, sp, marker="v", color="red",   s=80, zorder=5, label="Sell")

        ax.set_title(
            f"{ticker}   Strategy {'+'if res['total_return']>=0 else ''}{res['total_return']:.1f}%  |  "
            f"Buy&Hold {'+'if res['bh_return']>=0 else ''}{res['bh_return']:.1f}%  |  "
            f"Ann. {'+'if res['ann_return']>=0 else ''}{res['ann_return']:.1f}%  |  "
            f"Trades: {res['n_trades']}",
            fontsize=10
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\n圖表已儲存：{save_path}")


def print_summary(results: list[dict]):
    print("\n" + "=" * 72)
    print(f"{'代碼':<8} {'策略報酬':>10} {'買入持有':>10} {'年化報酬':>10} {'交易次數':>8} {'最終資產':>14}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['ticker']:<8} "
            f"{r['total_return']:>+9.1f}% "
            f"{r['bh_return']:>+9.1f}% "
            f"{r['ann_return']:>+9.1f}% "
            f"{r['n_trades']:>8} "
            f"${r['final_value']:>13,.0f}"
        )
    print("=" * 72)
    print(f"\n說明：資料為 2022-2024 年各股近似收盤價（以真實走勢關鍵節點插值生成）")
    print(f"      策略：MA{SHORT_MA} > MA{LONG_MA} 時滿倉持有，否則空倉\n")


if __name__ == "__main__":
    print("正在生成美股歷史近似資料並執行回測...\n")

    stock_data   = build_all_data()
    all_results  = []

    for ticker, series in stock_data.items():
        print(f"回測 {ticker}...", end=" ")
        df  = compute_signals(series)
        res = backtest(df, ticker)
        all_results.append(res)
        print(f"完成 ({len(df)} 個交易日)")

    print_summary(all_results)
    plot_results(all_results)
