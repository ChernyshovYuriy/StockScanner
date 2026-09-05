"""
Standalone research experiment (not wired into any paper-trading sleeve):
does a simple fixed-% "buy the dip, sell the rip" grid strategy have any edge
on large, liquid, obviously-good-quality US names (AAPL, NVDA, AMD)?

Rule set, as specified 2026-09-05:
  - While FLAT: track the running intraday high since the last sale (or since
    the start of the window, for the very first entry). Buy when price drops
    `dip_pct` below that high-water mark.
  - While LONG: sell at -`stop_pct` (loss) or +`target_pct` (gain) from the
    entry price, whichever is hit first (checked on bar closes).
  - After a sell, the high-water mark resets and starts tracking again from
    that bar forward (i.e. "wait until it dips enough again").

Data: yfinance hourly bars ("60m"), ~730 days back -- the longest intraday
history yfinance hands out for free. 5-minute bars only go back ~60 days,
which is too short a window to draw any conclusion from, so this script does
not use them.

This is a research tool, not a production sleeve: no DB writes, no live
trading, just console output. Run standalone:

    python dip_grid_backtest.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

TICKERS = ["AAPL", "NVDA", "AMD"]
INTERVAL = "60m"
PERIOD = "730d"

DIP_PCTS = [0.02, 0.03, 0.05]         # "dipped enough to buy" thresholds to sweep
STOP_PCT = 0.01                        # fixed per spec: sell at -1%
TARGET_PCTS = [0.05, 0.075, 0.10]      # sell at +5% to +10%, swept across the range
ROUND_TRIP_COST_BPS = 5                # crude commission+slippage assumption per round trip


def fetch_bars(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def simulate(closes: np.ndarray, dip_pct: float, target_pct: float,
             stop_pct: float = STOP_PCT, cost_bps: float = ROUND_TRIP_COST_BPS,
             adx: np.ndarray = None, adx_threshold: float = None):
    """Bar-close based simulation. Returns (equity_curve, trades).

    `adx`/`adx_threshold` are an optional regime gate: a new entry is only
    taken when adx[i] <= adx_threshold (i.e. skip entries while the ticker is
    trending, per the 2026-09-05 finding that this dip-grid rule loses to
    buy-and-hold on trending names but beats it on range-bound ones). Exits
    are never gated -- an open position still manages its own stop/target
    regardless of the regime reading. Passing neither argument (the default)
    reproduces the original, ungated behavior exactly.
    """
    n = len(closes)
    cost = cost_bps / 10_000

    state = "FLAT"
    high_water = closes[0]
    entry_price = None
    equity = 1.0  # realized equity multiplier, compounded at each exit
    equity_curve = np.empty(n)
    trades = []

    for i, px in enumerate(closes):
        if state == "FLAT":
            high_water = max(high_water, px)
            regime_ok = (adx is None or adx_threshold is None
                         or (not np.isnan(adx[i]) and adx[i] <= adx_threshold))
            if px <= high_water * (1 - dip_pct) and regime_ok:
                state = "LONG"
                entry_price = px
            equity_curve[i] = equity
        else:  # LONG
            change = (px - entry_price) / entry_price
            equity_curve[i] = equity * (1 + change)  # mark-to-market unrealized
            if change <= -stop_pct or change >= target_pct:
                realized = change - cost
                equity *= (1 + realized)
                trades.append({"entry_px": entry_price, "exit_px": px,
                                "return": realized, "win": realized > 0})
                state = "FLAT"
                high_water = px
                equity_curve[i] = equity

    return equity_curve, trades


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return dd.min()


def summarize(ticker: str, bars: pd.DataFrame, dip_pct: float, target_pct: float) -> dict:
    closes = bars["Close"].to_numpy()
    equity_curve, trades = simulate(closes, dip_pct, target_pct)
    total_return = equity_curve[-1] - 1
    buy_hold = closes[-1] / closes[0] - 1
    wins = [t for t in trades if t["win"]]
    return {
        "ticker": ticker,
        "dip_pct": dip_pct,
        "target_pct": target_pct,
        "trades": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else float("nan"),
        "avg_trade_return": (np.mean([t["return"] for t in trades]) if trades else float("nan")),
        "total_return": total_return,
        "buy_hold_return": buy_hold,
        "max_drawdown": max_drawdown(equity_curve),
    }


def main():
    rows = []
    for ticker in TICKERS:
        bars = fetch_bars(ticker)
        if bars.empty:
            print(f"WARNING: no data returned for {ticker}, skipping")
            continue
        print(f"{ticker}: {len(bars)} hourly bars, {bars.index[0].date()} -> {bars.index[-1].date()}")
        for dip_pct in DIP_PCTS:
            for target_pct in TARGET_PCTS:
                rows.append(summarize(ticker, bars, dip_pct, target_pct))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results -- data fetch failed for every ticker.")
        return

    df = df.sort_values(["ticker", "total_return"], ascending=[True, False])
    df_display = df.copy()
    for col in ["dip_pct", "target_pct", "win_rate", "avg_trade_return", "total_return",
                "buy_hold_return", "max_drawdown"]:
        df_display[col] = df_display[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")

    print()
    print(tabulate(df_display, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
