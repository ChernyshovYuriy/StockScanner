"""
Standalone research tool (not wired into any paper-trading sleeve): walk-forward
sweep of research/elder_ray.py's EMA parameters (--ema / --weekly-ema), to check
whether Elder's book default (13/13) is actually the best choice for a swing
trade, or just the textbook-conventional one. Answers the question raised after
building elder_ray.py: "correct" (matches the book) and "optimal" (best forward
returns) are different claims, and this is what actually tests the second one.

Reuses elder_ray.py's own compute_elder_ray()/classify_row()/trend_at()/
weekly_trend_by_date() unchanged -- this sweeps exactly the two parameters the
live tool exposes, nothing else.

Entry/exit rules (held constant across the whole sweep -- only ema/weekly_ema
vary):
  - Enter LONG at the next bar's open after a close classified "BUY SETUP"
    (weekly-gated) -- signal known at yesterday's close, traded at today's
    open, same next-open-after-close convention as backtest_runner.py.
  - Exit at STOP_PCT below entry or after MAX_HOLD_DAYS, whichever comes
    first, checked on bar closes (dip_grid_backtest.py's own convention).
  - Single position at a time. No short side (Elder-Ray's SELL SETUP here is
    read as "avoid/exit", not "go short" -- matches this repo's long-only
    stance elsewhere, e.g. the macro sleeve).

Walk-forward methodology (mirrors walk_forward_ab.py): each ticker's daily
Elder-Ray signal is computed once over the *full* fetched history (so the EMA
is fully warmed up going into the earliest fold), then sliced into one fold
per calendar year. Each fold starts fresh at equity=1.0 -- no cross-fold
compounding -- so a lucky/unlucky year can't hide inside a multi-year curve.
Per (ema, weekly_ema) combo: win rate vs. buy-and-hold across all ticker-year
folds, plus a paired t-test on the return differences (same style as the
gap-filter and sector-cap walk-forwards -- see CLAUDE.md / memory).

Caveat, stated plainly: 3 tickers x ~10 years is a small sample for a t-test,
same order of rigor as dip_grid_backtest.py's own basket, not a large-N study.
Treat a marginal p-value here as a lead, not a conclusion.

This is a research tool: no DB writes, no live trading, just console output.
Run standalone (edit the constants below to change the sweep grid or
tickers -- in-code constants, not CLI flags, matching walk_forward_ab.py's
own convention for this kind of one-off analysis):

    python research/elder_ray_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

# Repo root isn't on sys.path when this is run directly (`python
# research/elder_ray_backtest.py` puts only research/ itself on sys.path) —
# added so market_data's shared Yahoo Finance fetch can be reused here
# instead of a local yfinance call, same reasoning as elder_ray.py's own
# bootstrap.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from market_data import DEFAULT_PROVIDER  # noqa: E402

try:
    # documented usage: run directly (`python research/elder_ray_backtest.py`),
    # which puts research/ itself on sys.path
    from elder_ray import (
        TREND_LAG, WEEKLY_TREND_LAG, classify_row, compute_elder_ray, trend_at, weekly_trend_by_date,
    )
except ImportError:
    # fallback for import as a package member (e.g. `import research.elder_ray_backtest`)
    from research.elder_ray import (
        TREND_LAG, WEEKLY_TREND_LAG, classify_row, compute_elder_ray, trend_at, weekly_trend_by_date,
    )

TICKERS = ["AAPL", "NVDA", "AMD"]   # same basket as dip_grid_backtest.py: liquid, long-history large caps
DAILY_PERIOD = "12y"                 # ~10 fold years + ~2y EMA warmup buffer before the first fold
WEEKLY_PERIOD = "14y"                 # same idea at weekly scale
FOLD_YEARS = 10                        # only the most recent N calendar years are traded; earlier data is warmup-only
MIN_FOLD_TRADING_DAYS = 150             # drop a partial year (e.g. the current in-progress one)

EMA_CANDIDATES = [8, 13, 21]              # daily EMA sweep; 13 = Elder's book default / elder_ray.py's current default
WEEKLY_EMA_CANDIDATES = [8, 13, 21]        # weekly EMA sweep; same
BASELINE = (13, 13)                         # Elder's own textbook combo, called out explicitly in the output

STOP_PCT = 0.08            # fixed 8% stop-loss, held constant across the sweep
MAX_HOLD_DAYS = 20          # ~1 trading month time-stop, held constant across the sweep
ROUND_TRIP_COST_BPS = 5      # crude commission+slippage assumption, matches dip_grid_backtest.py


def fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return DEFAULT_PROVIDER.download_bars(ticker, period=period, interval=interval)


def compute_reads(daily_bars: pd.DataFrame, weekly_bars: pd.DataFrame, ema: int, weekly_ema: int):
    """Full-history Elder-Ray reads for one (ema, weekly_ema) combo -- the exact
    same per-row logic elder_ray.py's CLI prints, just run over the whole
    fetched range instead of a --lookback-limited tail.
    """
    df = compute_elder_ray(daily_bars, ema)
    weekly_df = compute_elder_ray(weekly_bars, weekly_ema)
    weekly_trend = weekly_trend_by_date(df.index, weekly_df, WEEKLY_TREND_LAG)

    reads = []
    for i in range(len(df)):
        trend = trend_at(df, i, TREND_LAG)
        prev_row = df.iloc[i - 1] if i > 0 else None
        reads.append(classify_row(df.iloc[i], prev_row, trend, weekly_trend.iloc[i]))
    return df, reads


def year_folds(daily_bars: pd.DataFrame, max_folds: int, min_trading_days: int):
    """(year, start_idx, end_idx) triples for the most recent `max_folds` full
    calendar years present in `daily_bars` -- earlier years are left as
    EMA-warmup only, never traded.
    """
    folds = []
    for year in sorted(daily_bars.index.year.unique()):
        idxs = np.where(daily_bars.index.year == year)[0]
        if len(idxs) < min_trading_days:
            continue
        folds.append((int(year), int(idxs[0]), int(idxs[-1]) + 1))
    return folds[-max_folds:]


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    return ((equity_curve - peak) / peak).min()


def simulate_fold(opens: np.ndarray, closes: np.ndarray, reads: list, start_idx: int, end_idx: int,
                   stop_pct: float = STOP_PCT, max_hold: int = MAX_HOLD_DAYS,
                   cost_bps: float = ROUND_TRIP_COST_BPS):
    """One ticker-year fold, equity reset to 1.0 at fold start (no cross-fold
    compounding). Returns (equity_curve, n_trades).
    """
    cost = cost_bps / 10_000
    n = end_idx - start_idx
    equity_curve = np.empty(n)
    equity = 1.0               # value locked in as of the last CLOSED trade
    entry_price = None
    entry_day = None
    equity_at_entry = None
    n_trades = 0

    for offset in range(n):
        i = start_idx + offset
        if entry_price is None:
            if i > 0 and reads[i - 1].startswith("BUY SETUP"):
                entry_price = opens[i]
                entry_day = i
                equity_at_entry = equity
                n_trades += 1
            equity_curve[offset] = equity
            continue

        held = i - entry_day
        px = closes[i]
        if px <= entry_price * (1 - stop_pct) or held >= max_hold:
            trade_return = px / entry_price - 1 - cost
            equity = equity_at_entry * (1 + trade_return)
            entry_price = None
            equity_curve[offset] = equity
        else:
            equity_curve[offset] = equity_at_entry * (px / entry_price)   # mark-to-market, unrealized

    if entry_price is not None:
        # fold ends mid-trade: mark it closed at the fold's last close for measurement purposes
        trade_return = closes[end_idx - 1] / entry_price - 1 - cost
        equity_curve[-1] = equity_at_entry * (1 + trade_return)

    return equity_curve, n_trades


def main():
    per_ticker = {}
    for ticker in TICKERS:
        daily_bars = fetch(ticker, DAILY_PERIOD, "1d")
        weekly_bars = fetch(ticker, WEEKLY_PERIOD, "1wk")
        if daily_bars.empty or weekly_bars.empty:
            print(f"WARNING: no data for {ticker}, skipping")
            continue
        folds = year_folds(daily_bars, FOLD_YEARS, MIN_FOLD_TRADING_DAYS)
        if not folds:
            print(f"WARNING: not enough full-year history for {ticker}, skipping")
            continue
        per_ticker[ticker] = (daily_bars, weekly_bars, folds)
        print(f"{ticker}: {len(daily_bars)} daily bars, folds = "
              f"{[y for y, _, _ in folds]}")

    rows = []
    for ticker, (daily_bars, weekly_bars, folds) in per_ticker.items():
        for ema in EMA_CANDIDATES:
            for weekly_ema in WEEKLY_EMA_CANDIDATES:
                df, reads = compute_reads(daily_bars, weekly_bars, ema, weekly_ema)
                opens = df["Open"].to_numpy()
                closes = df["Close"].to_numpy()
                for year, start_idx, end_idx in folds:
                    equity_curve, n_trades = simulate_fold(opens, closes, reads, start_idx, end_idx)
                    strategy_return = equity_curve[-1] - 1
                    buyhold_return = closes[end_idx - 1] / closes[start_idx] - 1
                    rows.append({
                        "ticker": ticker, "year": year, "ema": ema, "weekly_ema": weekly_ema,
                        "strategy_return": strategy_return, "buyhold_return": buyhold_return,
                        "trades": n_trades, "max_dd": max_drawdown(equity_curve),
                    })

    results = pd.DataFrame(rows)
    if results.empty:
        print("No results -- data fetch failed for every ticker.")
        return

    summary_rows = []
    for (ema, weekly_ema), group in results.groupby(["ema", "weekly_ema"]):
        diffs = (group["strategy_return"] - group["buyhold_return"]).to_numpy()
        if len(diffs) > 1 and diffs.std() > 0:
            tstat, pval = stats.ttest_1samp(diffs, popmean=0)
        else:
            tstat, pval = float("nan"), float("nan")
        summary_rows.append({
            "ema": ema, "weekly_ema": weekly_ema,
            "folds": len(group),
            "trades": int(group["trades"].sum()),
            "win_rate": (group["strategy_return"] > group["buyhold_return"]).mean(),
            "mean_return": group["strategy_return"].mean(),
            "mean_buyhold": group["buyhold_return"].mean(),
            "mean_excess": diffs.mean(),
            "mean_max_dd": group["max_dd"].mean(),
            "t_stat": tstat,
            "p_value": pval,
            "baseline": "<-- Elder's book default" if (ema, weekly_ema) == BASELINE else "",
        })

    summary = pd.DataFrame(summary_rows).sort_values(["win_rate", "p_value"], ascending=[False, True])
    display = summary.copy()
    for col in ["win_rate", "mean_return", "mean_buyhold", "mean_excess", "mean_max_dd"]:
        display[col] = display[col].map(lambda x: f"{x:.2%}")
    display["p_value"] = display["p_value"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    display["t_stat"] = display["t_stat"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")

    buyhold_by_ticker = (results.drop_duplicates(["ticker", "year"])
                         .groupby("ticker")["buyhold_return"].mean())
    print()
    print("Buy-and-hold mean annual return by ticker over the fold years (context for the "
          "'mean_buyhold' column below -- checks whether one name is driving the average):")
    for ticker, ret in buyhold_by_ticker.items():
        print(f"  {ticker}: {ret:+.1%}")

    print()
    print(f"{len(results)} ticker-year folds across {len(per_ticker)} tickers "
          f"x {len(EMA_CANDIDATES) * len(WEEKLY_EMA_CANDIDATES)} EMA combos")
    print(f"Fixed exit rule across every combo: stop={STOP_PCT:.0%}, time-stop={MAX_HOLD_DAYS} bars, "
          f"cost={ROUND_TRIP_COST_BPS}bps/round-trip")
    print()
    print(tabulate(display, headers="keys", tablefmt="github", showindex=False))

    baseline_row = summary[(summary["ema"] == BASELINE[0]) & (summary["weekly_ema"] == BASELINE[1])]
    best_row = summary.iloc[0]
    print()
    if not baseline_row.empty:
        b = baseline_row.iloc[0]
        print(f"Baseline (ema={BASELINE[0]}, weekly_ema={BASELINE[1]}): "
              f"win_rate={b['win_rate']:.0%}, mean_excess={b['mean_excess']:+.2%}, p={b['p_value']:.3f}")
    print(f"Best by win_rate: ema={int(best_row['ema'])}, weekly_ema={int(best_row['weekly_ema'])}, "
          f"win_rate={best_row['win_rate']:.0%}, mean_excess={best_row['mean_excess']:+.2%}, "
          f"p={best_row['p_value']:.3f}")
    if (best_row["p_value"] < 0.05 and best_row["mean_excess"] > 0
            and (int(best_row["ema"]), int(best_row["weekly_ema"])) != BASELINE):
        print("-> statistically significant POSITIVE edge over the book default at this sample "
              "size -- still just 3 tickers, treat as a lead, not a conclusion.")
    elif not baseline_row.empty and baseline_row.iloc[0]["mean_excess"] < 0 and baseline_row.iloc[0]["p_value"] < 0.05:
        print("-> every combo (including the book default) significantly UNDERperforms "
              "buy-and-hold on this basket/period -- see caveat below before reading this as "
              "'the EMA choice doesn't matter': it means the exit rule is giving back the "
              "basket's trend-following gains, not that the sweep found a winner.")
    else:
        print("-> no statistically significant edge found over Elder's book default (13/13); "
              "keep the textbook parameters.")


if __name__ == "__main__":
    main()
