"""
run_backtest.py
===============
Command-line entry point for the backtest system.

Single run:
    python run_backtest.py --start 2022-01-01 --end 2024-01-01

Custom tickers file:
    python run_backtest.py --tickers data/can_tickers --start 2022-01-01 --end 2024-01-01

Parameter sweep (test multiple risk_pct × top_n_buys combinations):
    python run_backtest.py --start 2022-01-01 --end 2024-01-01 --sweep

All options:
    python run_backtest.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── project imports ───────────────────────────────────────────────────────────
from backtest_report import write_backtest_report
from backtest_runner import BacktestConfig, BacktestResults, BacktestRunner
from position_monitor import ExitParams
from config import CAN_TICKERS_PATH, OUT_PATH
from market_data import HistoricalSliceProvider


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_tickers(path: str) -> list[str]:
    """Read tickers from a one-per-line text file, strip blanks and comments."""
    with open(path, "r", encoding="utf-8") as f:
        return [
            ln.strip() for ln in f
            if ln.strip() and not ln.strip().startswith("#")
        ]


def _fetch_provider(
    tickers: list[str],
    benchmark: str,
    start_date: str,
    lookback_days: int,
    end_date: str,
) -> tuple[HistoricalSliceProvider, pd.Series | None]:
    """
    Pre-load all historical data from Yahoo Finance in one pass.

    Returns (provider, benchmark_close_series).
    benchmark_close_series is None if the fetch fails (report skips overlay).
    """
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))  # deduplicated

    # Start far enough back to have lookback_days bars at start_date
    from backtest_runner import _lookback_start
    fetch_start = _lookback_start(start_date, lookback_days)

    print(f"\n  Fetching {len(all_tickers)} tickers  "
          f"{fetch_start} → {end_date}  …", end="", flush=True)

    t0 = time.perf_counter()
    provider = HistoricalSliceProvider.from_yfinance(
        tickers=all_tickers,
        start=fetch_start,
        end=end_date,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s  ({len(provider)} tickers loaded)")

    # Extract benchmark close series for report overlay
    bench_series: pd.Series | None = None
    try:
        bench_df = provider._data.get(benchmark)
        if bench_df is not None and not bench_df.empty:
            # Restrict to the simulation window
            mask = (bench_df.index >= pd.Timestamp(start_date)) & \
                   (bench_df.index <= pd.Timestamp(end_date))
            bench_series = bench_df.loc[mask, "Close"]
    except Exception:
        pass

    return provider, bench_series


def _run_single(
    args: argparse.Namespace,
    tickers: list[str],
    provider: HistoricalSliceProvider,
    bench_series: pd.Series | None,
    suffix: str = "",
) -> BacktestResults:
    """Run one backtest with the given args and write its report."""

    ep = ExitParams(
        initial_stop_atr_k   = args.stop_atr,
        chand_trail_atr_k    = args.trail_atr,
        time_stop_days       = args.time_stop_days,
        time_stop_min_profit = args.time_stop_pct,
        stop_trigger         = args.stop_trigger,
    )

    cfg = BacktestConfig(
        tickers            = tickers,
        benchmark          = args.benchmark,
        start_date         = args.start,
        end_date           = args.end,
        initial_cash       = args.capital,
        risk_pct           = args.risk,
        top_n_buys         = args.top_n,
        min_score          = args.min_score,
        lookback_days      = args.lookback,
        screener_frequency = args.screener_freq,
        min_rr             = args.min_rr,
        atr_stop_mult      = args.atr_mult,
        exit_params        = ep,
        _provider          = provider,
    )

    print(f"\n{'─'*60}")
    print(f"  Running backtest{suffix}")
    print(f"  Period   : {args.start} → {args.end}")
    print(f"  Capital  : ${args.capital:,.0f}   risk={args.risk}%   "
          f"top_n={args.top_n}   min_score={args.min_score}")
    print(f"  Tickers  : {len(tickers) - 1} + benchmark")
    print(f"  Exit     : {ep.summary()}")
    print(f"{'─'*60}\n")

    results = BacktestRunner(cfg).run(verbose=not args.quiet)
    print(results.summary())

    # Write HTML report
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%dT%H%M")
    filename = f"backtest_{args.start}_{args.end}{suffix}_{ts}.html"
    out_path = OUT_PATH / filename
    write_backtest_report(results, str(out_path), benchmark_equity=bench_series)
    print(f"\n  Report → {out_path.resolve()}")

    # Also write CSVs
    eq_path = OUT_PATH / f"backtest_equity{suffix}_{ts}.csv"
    tl_path = OUT_PATH / f"backtest_trades{suffix}_{ts}.csv"
    results.equity_curve_df().to_csv(eq_path, index=False)
    results.trade_log_df().to_csv(tl_path, index=False)
    print(f"  Equity  → {eq_path.resolve()}")
    print(f"  Trades  → {tl_path.resolve()}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def _run_sweep(
    args: argparse.Namespace,
    tickers: list[str],
    provider: HistoricalSliceProvider,
    bench_series: pd.Series | None,
) -> None:
    """
    Run a parameter grid over risk_pct × top_n_buys and print a summary table.

    Screener scores are pre-computed ONCE across all trading days, then injected
    into every combination.  This reduces sweep time from O(combos × days) to
    O(days + combos × days_monitor_only) — roughly 16× faster for a 4×4 grid.
    """
    import itertools
    from backtest_runner import _run_screener_step, _trading_days, BacktestConfig
    from time_utils import set_backtest_clock, TSX_TZ
    from datetime import datetime as _dt

    risk_vals  = [0.5, 1.0, 1.5, 2.0]
    top_n_vals = [1, 2, 3, 5]

    combos  = list(itertools.product(risk_vals, top_n_vals))
    summary = []

    # ── Pre-compute screener results once for all trading days ────────────────
    # Scores depend only on price data, not on risk_pct or top_n_buys.
    base_ep = ExitParams(
        initial_stop_atr_k   = args.stop_atr,
        chand_trail_atr_k    = args.trail_atr,
        time_stop_days       = args.time_stop_days,
        time_stop_min_profit = args.time_stop_pct,
        stop_trigger         = args.stop_trigger,
    )
    base_cfg = BacktestConfig(
        tickers            = tickers,
        benchmark          = args.benchmark,
        start_date         = args.start,
        end_date           = args.end,
        initial_cash       = args.capital,
        risk_pct           = 1.0,
        top_n_buys         = 3,
        min_score          = args.min_score,
        lookback_days      = args.lookback,
        screener_frequency = args.screener_freq,
        min_rr             = args.min_rr,
        exit_params        = base_ep,
        _provider          = provider,
    )

    trading_days = _trading_days(args.start, args.end)
    freq = args.screener_freq
    screener_cache: dict = {}

    print(f"\n  Pre-computing screener scores  ", end="", flush=True)
    t_pre = time.perf_counter()
    for i, day in enumerate(trading_days):
        if i % freq == 0:
            sim_ts = pd.Timestamp(day)
            set_backtest_clock(_dt(day.year, day.month, day.day, 16, 5, tzinfo=TSX_TZ))
            screener_cache[i] = _run_screener_step(base_cfg, provider, sim_ts)
            print(".", end="", flush=True)
    set_backtest_clock(None)
    print(f"  done in {time.perf_counter()-t_pre:.1f}s  "
          f"({len(screener_cache)} score snapshots)")

    print(f"\n  Parameter sweep: {len(combos)} combinations  "
          f"(risk_pct × top_n_buys)\n")

    for k, (risk, top_n) in enumerate(combos, 1):
        args.risk  = risk
        args.top_n = top_n

        ep_sweep = ExitParams(
            initial_stop_atr_k   = args.stop_atr,
            chand_trail_atr_k    = args.trail_atr,
            time_stop_days       = args.time_stop_days,
            time_stop_min_profit = args.time_stop_pct,
            stop_trigger         = args.stop_trigger,
        )

        cfg = BacktestConfig(
            tickers            = tickers,
            benchmark          = args.benchmark,
            start_date         = args.start,
            end_date           = args.end,
            initial_cash       = args.capital,
            risk_pct           = risk,
            top_n_buys         = top_n,
            min_score          = args.min_score,
            lookback_days      = args.lookback,
            screener_frequency = args.screener_freq,
            min_rr             = args.min_rr,
            atr_stop_mult      = args.atr_mult,
            exit_params        = ep_sweep,
            _provider          = provider,
            _screener_cache    = screener_cache,
        )

        print(f"  [{k:>2}/{len(combos)}] risk={risk}%  top_n={top_n}  … ",
              end="", flush=True)
        t0 = time.perf_counter()
        results = BacktestRunner(cfg).run(verbose=False)
        elapsed = time.perf_counter() - t0

        eq = results.equity_curve_df()
        tl = results.trade_log_df()

        end_eq   = float(eq["total_equity"].iloc[-1]) if not eq.empty else args.capital
        ret_pct  = (end_eq / args.capital - 1) * 100

        equity   = eq["total_equity"]
        roll_max = equity.cummax()
        dd       = (equity - roll_max) / roll_max * 100
        max_dd   = float(dd.min()) if not eq.empty else 0.0

        n_trades = len(tl)
        wins     = tl[tl["pnl"] > 0] if n_trades > 0 else tl
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0.0

        gp = float(wins["pnl"].sum()) if not wins.empty else 0.0
        gl = abs(float(tl[tl["pnl"] <= 0]["pnl"].sum())) \
             if n_trades > 0 and len(tl[tl["pnl"] <= 0]) > 0 else 0.0
        pf = gp / gl if gl > 0 else 0.0

        # Annualised Sharpe from daily equity returns
        daily_ret = equity.pct_change().dropna()
        sharpe    = 0.0
        if len(daily_ret) > 10 and daily_ret.std() > 0:
            sharpe = float(daily_ret.mean() / daily_ret.std() * (252 ** 0.5))

        summary.append({
            "risk_%":    risk,
            "top_n":     top_n,
            "ret_%":     round(ret_pct, 2),
            "max_dd_%":  round(max_dd, 2),
            "sharpe":    round(sharpe, 2),
            "trades":    n_trades,
            "win_rate%": round(win_rate, 1),
            "pf":        round(pf, 2),
        })
        sign = "+" if ret_pct >= 0 else ""
        print(f"ret={sign}{ret_pct:.1f}%  dd={max_dd:.1f}%  "
              f"sharpe={sharpe:.2f}  trades={n_trades}  ({elapsed:.1f}s)")

    # Print ranked table
    df_sw = pd.DataFrame(summary).sort_values("sharpe", ascending=False)
    print(f"\n{'─'*70}")
    print("  Sweep results ranked by Sharpe ratio")
    print(f"{'─'*70}")
    print(df_sw.to_string(index=False))

    # Save sweep CSV
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%dT%H%M")
    sw_path  = OUT_PATH / f"backtest_sweep_{args.start}_{args.end}_{ts}.csv"
    df_sw.to_csv(sw_path, index=False)
    print(f"\n  Sweep results → {sw_path.resolve()}")

    # Run full report for the best Sharpe combo
    best = df_sw.iloc[0]
    print(f"\n  Best combo: risk={best['risk_%']}%  top_n={int(best['top_n'])}  "
          f"(Sharpe={best['sharpe']})\n  Generating full report…")
    args.risk  = float(best["risk_%"])
    args.top_n = int(best["top_n"])
    _run_single(args, tickers, provider, bench_series,
                suffix=f"_best_r{best['risk_%']}_n{int(best['top_n'])}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a historical backtest of the TSX swing-trading system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Date range
    p.add_argument("--start",    default="2022-01-01",
                   help="Backtest start date (YYYY-MM-DD, inclusive)")
    p.add_argument("--end",      default="2024-01-01",
                   help="Backtest end date   (YYYY-MM-DD, exclusive)")

    # Universe
    p.add_argument("--tickers",  default=str(CAN_TICKERS_PATH),
                   help="Path to tickers file (one ticker per line)")
    p.add_argument("--benchmark", default="XIU.TO",
                   help="Benchmark ETF ticker")

    # Capital
    p.add_argument("--capital",  type=float, default=100_000.0,
                   help="Starting capital (CAD)")
    p.add_argument("--risk",     type=float, default=1.0,
                   help="Risk per trade as %% of account")

    # Strategy knobs
    p.add_argument("--top-n",   dest="top_n", type=int, default=3,
                   help="Max concurrent positions opened per day")
    p.add_argument("--min-score", dest="min_score", type=float, default=0.0,
                   help="Min composite screener score (0=off, 55=large universes)")
    p.add_argument("--lookback", type=int, default=504,
                   help="Screener lookback window in calendar days")
    p.add_argument("--screener-freq", dest="screener_freq", type=int, default=5,
                   help="Run full screener every N days (1=daily, 5=weekly)")
    p.add_argument("--min-rr",   dest="min_rr", type=float, default=2.0,
                   help="Minimum risk:reward to accept a signal")
    p.add_argument("--atr-mult", dest="atr_mult", type=float, default=1.5,
                   help="ATR multiplier for initial stop loss")

    # Exit rule tuning (override position_monitor.py defaults)
    p.add_argument("--stop-atr",       dest="stop_atr",       type=float, default=1.5,
                   help="Initial stop = entry - N × ATR14  (default 1.5)")
    p.add_argument("--trail-atr",      dest="trail_atr",      type=float, default=2.5,
                   help="Chandelier trail = highest_high - N × ATR14  (default 2.5)")
    p.add_argument("--time-stop-days", dest="time_stop_days", type=int,   default=14,
                   help="Exit if no profit after N trading days  (default 14, was 7)")
    p.add_argument("--time-stop-pct",  dest="time_stop_pct",  type=float, default=0.5,
                   help="Min profit %% required by time-stop day  (default 0.5)")
    p.add_argument("--stop-trigger",   dest="stop_trigger",   default="close",
                   choices=["low", "close"],
                   help="Stop trigger: 'low' (intraday) or 'close' (EOD)  (default close)")

    # Modes
    p.add_argument("--sweep",  action="store_true",
                   help="Run parameter sweep over risk_pct × top_n_buys grid")
    p.add_argument("--quiet",  action="store_true",
                   help="Suppress per-day progress output")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ── Load tickers ─────────────────────────────────────────────────────────
    try:
        tickers = _load_tickers(args.tickers)
    except FileNotFoundError:
        print(f"ERROR: tickers file not found: {args.tickers}", file=sys.stderr)
        sys.exit(1)

    if not tickers:
        print(f"ERROR: no tickers found in {args.tickers}", file=sys.stderr)
        sys.exit(1)

    # Ensure benchmark is included
    if args.benchmark not in tickers:
        tickers = tickers + [args.benchmark]

    print(f"\n{'='*60}")
    print(f"  TSX Backtest  {args.start} → {args.end}")
    print(f"  Universe: {len(tickers) - 1} tickers + {args.benchmark}")
    print(f"{'='*60}")

    # ── Pre-load data once ───────────────────────────────────────────────────
    provider, bench_series = _fetch_provider(
        tickers      = tickers,
        benchmark    = args.benchmark,
        start_date   = args.start,
        lookback_days= args.lookback,
        end_date     = args.end,
    )

    if len(provider) == 0:
        print("\nERROR: no data loaded. Check your internet connection "
              "and ticker symbols.", file=sys.stderr)
        sys.exit(1)

    # ── Run ──────────────────────────────────────────────────────────────────
    t_total = time.perf_counter()

    if args.sweep:
        _run_sweep(args, tickers, provider, bench_series)
    else:
        _run_single(args, tickers, provider, bench_series)

    print(f"\n  Total wall time: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
