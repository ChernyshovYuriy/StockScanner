"""
run_backtest.py
===============
Command-line entry point for the backtest system.

Single run:
    python run_backtest.py --start 2022-01-01 --end 2024-01-01

Custom tickers source (URL or file):
    python run_backtest.py --tickers https://example.com/tickers.txt --start 2022-01-01 --end 2024-01-01

Parameter sweep (exit params — time_stop_days × stop_atr, 4×4=16 combos):
    python run_backtest.py --start 2022-01-01 --end 2024-01-01 --sweep

Walk-forward gap filter optimization (find optimal GAP_FILTER_PCT):
    python run_backtest.py --start 2022-01-01 --end 2024-01-01 --walk-forward-gap
    python run_backtest.py --start 2022-01-01 --end 2024-01-01 --walk-forward-gap --wf-in-days 126 --wf-out-days 42

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

# ── project imports ───────────────────────────────────────────────────────────
from backtest_report import write_backtest_report
from backtest_runner import BacktestConfig, BacktestResults, BacktestRunner
from position_monitor import ExitParams
from config import CAN_TICKERS_URL, OUT_PATH
from market_data import HistoricalSliceProvider


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_tickers(source: str) -> list[str]:
    """Read tickers from a URL or local file (one per line), strip blanks and comments."""
    import urllib.request
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            content = resp.read().decode("utf-8")
        return [
            ln.strip() for ln in content.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    with open(source, "r", encoding="utf-8") as f:
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
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[HistoricalSliceProvider, pd.Series | None]:
    """
    Pre-load all historical data in one pass — from the local DuckDB market
    data cache by default (only missing tickers/date-ranges are fetched from
    Yahoo Finance), or a fresh Yahoo Finance download when use_cache=False.

    Returns (provider, benchmark_close_series).
    benchmark_close_series is None if the fetch fails (report skips overlay).
    """
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))  # deduplicated

    # Start far enough back to have lookback_days bars at start_date
    from backtest_runner import _lookback_start
    fetch_start = _lookback_start(start_date, lookback_days)

    cache_note = "  (cache)" if use_cache else ""
    print(f"\n  Fetching {len(all_tickers)} tickers  "
          f"{fetch_start} → {end_date}{cache_note}  …", end="", flush=True)

    t0 = time.perf_counter()
    if use_cache:
        provider = HistoricalSliceProvider.from_cache(
            tickers=all_tickers,
            start=fetch_start,
            end=end_date,
            force_refresh=refresh_cache,
        )
    else:
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
        max_tracked_tickers= args.max_tickers,
        regime_filter      = args.regime_filter,
        exit_params        = ep,
        _provider          = provider,
        gap_filter_pct     = args.gap_filter,
        max_positions      = args.max_positions,
        sizing             = args.sizing,
        sizing_basis       = args.sizing_basis,
    )

    print(f"\n{'─'*60}")
    print(f"  Running backtest{suffix}")
    print(f"  Period   : {args.start} → {args.end}")
    print(f"  Capital  : ${args.capital:,.0f}   risk={args.risk}%   "
          f"top_n={args.top_n}   min_score={args.min_score}")
    print(f"  Tickers  : {len(tickers) - 1} + benchmark")
    print(f"  Exit     : {ep.summary()}")
    regime_str = "ON  (buys blocked when XIU.TO < 200d SMA)" if args.regime_filter else "OFF"
    print(f"  Regime   : {regime_str}")
    max_pos_str = str(args.max_positions) if args.max_positions else "unlimited"
    sizing_str = f"{args.sizing}" + (f" (basis={args.sizing_basis})" if args.sizing == "live" else "")
    gap_str = f"{args.gap_filter}%" if args.gap_filter is not None else "OFF"
    print(f"  Sizing   : {sizing_str}   max_pos={max_pos_str}   gap_filter={gap_str}")
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
    Sweep the two parameters that actually change outcomes:
      time_stop_days × stop_atr  (4 × 4 = 16 combinations)

    risk_pct and top_n_buys are held constant at their CLI values because
    equal-allocation sizing means risk_pct has no effect on share count, and
    top_n=1 (one position at a time) consistently outperforms on this strategy.

    Screener scores are pre-computed once and injected into every combo via
    _screener_cache, so sweep time ≈ pre-compute + 16 × per-combo-monitor-only.
    """
    import itertools
    from backtest_runner import _run_screener_step, _trading_days, BacktestConfig
    from time_utils import set_backtest_clock, TSX_TZ
    from datetime import datetime as _dt

    time_stop_vals = [7, 10, 14, 21]       # trading days
    stop_atr_vals  = [1.5, 2.0, 2.5, 3.0]  # ATR multiples for initial stop

    combos  = list(itertools.product(time_stop_vals, stop_atr_vals))
    summary = []

    # ── Pre-compute screener results once ─────────────────────────────────────
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
        risk_pct           = args.risk,
        top_n_buys         = args.top_n,
        min_score          = args.min_score,
        lookback_days      = args.lookback,
        screener_frequency = args.screener_freq,
        max_tracked_tickers= args.max_tickers,
        regime_filter      = args.regime_filter,
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
          f"(time_stop_days × stop_atr)  "
          f"[top_n={args.top_n}  trigger={args.stop_trigger}  trail={args.trail_atr}×ATR]\n")

    for k, (ts_days, s_atr) in enumerate(combos, 1):
        ep_sweep = ExitParams(
            initial_stop_atr_k   = s_atr,
            chand_trail_atr_k    = args.trail_atr,
            time_stop_days       = ts_days,
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
            max_tracked_tickers= args.max_tickers,
            regime_filter      = args.regime_filter,
            min_rr             = args.min_rr,
            atr_stop_mult      = args.atr_mult,
            exit_params        = ep_sweep,
            _provider          = provider,
            _screener_cache    = screener_cache,
        )

        print(f"  [{k:>2}/{len(combos)}] time_stop={ts_days:>2}d  stop={s_atr}×ATR  … ",
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

        daily_ret = equity.pct_change().dropna()
        sharpe    = 0.0
        if len(daily_ret) > 10 and daily_ret.std() > 0:
            sharpe = float(daily_ret.mean() / daily_ret.std() * (252 ** 0.5))

        # Calmar = annualised return / max drawdown depth
        years  = len(trading_days) / 252
        ann_ret = (end_eq / args.capital) ** (1 / max(years, 0.1)) - 1
        calmar  = round(ann_ret / abs(max_dd / 100), 2) if max_dd < 0 else 0.0

        summary.append({
            "time_stop_d":  ts_days,
            "stop_atr":     s_atr,
            "ret_%":        round(ret_pct, 2),
            "max_dd_%":     round(max_dd, 2),
            "calmar":       calmar,
            "sharpe":       round(sharpe, 2),
            "trades":       n_trades,
            "win_rate%":    round(win_rate, 1),
            "pf":           round(pf, 2),
        })
        sign = "+" if ret_pct >= 0 else ""
        print(f"ret={sign}{ret_pct:.1f}%  dd={max_dd:.1f}%  "
              f"sharpe={sharpe:.2f}  calmar={calmar:.2f}  "
              f"trades={n_trades}  ({elapsed:.1f}s)")

    # ── Print ranked table ────────────────────────────────────────────────────
    df_sw = pd.DataFrame(summary).sort_values("sharpe", ascending=False)
    print(f"\n{'─'*75}")
    print("  Sweep results ranked by Sharpe  "
          f"(period {args.start} → {args.end}  "
          f"top_n={args.top_n}  trigger={args.stop_trigger})")
    print(f"{'─'*75}")
    print(df_sw.to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y%m%dT%H%M")
    sw_path  = OUT_PATH / f"backtest_sweep_{args.start}_{args.end}_{ts_str}.csv"
    df_sw.to_csv(sw_path, index=False)
    print(f"\n  Sweep results → {sw_path.resolve()}")

    # ── Full report for best Sharpe combo ─────────────────────────────────────
    best = df_sw.iloc[0]
    best_ts   = int(best["time_stop_d"])
    best_satr = float(best["stop_atr"])
    print(f"\n  Best: time_stop={best_ts}d  stop={best_satr}×ATR  "
          f"(Sharpe={best['sharpe']})\n  Generating full report…")
    args.time_stop_days = best_ts
    args.stop_atr       = best_satr
    _run_single(args, tickers, provider, bench_series,
                suffix=f"_best_ts{best_ts}_sa{best_satr}")

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD GAP OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _run_walk_forward_gap(
    args: argparse.Namespace,
    tickers: list[str],
    provider: HistoricalSliceProvider,
) -> None:
    """
    Walk-forward optimization for gap_filter_pct.

    Sliding window approach:
      - In-sample (IS): sweep GAP_VALUES, pick the value with best Sharpe.
      - Out-of-sample (OOS): validate the chosen value on unseen data.
      - Slide the window forward by out_days and repeat.

    Screener scores are pre-computed once for the full period and re-indexed
    into each sub-window — no redundant screener computation.
    """
    from backtest_runner import (
        _run_screener_step, _trading_days,
        BacktestConfig, BacktestResults, BacktestRunner,
    )
    from time_utils import set_backtest_clock, TSX_TZ
    from datetime import datetime as _dt

    GAP_VALUES: list = [None, 1.0, 2.0, 3.0, 4.0, 5.0]
    GAP_LABELS: dict = {None: "OFF", 1.0: "1%", 2.0: "2%", 3.0: "3%", 4.0: "4%", 5.0: "5%"}

    in_days  = args.wf_in_days
    out_days = args.wf_out_days
    freq     = args.screener_freq

    trading_days = _trading_days(args.start, args.end)
    n = len(trading_days)

    if n < in_days + out_days:
        print(
            f"\nERROR: period too short for walk-forward "
            f"({n} trading days available, need {in_days + out_days})",
            file=sys.stderr,
        )
        return

    # ── Pre-compute screener scores for the full period once ──────────────────
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
        risk_pct           = args.risk,
        top_n_buys         = args.top_n,
        min_score          = args.min_score,
        lookback_days      = args.lookback,
        screener_frequency = freq,
        max_tracked_tickers= args.max_tickers,
        regime_filter      = args.regime_filter,
        min_rr             = args.min_rr,
        exit_params        = base_ep,
        _provider          = provider,
    )

    print(f"\n  Pre-computing screener scores for full period  ", end="", flush=True)
    t_pre = time.perf_counter()
    full_cache: dict = {}
    for i, day in enumerate(trading_days):
        if i % freq == 0:
            sim_ts = pd.Timestamp(day)
            set_backtest_clock(_dt(day.year, day.month, day.day, 16, 5, tzinfo=TSX_TZ))
            full_cache[i] = _run_screener_step(base_cfg, provider, sim_ts)
            print(".", end="", flush=True)
    set_backtest_clock(None)
    print(f"  done in {time.perf_counter() - t_pre:.1f}s  ({len(full_cache)} snapshots)")

    # ── Build sliding windows ─────────────────────────────────────────────────
    # Each window: (is_start_idx, is_end_idx, oos_end_idx) — all exclusive-right.
    windows: list[tuple[int, int, int]] = []
    s = 0
    while s + in_days + out_days <= n:
        windows.append((s, s + in_days, s + in_days + out_days))
        s += out_days
    n_windows = len(windows)

    print(f"\n  Walk-Forward Gap Optimization")
    print(f"  In-sample: {in_days}d  Out-of-sample: {out_days}d  "
          f"{n_windows} windows  {len(GAP_VALUES)} gap values  "
          f"({n_windows * len(GAP_VALUES)} IS + {n_windows} OOS backtests)\n")

    def _make_sub_cache(offset: int, length: int) -> dict:
        """Re-index full_cache entries into the 0-based day space of a sub-window."""
        sub: dict = {}
        for j in range(length):
            sub_key = (j // freq) * freq
            if sub_key in sub:
                continue
            full_key = ((offset + j) // freq) * freq
            if full_key in full_cache:
                sub[sub_key] = full_cache[full_key]
        return sub

    def _sharpe(r: BacktestResults) -> float:
        eq = r.equity_curve_df()
        if eq.empty or len(eq) < 5:
            return -999.0
        dr = eq["total_equity"].pct_change().dropna()
        return float(dr.mean() / dr.std() * (252 ** 0.5)) if dr.std() > 0 else 0.0

    def _run_sub(start_i: int, end_i: int, gap, sub_cache: dict) -> BacktestResults:
        start_date = str(trading_days[start_i])
        end_date   = str(trading_days[end_i]) if end_i < n else args.end
        cfg = BacktestConfig(
            tickers            = tickers,
            benchmark          = args.benchmark,
            start_date         = start_date,
            end_date           = end_date,
            initial_cash       = args.capital,
            risk_pct           = args.risk,
            top_n_buys         = args.top_n,
            min_score          = args.min_score,
            lookback_days      = args.lookback,
            screener_frequency = freq,
            max_tracked_tickers= args.max_tickers,
            regime_filter      = args.regime_filter,
            min_rr             = args.min_rr,
            atr_stop_mult      = args.atr_mult,
            exit_params        = base_ep,
            gap_filter_pct     = gap,
            max_positions      = args.max_positions,
            sizing             = args.sizing,
            sizing_basis       = args.sizing_basis,
            _provider          = provider,
            _screener_cache    = sub_cache,
        )
        return BacktestRunner(cfg).run(verbose=False)

    summary_rows: list[dict] = []
    gap_wins:  dict = {g: 0 for g in GAP_VALUES}
    oos_stats: dict = {g: [] for g in GAP_VALUES}  # [(sharpe, ret_pct, n_trades)]

    for w_idx, (is_s, is_e, oos_e) in enumerate(windows, 1):
        is_str  = f"{trading_days[is_s]} -> {trading_days[is_e - 1]}"
        oos_end = trading_days[min(oos_e, n) - 1]
        oos_str = f"{trading_days[is_e]} -> {oos_end}"
        print(f"  Window {w_idx}/{n_windows}  IS: {is_str}  OOS: {oos_str}")

        is_cache  = _make_sub_cache(is_s, is_e - is_s)
        oos_cache = _make_sub_cache(is_e, oos_e - is_e)

        # IS sweep — all gap values
        is_results: dict = {}
        for gap in GAP_VALUES:
            r = _run_sub(is_s, is_e, gap, is_cache)
            is_results[gap] = (_sharpe(r), len(r.trades))

        best_gap    = max(is_results, key=lambda g: is_results[g][0])
        best_sharpe = is_results[best_gap][0]

        for gap in GAP_VALUES:
            sh, nt = is_results[gap]
            marker = "  <- best" if gap == best_gap else ""
            print(f"    IS  gap={GAP_LABELS[gap]:>3}  sharpe={sh:>6.2f}  trades={nt}{marker}")

        # OOS validation with the IS winner
        oos_r   = _run_sub(is_e, oos_e, best_gap, oos_cache)
        oos_sh  = _sharpe(oos_r)
        oos_eq  = oos_r.equity_curve_df()
        oos_ret = (
            (float(oos_eq["total_equity"].iloc[-1]) / args.capital - 1) * 100
            if not oos_eq.empty else 0.0
        )
        oos_nt = len(oos_r.trades)

        gap_wins[best_gap] += 1
        oos_stats[best_gap].append((oos_sh, oos_ret, oos_nt))

        warn = "  (low trade count)" if oos_nt < 3 else ""
        print(f"    OOS gap={GAP_LABELS[best_gap]:>3}  sharpe={oos_sh:>6.2f}  "
              f"ret={oos_ret:>+6.1f}%  trades={oos_nt}{warn}\n")

        summary_rows.append({
            "window":     w_idx,
            "IS":         is_str,
            "OOS":        oos_str,
            "best_gap":   GAP_LABELS[best_gap],
            "IS_sharpe":  round(best_sharpe, 2),
            "OOS_sharpe": round(oos_sh, 2),
            "OOS_ret_%":  round(oos_ret, 2),
            "OOS_trades": oos_nt,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    df_sum = pd.DataFrame(summary_rows)
    print(f"{'─' * 70}")
    print("  Walk-Forward Results")
    print(f"{'─' * 70}")
    print(df_sum.to_string(index=False))

    # ── Vote count ────────────────────────────────────────────────────────────
    print(f"\n  IS vote count (how many windows each gap value had best Sharpe):")
    for gap in GAP_VALUES:
        label  = GAP_LABELS[gap]
        count  = gap_wins[gap]
        marker = "  <- current config" if gap == 2.0 else ""
        bar    = "#" * count
        print(f"    {label:>3}  {bar:<{n_windows}}  {count}{marker}")

    # ── Aggregate OOS by winning gap ──────────────────────────────────────────
    print(f"\n  Aggregate OOS performance (windows where each gap value was chosen):")
    for gap in GAP_VALUES:
        rows = oos_stats[gap]
        label = GAP_LABELS[gap]
        marker = "  <- current config" if gap == 2.0 else ""
        if not rows:
            print(f"    {label:>3}  -- (never chosen as IS winner)")
            continue
        avg_sh  = sum(r[0] for r in rows) / len(rows)
        avg_ret = sum(r[1] for r in rows) / len(rows)
        tot_nt  = sum(r[2] for r in rows)
        print(f"    {label:>3}  avg_sharpe={avg_sh:>6.2f}  "
              f"avg_ret={avg_ret:>+6.1f}%  total_trades={tot_nt}{marker}")

    # ── Recommendation ────────────────────────────────────────────────────────
    top = max(gap_wins, key=lambda g: (
        gap_wins[g],
        sum(r[0] for r in oos_stats[g]) / max(len(oos_stats[g]), 1),
    ))
    print(f"\n  Recommendation: GAP_FILTER_PCT = {GAP_LABELS[top]} "
          f"({gap_wins[top]}/{n_windows} IS windows)")
    if gap_wins[top] <= n_windows // 2:
        print("  Note: no single value dominated — consider a longer date range "
              "for a more conclusive result.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    ts_str  = datetime.now().strftime("%Y%m%dT%H%M")
    wf_path = OUT_PATH / f"backtest_wf_gap_{args.start}_{args.end}_{ts_str}.csv"
    df_sum.to_csv(wf_path, index=False)
    print(f"\n  Results -> {wf_path.resolve()}")


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
    p.add_argument("--tickers",  default=CAN_TICKERS_URL,
                   help="URL or file path for the ticker list (one ticker per line)")
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
    p.add_argument("--max-tickers", dest="max_tickers", type=int, default=40,
                   help="Max tickers passed to pattern detection per day (default 40)")
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

    # Live-parity knobs (mirror the live services' rules in simulation)
    p.add_argument("--gap-filter", dest="gap_filter", type=float, default=None,
                   help="Skip buys where open > planned entry + N%% (live GAP_FILTER_PCT is None/off since 2026-07; default off)")
    p.add_argument("--max-positions", dest="max_positions", type=int, default=None,
                   help="Cap concurrent open positions (live MAX_POSITIONS=8; default unlimited)")
    p.add_argument("--sizing", choices=["equal_split", "live"], default="equal_split",
                   help="Buy sizing: equal_split (backtest default) or live (virtual_buy.py formula)")
    p.add_argument("--sizing-basis", dest="sizing_basis", choices=["cash", "equity"], default="cash",
                   help="Base for live sizing: cash (current live behaviour) or equity (proposed fix)")

    # Modes
    p.add_argument("--regime-filter", dest="regime_filter", action="store_true",
                   help="Only open new positions when XIU.TO > 200-day SMA (regime filter)")
    p.add_argument("--sweep",  action="store_true",
                   help="Sweep time_stop_days × stop_atr (4×4=16 combos); ranks by Sharpe")
    p.add_argument("--walk-forward-gap", dest="walk_forward_gap", action="store_true",
                   help="Walk-forward optimization for gap_filter_pct; ranks by OOS Sharpe")
    p.add_argument("--wf-in-days",  dest="wf_in_days",  type=int, default=126,
                   help="Walk-forward in-sample window size in trading days (default 126 ~6 months)")
    p.add_argument("--wf-out-days", dest="wf_out_days", type=int, default=42,
                   help="Walk-forward out-of-sample window size in trading days (default 42 ~2 months)")
    p.add_argument("--quiet",  action="store_true",
                   help="Suppress per-day progress output")

    # Local market data cache (data/market_cache.db)
    p.add_argument("--no-cache", dest="no_cache", action="store_true",
                   help="Skip the local market data cache; fetch fresh from Yahoo Finance every run")
    p.add_argument("--refresh-cache", dest="refresh_cache", action="store_true",
                   help="Ignore existing cache coverage and re-download the full range for every ticker")

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
        use_cache    = not args.no_cache,
        refresh_cache= args.refresh_cache,
    )

    if len(provider) == 0:
        print("\nERROR: no data loaded. Check your internet connection "
              "and ticker symbols.", file=sys.stderr)
        sys.exit(1)

    # ── Run ──────────────────────────────────────────────────────────────────
    t_total = time.perf_counter()

    if args.walk_forward_gap:
        _run_walk_forward_gap(args, tickers, provider)
    elif args.sweep:
        _run_sweep(args, tickers, provider, bench_series)
    else:
        _run_single(args, tickers, provider, bench_series)

    print(f"\n  Total wall time: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
