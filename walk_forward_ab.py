"""
walk_forward_ab.py -- 16-fold walk-forward A/B validation of one exit-param
combo against another (e.g. a backtest-sweep winner vs the current live
config), over a 4-year window.

This is a fixed head-to-head A/B, not an in-sample-optimize/out-of-sample
cycle: it's meant to run AFTER a candidate combo has already been chosen
(e.g. by a grid search over the full window), to check whether it survives
being split into independent sub-periods rather than just looking good in
aggregate -- the same overfitting risk a "best of N combos" search always
carries. Per-fold win count + paired t-test on the return and max-DD
differences, same methodology as the sector-cap and gap-filter
walk-forwards (see sector-concentration-2026-08 / gap-filter-walk-forward-
2026-07 memories).

Each fold starts fresh with its own initial_cash (no cross-fold
compounding), matching run_backtest.py's --walk-forward-gap convention.
Screener scores are pre-computed once for the full period and re-indexed
per fold.

Edit CANDIDATE / BASELINE below to set up a new comparison -- like
param_sweep-style research scripts, this is a one-off analysis tool, not
a general CLI (see AGENTS.md scope discipline: in-code constants, not
flags, for this kind of thing).

Usage:
    python walk_forward_ab.py
"""
from __future__ import annotations

import signal
import time
import traceback
import urllib.request
from datetime import datetime as _dt

import pandas as pd
from scipy import stats

from backtest_runner import (
    BacktestConfig, BacktestRunner, _lookback_start,
    _run_screener_step, _trading_days,
)
from position_monitor import ExitParams
from market_data import HistoricalSliceProvider
from time_utils import set_backtest_clock, TSX_TZ
from config import CAN_TICKERS_URL, OUT_PATH

# Per-run hang guard: an earlier grid-search run over this same backtest
# loop stalled on one combo for 7+ hours with zero CPU/progress and no
# identified root cause (every network call site in the loop is
# provider-only; no live yfinance fetch should be reachable mid-run).
# Bounds each of the fold-config runs individually; a timeout prints the
# stack it was actually parked on (diagnostic) and logs a TIMEOUT row
# instead of silently stalling the rest of the sweep.
RUN_TIMEOUT_S = 900


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout()


# TICKERS_SOURCE accepts a URL or a local file path (one ticker per line).
# NOTE: CAN_TICKERS_URL (the Financing repo's published list) has been seen
# to serve unresolved git merge-conflict markers when the remote repo has
# a pending conflict -- if that happens, point this at a locally-saved
# clean snapshot instead rather than let it silently corrupt the universe.
TICKERS_SOURCE = CAN_TICKERS_URL

START = "2022-08-26"
END = "2026-08-26"
BENCHMARK = "XIU.TO"
CAPITAL = 50_000.0
LOOKBACK_DAYS = 504
SCREENER_FREQ = 5
N_FOLDS = 16
OUT_DIR = str(OUT_PATH)
RESULTS_CSV = f"{OUT_DIR}/walk_forward_ab_results.csv"

CANDIDATE = dict(label="candidate", atr_stop_mult=1.0, trail_atr=2.5, ts_days=14, ts_pct=0.5)
BASELINE  = dict(label="baseline",  atr_stop_mult=1.5, trail_atr=2.5, ts_days=14, ts_pct=0.5)


def load_tickers(source: str) -> list[str]:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            content = resp.read().decode("utf-8")
    else:
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
    return [ln.strip() for ln in content.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def _sharpe(equity: pd.Series) -> float:
    if equity is None or len(equity) < 5:
        return 0.0
    dr = equity.pct_change().dropna()
    if len(dr) < 5 or dr.std() == 0:
        return 0.0
    return float(dr.mean() / dr.std() * (252 ** 0.5))


def _max_dd(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max * 100
    return float(dd.min())


def main():
    tickers = load_tickers(TICKERS_SOURCE)
    if BENCHMARK not in tickers:
        tickers = tickers + [BENCHMARK]
    tickers = list(dict.fromkeys(tickers))
    print(f"Universe: {len(tickers)-1} tickers + {BENCHMARK}")

    fetch_start = _lookback_start(START, LOOKBACK_DAYS)
    t0 = time.perf_counter()
    provider = HistoricalSliceProvider.from_cache(tickers=tickers, start=fetch_start, end=END)
    print(f"Data loaded: {len(provider)} tickers in {time.perf_counter()-t0:.1f}s")

    base_kwargs = dict(
        tickers=tickers, benchmark=BENCHMARK,
        initial_cash=CAPITAL, risk_pct=1.0, top_n_buys=3, min_score=0.0,
        lookback_days=LOOKBACK_DAYS, screener_frequency=SCREENER_FREQ,
        max_tracked_tickers=40, regime_filter=True, min_rr=2.0,
        max_positions=8, sizing="live", sizing_basis="cash",
        gap_filter_pct=None, _provider=provider,
    )

    # Pre-compute screener scores once for the FULL period (base config just
    # needs to be a valid BacktestConfig -- exit params don't affect the
    # screener step, only pattern detection/sizing downstream reruns per-run).
    precompute_cfg = BacktestConfig(
        exit_params=ExitParams(initial_stop_atr_k=1.5, chand_trail_atr_k=2.5,
                                time_stop_days=14, time_stop_min_profit=0.5,
                                stop_trigger="close"),
        atr_stop_mult=1.5, start_date=START, end_date=END, **base_kwargs,
    )
    trading_days = _trading_days(START, END)
    n = len(trading_days)
    full_cache: dict = {}
    print("Pre-computing screener scores (once, reused across all folds/configs)...", end="", flush=True)
    t0 = time.perf_counter()
    for i, day in enumerate(trading_days):
        if i % SCREENER_FREQ == 0:
            set_backtest_clock(_dt(day.year, day.month, day.day, 16, 5, tzinfo=TSX_TZ))
            full_cache[i] = _run_screener_step(precompute_cfg, provider, pd.Timestamp(day))
            print(".", end="", flush=True)
    set_backtest_clock(None)
    print(f"  done in {time.perf_counter()-t0:.1f}s ({len(full_cache)} snapshots)")

    def _make_sub_cache(offset: int, length: int) -> dict:
        sub: dict = {}
        for j in range(length):
            sub_key = (j // SCREENER_FREQ) * SCREENER_FREQ
            if sub_key in sub:
                continue
            full_key = ((offset + j) // SCREENER_FREQ) * SCREENER_FREQ
            if full_key in full_cache:
                sub[sub_key] = full_cache[full_key]
        return sub

    # N_FOLDS non-overlapping folds spanning the full period (~65 trading
    # days each at 16 folds over 4 years, i.e. ~1 quarter) -- each fold is a
    # fresh independent OOS window, not an in-sample-optimize/out-of-sample
    # split (the candidate combo is assumed already chosen elsewhere; this
    # only checks it doesn't just look good in aggregate).
    bounds = [round(i * n / N_FOLDS) for i in range(N_FOLDS + 1)]
    folds = list(zip(bounds[:-1], bounds[1:]))

    def _run(cfg_spec: dict, start_i: int, end_i: int, sub_cache: dict):
        start_date = str(trading_days[start_i])
        end_date = str(trading_days[end_i - 1]) if end_i <= n else END
        ep = ExitParams(initial_stop_atr_k=1.5, chand_trail_atr_k=cfg_spec["trail_atr"],
                         time_stop_days=cfg_spec["ts_days"], time_stop_min_profit=cfg_spec["ts_pct"],
                         stop_trigger="close")
        cfg = BacktestConfig(exit_params=ep, atr_stop_mult=cfg_spec["atr_stop_mult"],
                              start_date=start_date, end_date=end_date,
                              _screener_cache=sub_cache, **base_kwargs)
        signal.alarm(RUN_TIMEOUT_S)
        try:
            return BacktestRunner(cfg).run(verbose=False)
        finally:
            signal.alarm(0)

    signal.signal(signal.SIGALRM, _alarm_handler)

    rows = []
    ret_diffs = []   # candidate - baseline, per fold
    dd_diffs = []
    ret_wins = 0
    dd_wins = 0
    sharpe_wins = 0

    for f_idx, (s, e) in enumerate(folds, 1):
        fold_str = f"{trading_days[s]} -> {trading_days[e-1]}"
        sub_cache = _make_sub_cache(s, e - s)
        fold_row = {"fold": f_idx, "period": fold_str}
        metrics = {}
        for spec in (CANDIDATE, BASELINE):
            label = spec["label"]
            t0 = time.perf_counter()
            try:
                res = _run(spec, s, e, sub_cache)
                eq = res.equity_curve_df()
                equity = eq["total_equity"] if not eq.empty else pd.Series([CAPITAL])
                end_eq = float(equity.iloc[-1])
                ret_pct = (end_eq / CAPITAL - 1) * 100
                dd = _max_dd(equity)
                sh = _sharpe(equity)
                n_trades = len(res.trade_log_df())
                status = "OK"
            except RunTimeout:
                elapsed = time.perf_counter() - t0
                print(f"  fold {f_idx} [{label}] -> TIMEOUT after {elapsed:.1f}s, see traceback:")
                traceback.print_exc()
                ret_pct, dd, sh, n_trades, status = float("nan"), float("nan"), float("nan"), 0, "TIMEOUT"
            elapsed = time.perf_counter() - t0
            metrics[label] = dict(ret=ret_pct, dd=dd, sharpe=sh, trades=n_trades)
            fold_row[f"{label}_ret_pct"] = round(ret_pct, 2) if status == "OK" else float("nan")
            fold_row[f"{label}_dd_pct"] = round(dd, 2) if status == "OK" else float("nan")
            fold_row[f"{label}_sharpe"] = round(sh, 2) if status == "OK" else float("nan")
            fold_row[f"{label}_trades"] = n_trades
            fold_row[f"{label}_status"] = status
            fold_row[f"{label}_elapsed_s"] = round(elapsed, 1)

        c, b = metrics["candidate"], metrics["baseline"]
        if fold_row["candidate_status"] == "OK" and fold_row["baseline_status"] == "OK":
            ret_diffs.append(c["ret"] - b["ret"])
            dd_diffs.append(c["dd"] - b["dd"])  # less negative (higher) = better
            ret_wins += int(c["ret"] > b["ret"])
            dd_wins += int(c["dd"] > b["dd"])
            sharpe_wins += int(c["sharpe"] > b["sharpe"])

        print(f"[fold {f_idx:>2}/{N_FOLDS}] {fold_str}  "
              f"candidate: ret={c['ret']:+.1f}% dd={c['dd']:.1f}% sharpe={c['sharpe']:.2f} trades={c['trades']}  |  "
              f"baseline: ret={b['ret']:+.1f}% dd={b['dd']:.1f}% sharpe={b['sharpe']:.2f} trades={b['trades']}")

        rows.append(fold_row)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    n_valid = len(ret_diffs)
    print(f"\n{'='*100}\nWalk-Forward A/B Summary ({n_valid}/{N_FOLDS} valid folds)\n{'='*100}")
    print(f"Candidate: atr_stop={CANDIDATE['atr_stop_mult']} trail={CANDIDATE['trail_atr']} ts_days={CANDIDATE['ts_days']}")
    print(f"Baseline:  atr_stop={BASELINE['atr_stop_mult']} trail={BASELINE['trail_atr']} ts_days={BASELINE['ts_days']}")
    print(f"\nReturn wins (candidate > baseline):   {ret_wins}/{n_valid}")
    print(f"Max-DD wins (candidate shallower):    {dd_wins}/{n_valid}")
    print(f"Sharpe wins (candidate > baseline):   {sharpe_wins}/{n_valid}")

    if n_valid >= 2:
        t_ret, p_ret = stats.ttest_1samp(ret_diffs, 0.0)
        t_dd, p_dd = stats.ttest_1samp(dd_diffs, 0.0)
        print(f"\nPaired t-test on per-fold return diff  (candidate - baseline): "
              f"mean={sum(ret_diffs)/n_valid:+.2f}pp  t={t_ret:.2f}  p={p_ret:.3f}")
        print(f"Paired t-test on per-fold max-DD diff  (candidate - baseline): "
              f"mean={sum(dd_diffs)/n_valid:+.2f}pp  t={t_dd:.2f}  p={p_dd:.3f}")

    print(f"\nResults -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
