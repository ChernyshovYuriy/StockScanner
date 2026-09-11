"""
Phase 2: historical verification of the bullish Kangaroo Tail detector --
does it have a real forward-looking edge, or does it just look plausible on
a chart? Same methodology as this repo's other validated/rejected features
(walk_forward_ab.py, the sector-cap and gap-filter walk-forwards): 16
non-overlapping chronological folds over a multi-year window, a per-fold
"candidate vs. baseline" metric, win count + a paired one-sample t-test
(scipy.stats.ttest_1samp) on the per-fold differences -- not just an
aggregate number, which every "looked great in aggregate, failed
out-of-sample" precedent in this repo (dip-bounce grid, rotation exit,
entry-timing A/B) exists precisely to guard against.

This is an event study, not a strategy backtest -- there's no equity curve,
sizing, or exits to simulate. For each (TailConfig variant, alert-timing
variant, holding-window K) combination:
  - "candidate" = the mean K-trading-day forward return starting at the
    signal's entry point, pooled across every detection in a fold.
  - "baseline"  = the mean K-trading-day forward return of EVERY ticker-day
    in that same fold (not just signal days) -- i.e. each ticker's own
    normal drift over the same period, so the comparison isolates whether
    the tail predicts an ABOVE-NORMAL move, not just "stocks went up that
    quarter." A single index baseline (e.g. XIU.TO) would conflate
    universe-wide drift with ticker-specific drift; this self-baseline
    controls for both.
  - A fold "wins" when candidate > baseline. Across all folds with at
    least one detection, the per-fold (candidate - baseline) differences
    feed the paired t-test.

Alert-timing variants (see detector.confirms_next_bar()):
  same_day        -- entry at the tail bar's own close.
  confirmed_high  -- entry at the next bar's close, but ONLY if that bar
                     closed above the tail bar's high (the standard,
                     stricter price-action confirmation).
  confirmed_mid   -- entry at the next bar's close, but only if it closed
                     above the tail bar's midpoint (a looser bar to clear).

TailConfig variants swept (see CONFIGS below): the shipped default
("strict"), a loosened-wick variant, and a "no_break" variant that
effectively disables the structure-break rule (break_atr_mult set far
below any real value) -- the plain-hammer comparison done ad hoc
in-session for the AAPL 2026-09-09 bar, now run properly across the whole
universe and history instead of eyeballed on one ticker/one day.

Edit CONFIGS / TIMINGS / HOLDING_DAYS / N_FOLDS below to change the sweep
-- in-code constants, not CLI flags, per this repo's own convention for
one-off research scripts (see AGENTS.md scope discipline / walk_forward_ab.py's
own "Edit CANDIDATE/BASELINE below" comment).

Usage:
    python -m research.kangaroo_tail.verify
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import CAN_TICKERS_URL, OUT_PATH
from market_data_cache import sync_and_load
from time_utils import market_today
from .batch import load_tickers
from .detector import confirms_next_bar, scan_history
from .types import TailConfig

# ── Sweep definition ────────────────────────────────────────────────────────

#
# NOTE: every entry below passes its thresholds explicitly rather than
# relying on TailConfig()'s bare defaults, precisely so this sweep's labels
# keep their original meaning even after types.py's shipped defaults were
# updated (2026-09) to match "sweet_spot" below -- otherwise "strict" would
# silently start meaning something different the next time defaults change.
CONFIGS: dict[str, TailConfig] = {
    "strict":     TailConfig(break_atr_mult=1.0, wick_dominance_pct=0.6, wick_atr_mult=1.5),
    "loose_wick": TailConfig(break_atr_mult=1.0, wick_dominance_pct=0.5, wick_atr_mult=1.0),
    # break_atr_mult this far below zero makes "break_amount < break_atr_mult
    # * atr" practically never true regardless of atr's sign/magnitude --
    # i.e. the structure-break rule is a no-op. Every other rule (wick
    # dominance, body, close position, volume) stays at strict defaults, so
    # this isolates exactly the plain-hammer comparison done ad hoc for the
    # AAPL 2026-09-09 bar.
    "no_break":   TailConfig(break_atr_mult=-999.0, wick_dominance_pct=0.6, wick_atr_mult=1.5),
    # Fine sweep (see KANGAROO_TAIL_VERIFICATION_FINDINGS.md) found a sharp
    # cliff in edge quality between wick_atr_mult=1.2 and 1.0, not a smooth
    # gradient -- 1.2/0.55 is the best statistically-supported point above
    # that cliff (n=143, win=56.6%, avg_R=+0.246, PF=1.72, p=0.033 on
    # breakout entry, rr=1.5) and is now types.py's shipped default.
    "sweet_spot": TailConfig(break_atr_mult=-999.0, wick_dominance_pct=0.55, wick_atr_mult=1.2),
    # Loosened further still (0.7x ATR) to also catch the AAPL 2026-09-09
    # bar itself (wick=0.71x its own ATR) -- included here only to document
    # that catching that specific bar requires going BELOW the quality
    # cliff (n=949, win=46.7%, avg_R=+0.11, PF=1.23, still p=0.048 due to
    # sheer sample size, not per-trade quality). Not shipped as a default.
    "loose_all":  TailConfig(break_atr_mult=-999.0, wick_dominance_pct=0.5, wick_atr_mult=0.7),
}

TIMINGS = ["same_day", "confirmed_high", "confirmed_mid"]
HOLDING_DAYS = [5, 10, 20]
N_FOLDS = 16

# 4-year-ish window ending a bit before "today" -- FORWARD_BUFFER_DAYS
# leaves enough trailing daily bars (calendar days, so weekends/holidays
# are covered) for even the largest HOLDING_DAYS to have real forward data
# for events near the end of the analysis window. Computed relative to
# market_today() rather than hardcoded so re-running this later doesn't
# require hand-editing a stale end date -- still a one-off research script,
# just one that doesn't go stale on its own (see module docstring).
FORWARD_BUFFER_DAYS = 45
_END_TS = market_today() - pd.Timedelta(days=FORWARD_BUFFER_DAYS)
END = _END_TS.strftime("%Y-%m-%d")
START = (_END_TS - pd.Timedelta(days=365 * 4)).strftime("%Y-%m-%d")
FETCH_END = market_today().strftime("%Y-%m-%d")  # fetch through today so
                                                   # forward closes exist
                                                   # near the analysis END

RESULTS_CSV = OUT_PATH / "kangaroo_tail_verify_results.csv"


# ── Data prep ────────────────────────────────────────────────────────────────

def _clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.sort_index().dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def _entry_events(ticker: str, clean: pd.DataFrame, config: TailConfig,
                   timing: str) -> list[tuple[pd.Timestamp, float]]:
    """Every (entry_date, entry_price) pair `timing` would have actually
    traded for `ticker` under `config` -- same_day enters on the tail bar
    itself; confirmed_* only keeps signals where the very next bar
    confirms (see detector.confirms_next_bar()), entering at that next
    bar's close instead.
    """
    signals = scan_history(ticker, clean, config)
    events = []
    for signal in signals:
        if timing == "same_day":
            events.append((signal.date, signal.close))
            continue
        pos = clean.index.get_loc(signal.date)
        if pos + 1 >= len(clean):
            continue  # no next bar yet (signal too close to end of history)
        next_bar = clean.iloc[pos + 1]
        use_midpoint = timing == "confirmed_mid"
        if confirms_next_bar(signal, next_bar, use_midpoint=use_midpoint):
            events.append((clean.index[pos + 1], float(next_bar["Close"])))
    return events


def _forward_return(clean: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float,
                     k: int) -> float | None:
    pos = clean.index.get_loc(entry_date)
    if pos + k >= len(clean):
        return None  # not enough trailing data yet for this K
    exit_price = float(clean["Close"].iloc[pos + k])
    if entry_price <= 0 or not np.isfinite(exit_price):
        return None
    return exit_price / entry_price - 1.0


def _baseline_forward_returns(clean: pd.DataFrame, k: int) -> pd.Series:
    """Every day's own unconditional K-day forward return -- ticker's
    normal drift, the "would this ticker have moved this much anyway"
    control (see module docstring)."""
    closes = clean["Close"]
    n = len(closes)
    if n <= k:
        return pd.Series(dtype=float)
    fwd = closes.shift(-k) / closes - 1.0
    return fwd.iloc[:n - k]


def _fold_bounds() -> np.ndarray:
    bounds = pd.date_range(start=START, end=END, periods=N_FOLDS + 1)
    return bounds.values


def _fold_of(dates: pd.DatetimeIndex, bounds: np.ndarray) -> np.ndarray:
    """Fold index (0 .. N_FOLDS-1) for each date, or -1 if outside
    [START, END]."""
    idx = np.searchsorted(bounds, dates.values, side="right") - 1
    idx[(dates.values < bounds[0]) | (dates.values > bounds[-1])] = -1
    idx[idx == N_FOLDS] = N_FOLDS - 1  # date exactly == END lands in last fold
    return idx


# ── Main sweep ───────────────────────────────────────────────────────────────

def main() -> None:
    tickers = load_tickers(CAN_TICKERS_URL)
    print(f"Universe: {len(tickers)} tickers  |  window {START} .. {END}  "
          f"(data fetched through {FETCH_END})  |  {N_FOLDS} folds")

    t0 = time.perf_counter()
    bars_by_ticker = sync_and_load(tickers, start=START, end=FETCH_END)
    print(f"Data loaded in {time.perf_counter() - t0:.1f}s")

    cleaned: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        bars = bars_by_ticker.get(ticker)
        if bars is None or bars.empty:
            continue
        clean = _clean_bars(bars)
        if len(clean) < 60:  # not enough history to be worth scanning
            continue
        cleaned[ticker] = clean
    print(f"{len(cleaned)}/{len(tickers)} tickers have usable history")

    bounds = _fold_bounds()

    # Baseline forward-return pool per K, restricted to [START, END] and
    # tagged with its fold -- computed once, reused across every
    # config/timing combo swept below (the baseline doesn't depend on
    # either).
    print("Pre-computing per-ticker baseline forward returns...")
    baseline_by_fold_k: dict[int, dict[int, list[float]]] = {k: {f: [] for f in range(N_FOLDS)} for k in HOLDING_DAYS}
    for ticker, clean in cleaned.items():
        in_window = clean.index[(clean.index >= bounds[0]) & (clean.index <= bounds[-1])]
        folds = _fold_of(in_window, bounds)
        for k in HOLDING_DAYS:
            fwd = _baseline_forward_returns(clean, k)
            fwd = fwd.reindex(in_window).dropna()
            for date, ret in fwd.items():
                fold = folds[in_window.get_loc(date)]
                if fold >= 0:
                    baseline_by_fold_k[k][fold].append(float(ret))

    summary_rows = []
    detail_rows = []

    for config_label, config in CONFIGS.items():
        for timing in TIMINGS:
            events_by_ticker: dict[str, list[tuple[pd.Timestamp, float]]] = {}
            for ticker, clean in cleaned.items():
                events_by_ticker[ticker] = _entry_events(ticker, clean, config, timing)
            total_events = sum(len(v) for v in events_by_ticker.values())

            for k in HOLDING_DAYS:
                candidate_by_fold: dict[int, list[float]] = {f: [] for f in range(N_FOLDS)}
                for ticker, events in events_by_ticker.items():
                    clean = cleaned[ticker]
                    for entry_date, entry_price in events:
                        if not (bounds[0] <= np.datetime64(entry_date) <= bounds[-1]):
                            continue
                        fold = int(_fold_of(pd.DatetimeIndex([entry_date]), bounds)[0])
                        if fold < 0:
                            continue
                        ret = _forward_return(clean, entry_date, entry_price, k)
                        if ret is not None:
                            candidate_by_fold[fold].append(ret)

                ret_diffs, wins, n_valid, n_events_used = [], 0, 0, 0
                for f in range(N_FOLDS):
                    cand = candidate_by_fold[f]
                    base = baseline_by_fold_k[k][f]
                    if not cand or not base:
                        continue
                    cand_mean = float(np.mean(cand))
                    base_mean = float(np.mean(base))
                    diff = cand_mean - base_mean
                    ret_diffs.append(diff)
                    wins += int(diff > 0)
                    n_valid += 1
                    n_events_used += len(cand)
                    detail_rows.append({
                        "config": config_label, "timing": timing, "k_days": k,
                        "fold": f + 1, "n_events": len(cand),
                        "candidate_ret_pct": round(cand_mean * 100, 2),
                        "baseline_ret_pct": round(base_mean * 100, 2),
                        "diff_pct": round(diff * 100, 2),
                    })

                if n_valid >= 2:
                    t_stat, p_val = stats.ttest_1samp(ret_diffs, 0.0)
                    mean_diff = float(np.mean(ret_diffs))
                else:
                    t_stat, p_val, mean_diff = float("nan"), float("nan"), (
                        float(np.mean(ret_diffs)) if ret_diffs else float("nan"))

                summary_rows.append({
                    "config": config_label, "timing": timing, "k_days": k,
                    "total_events": total_events, "events_with_fwd_data": n_events_used,
                    "folds_with_data": n_valid, "wins": wins,
                    "mean_diff_pct": round(mean_diff * 100, 2) if np.isfinite(mean_diff) else float("nan"),
                    "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else float("nan"),
                    "p_value": round(p_val, 3) if np.isfinite(p_val) else float("nan"),
                })
                if np.isfinite(p_val):
                    print(f"[{config_label:<11} {timing:<14} k={k:>2}d]  "
                          f"events={total_events:>4}  folds_used={n_valid:>2}/{N_FOLDS}  "
                          f"wins={wins:>2}/{n_valid}  "
                          f"mean_diff={mean_diff*100:+.2f}pp  p={p_val:.3f}")
                else:
                    print(f"[{config_label:<11} {timing:<14} k={k:>2}d]  events={total_events:>4}  "
                          f"(insufficient folds for a t-test)")

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(RESULTS_CSV.with_name("kangaroo_tail_verify_folds.csv"), index=False)
    df_summary = pd.DataFrame(summary_rows).sort_values("p_value", na_position="last")
    df_summary.to_csv(RESULTS_CSV, index=False)

    print(f"\n{'='*100}\nKangaroo Tail Phase 2 -- summary, ranked by p-value (lower = more significant)\n{'='*100}")
    print(tabulate(df_summary, headers="keys", tablefmt="github", showindex=False))
    print(f"\nPer-fold detail -> {RESULTS_CSV.with_name('kangaroo_tail_verify_folds.csv')}")
    print(f"Summary          -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
