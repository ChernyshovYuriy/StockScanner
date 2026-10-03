"""
volume_breakout_target_backtest.py
===================================
Follow-up to volume_breakout_backtest.py (2026-10): instead of a trailing
stop, sell as soon as the trade is up N% ("volume spike causes a price
spike -- take the pop"). Same events (signal_masks() from that script),
same next-open entry, same 1% round-trip cost.

Exit grid, every combination:
  target    +N% above entry, N in TARGETS_PCT. Filled at the target, or at
            the open if a later day gaps above it.
  stop      none, or the signal day's low (filled at the stop, or at the
            open if a day gaps below it). Checked BEFORE the target on a
            bar that touches both -- the conservative assumption, since a
            daily bar can't say which came first.
  max hold  sell at the close after MAX_HOLD trading days if neither hit.

A profit target makes almost ANY entry look good on win rate (many small
wins, a few big losses), so every combination is also run on random
entry days (every BASELINE_SAMPLE_STEP-th liquidity-eligible day) and
judged on (event mean - baseline mean), not on win rate.

Choosing the best N after seeing the whole history would be curve
fitting, so the headline number is walk-forward: for each test fold f
(from WF_FIRST_TEST_FOLD on), pick the combination with the best
(event - baseline) on folds < f only, then score it on fold f.

Usage:
    python volume_breakout_target_backtest.py
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH, VOLUME_SPIKE_TICKERS_URL
from market_data_cache import sync_and_load
from research.triple_screen.batch import load_tickers
from volume_breakout_backtest import (BREAKOUT_LOOKBACK, END, FETCH_START, N_FOLDS,
                                      ROUND_TRIP_COST, START, signal_masks)

TARGETS_PCT = [3, 5, 7, 10, 15, 20, 30, 50]
STOPS = ["none", "signal_low"]
MAX_HOLDS = [20, 60]
VARIANTS = ["current_rule", "breakout_vol3_quiet", "breakout_vol5_quiet"]
BASELINE_SAMPLE_STEP = 10
WF_FIRST_TEST_FOLD = 4

COMBOS = list(itertools.product(TARGETS_PCT, STOPS, MAX_HOLDS))
SUMMARY_CSV = OUT_PATH / "volume_breakout_target_summary.csv"
WALK_FORWARD_CSV = OUT_PATH / "volume_breakout_target_walk_forward.csv"


def _exit_returns(i: int, o, h, l, c) -> np.ndarray:
    """Net return for every COMBOS entry, for a signal on bar i. NaN where
    the trade can't complete inside the available history."""
    out = np.full(len(COMBOS), np.nan)
    n = len(c)
    if i + 1 >= n or not o[i + 1] > 0:
        return out
    entry = o[i + 1]
    stop0 = l[i]
    for idx, (tgt_pct, stop_mode, max_hold) in enumerate(COMBOS):
        last = i + max_hold
        if last >= n:
            continue
        tgt = entry * (1 + tgt_pct / 100)
        sl = slice(i + 1, last + 1)
        oo, hh, ll = o[sl], h[sl], l[sl]
        hit_t = np.flatnonzero(hh >= tgt)
        hit_s = np.flatnonzero(ll < stop0) if stop_mode == "signal_low" else np.array([], dtype=int)
        jt = hit_t[0] if hit_t.size else 10**9
        js = hit_s[0] if hit_s.size else 10**9
        if js <= jt and js < 10**9:
            px = min(oo[js], stop0) if js > 0 else min(entry, stop0)
        elif jt < 10**9:
            px = max(oo[jt], tgt) if jt > 0 else tgt
        else:
            px = c[last]
        out[idx] = px / entry - 1 - ROUND_TRIP_COST
    return out


def main() -> None:
    tickers = [t for t in load_tickers(VOLUME_SPIKE_TICKERS_URL) if not t.endswith(".NE")]
    print(f"Universe: {len(tickers)} tickers  |  window {START} .. {END}  |  {len(COMBOS)} exit combos")
    t0 = time.perf_counter()
    data = sync_and_load(tickers, start=FETCH_START, end=END, quiet=True)
    print(f"Daily bars loaded in {time.perf_counter() - t0:.1f}s")

    bounds = pd.date_range(start=START, end=END, periods=N_FOLDS + 1)
    ev_rows, base_rows = [], []
    t0 = time.perf_counter()
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < BREAKOUT_LOOKBACK + 50:
            continue
        df = df.sort_index()
        eligible, _volx, variants = signal_masks(df)
        in_window = np.asarray(df.index >= pd.Timestamp(START))
        o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
        folds = np.clip(np.searchsorted(bounds, df.index, side="right") - 1, 0, N_FOLDS - 1)
        for name in VARIANTS:
            mask = variants[name] & in_window
            if name != "current_rule":
                from volume_breakout_backtest import _cooldown
                mask = _cooldown(mask)
            for i in np.flatnonzero(mask):
                ev_rows.append((name, folds[i], _exit_returns(i, o, h, l, c)))
        for i in np.flatnonzero(eligible & in_window)[::BASELINE_SAMPLE_STEP]:
            base_rows.append(("baseline", folds[i], _exit_returns(i, o, h, l, c)))
    print(f"Simulated in {time.perf_counter() - t0:.1f}s")

    def to_frame(rows):
        return pd.DataFrame({"variant": [r[0] for r in rows], "fold": [r[1] for r in rows]}).join(
            pd.DataFrame(np.vstack([r[2] for r in rows]), columns=range(len(COMBOS))))

    ev, base = to_frame(ev_rows), to_frame(base_rows)
    base_fold = base.groupby("fold")[list(range(len(COMBOS)))].mean()

    summary, wf_rows = [], []
    for name in VARIANTS:
        e = ev[ev["variant"] == name]
        ev_fold = e.groupby("fold")[list(range(len(COMBOS)))].mean()
        n_fold = e.groupby("fold").size()
        diff_fold = (ev_fold - base_fold).loc[n_fold[n_fold >= 3].index]
        for k, (tgt, stop, hold) in enumerate(COMBOS):
            r = e[k].dropna()
            summary.append({
                "variant": name, "target%": tgt, "stop": stop, "max_hold": hold, "n": len(r),
                "mean%": round(r.mean() * 100, 2), "median%": round(r.median() * 100, 2),
                "win%": round((r > 0).mean() * 100, 1),
                "base_mean%": round(base[k].mean() * 100, 2), "base_win%": round((base[k].dropna() > 0).mean() * 100, 1),
                "edge%": round((diff_fold[k].mean()) * 100, 2),
                "folds+": f"{int((diff_fold[k] > 0).sum())}/{diff_fold[k].notna().sum()}",
            })
        for f in range(WF_FIRST_TEST_FOLD, N_FOLDS):
            train = diff_fold[diff_fold.index < f]
            if f not in diff_fold.index or train.empty:
                continue
            best = int(train.mean().idxmax())
            tgt, stop, hold = COMBOS[best]
            wf_rows.append({"variant": name, "test_fold": f, "chosen": f"+{tgt}% / {stop} / {hold}d",
                            "test_edge%": round(diff_fold.loc[f, best] * 100, 2)})

    summary = pd.DataFrame(summary)
    wf = pd.DataFrame(wf_rows)
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    wf.to_csv(WALK_FORWARD_CSV, index=False)

    for name in VARIANTS:
        s = summary[summary["variant"] == name].sort_values("edge%", ascending=False)
        print(f"\n=== {name}: top 8 combos on the FULL history (in-sample) ===")
        print(tabulate(s.head(8).drop(columns="variant"), headers="keys", tablefmt="simple", showindex=False))
        w = wf[wf["variant"] == name]
        if len(w) >= 2:
            t_stat, p_val = stats.ttest_1samp(w["test_edge%"], 0.0)
            print(f"--- walk-forward (out-of-sample): mean edge {w['test_edge%'].mean():+.2f}%/trade, "
                  f"{(w['test_edge%'] > 0).sum()}/{len(w)} folds positive, t={t_stat:.2f}, p={p_val:.3f}")
            print("    chosen per fold:", ", ".join(w["chosen"].value_counts().head(4).index))
    print(f"\nWrote {SUMMARY_CSV} and {WALK_FORWARD_CSV}")


if __name__ == "__main__":
    main()
