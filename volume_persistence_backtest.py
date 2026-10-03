"""
volume_persistence_backtest.py
===============================
Follow-up to volume_breakout_backtest.py (2026-10), testing the user's
observation from MATR.TO (Jul 2026) and GWM.V (Sep 2026): a real spike
keeps going for several days, heavy volume day after day, rather than one
heavy day and done. volume_breakout_backtest.py only ever bought after the
FIRST heavy day, so it couldn't tell those apart.

Day 0: close at a BREAKOUT_LOOKBACK-day closing high on volume >= 3x the
long baseline (the baseline is frozen as of the day before day 0, so the
spike's own days never inflate it). Liquidity filter as in that script.
Confirmation over the next D days (D = 1, 2, 4), the buy at the open
after day D only if confirmed:
  hot  every day 1..D has volume >= HOT_MULT x baseline
  held day D's close >= day 0's close (the spike's gain hasn't been given back)
Variants compare confirmed vs NOT confirmed after the same day-0 events,
so the persistence condition itself is what's being isolated. Exits,
cost, baseline (random liquidity-eligible days), folds and t-test are
volume_breakout_backtest.py's.

Usage:
    python volume_persistence_backtest.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH, VOLUME_SPIKE_TICKERS_URL
from market_data_cache import sync_and_load
from research.triple_screen.batch import load_tickers
from volume_breakout_backtest import (BASE_VOL_DAYS, BASELINE_SAMPLE_STEP, BREAKOUT_LOOKBACK, END, FETCH_START,
                                      HOLD_DAYS, MIN_DOLLAR_VOL, MIN_PRICE, N_FOLDS, ROUND_TRIP_COST, START,
                                      _cooldown, _trail_return)

DAY0_MULT = 3.0
HOT_MULT = 2.0
CONFIRM_DAYS = [1, 2, 4]
EXITS = [f"hold_{k}" for k in HOLD_DAYS] + ["trail"]
SUMMARY_CSV = OUT_PATH / "volume_persistence_summary.csv"
EVENTS_CSV = OUT_PATH / "volume_persistence_events.csv"


def _ticker(ticker: str, df: pd.DataFrame):
    df = df.sort_index()
    o, l, c = df["Open"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    v = df["Volume"].astype(float)
    base = v.rolling(BASE_VOL_DAYS, min_periods=int(BASE_VOL_DAYS * 0.8)).mean().shift(1)
    dollar = (df["Close"] * v).rolling(BASE_VOL_DAYS, min_periods=int(BASE_VOL_DAYS * 0.8)).mean().shift(1)
    hi = df["Close"].shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=int(BREAKOUT_LOOKBACK * 0.8)).max()
    eligible = ((dollar >= MIN_DOLLAR_VOL) & (df["Close"] >= MIN_PRICE)).to_numpy()
    in_window = np.asarray(df.index >= pd.Timestamp(START))
    base_a, v_a = base.to_numpy(), v.to_numpy()
    day0 = _cooldown((df["Close"] > hi).to_numpy() & (v / base >= DAY0_MULT).to_numpy() & eligible & in_window)

    n = len(c)
    entry = np.r_[o[1:], np.nan]
    fixed = {k: np.where(entry > 0, np.r_[c[k:], np.full(k, np.nan)] / entry - 1 - ROUND_TRIP_COST, np.nan)
             for k in HOLD_DAYS}

    def returns(i: int) -> dict:
        r = {f"hold_{k}": fixed[k][i] for k in HOLD_DAYS}
        r["trail"] = _trail_return(i, o, c, l)
        return r

    rows = []
    for i0 in np.flatnonzero(day0):
        b = base_a[i0]
        rows.append({"ticker": ticker, "date": df.index[i0], "variant": "day0_all", **returns(i0)})
        for d in CONFIRM_DAYS:
            iD = i0 + d
            if iD >= n:
                continue
            hot = bool(np.all(v_a[i0 + 1:iD + 1] >= HOT_MULT * b))
            held = c[iD] >= c[i0]
            tag = "confirmed" if hot and held else ("hot_not_held" if hot else ("held_not_hot" if held else "faded"))
            rows.append({"ticker": ticker, "date": df.index[i0], "variant": f"D{d}_{tag}", **returns(iD)})

    base_idx = np.flatnonzero(eligible & in_window)
    brows = []
    for j, i in enumerate(base_idx):
        r = {"date": df.index[i], **{f"hold_{k}": fixed[k][i] for k in HOLD_DAYS}}
        r["trail"] = _trail_return(i, o, c, l) if j % BASELINE_SAMPLE_STEP == 0 else np.nan
        brows.append(r)
    return rows, brows


def main() -> None:
    tickers = [t for t in load_tickers(VOLUME_SPIKE_TICKERS_URL) if not t.endswith(".NE")]
    print(f"Universe: {len(tickers)} tickers  |  window {START} .. {END}")
    data = sync_and_load(tickers, start=FETCH_START, end=END, quiet=True)
    t0 = time.perf_counter()
    ev, bs = [], []
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < BREAKOUT_LOOKBACK + 50:
            continue
        r, b = _ticker(t, df)
        ev.extend(r)
        bs.extend(b)
    ev, bs = pd.DataFrame(ev), pd.DataFrame(bs)
    print(f"Simulated in {time.perf_counter() - t0:.1f}s")

    bounds = pd.date_range(start=START, end=END, periods=N_FOLDS + 1)
    for f in (ev, bs):
        f["fold"] = np.clip(np.searchsorted(bounds, f["date"], side="right") - 1, 0, N_FOLDS - 1)

    out = []
    order = ["day0_all"] + [f"D{d}_{t}" for d in CONFIRM_DAYS for t in ("confirmed", "hot_not_held", "held_not_hot", "faded")]
    for variant in order:
        e = ev[ev["variant"] == variant]
        if e.empty:
            continue
        for ex in EXITS:
            r = e[ex].dropna()
            diffs = [e.loc[e["fold"] == f, ex].dropna().mean() - bs.loc[bs["fold"] == f, ex].dropna().mean()
                     for f in range(N_FOLDS) if e.loc[e["fold"] == f, ex].notna().sum() >= 3]
            t_stat, p_val = stats.ttest_1samp(diffs, 0.0) if len(diffs) >= 2 else (np.nan, np.nan)
            srt = r.sort_values(ascending=False)
            out.append({
                "variant": variant, "exit": ex, "n": len(r), "mean%": round(r.mean() * 100, 2),
                "mean_ex_top5%": round(srt.iloc[5:].mean() * 100, 2) if len(srt) > 5 else np.nan,
                "median%": round(r.median() * 100, 2), "win%": round((r > 0).mean() * 100, 1),
                "base_mean%": round(bs[ex].dropna().mean() * 100, 2),
                "folds+": f"{sum(x > 0 for x in diffs)}/{len(diffs)}",
                "t": round(t_stat, 2), "p": round(p_val, 3),
            })
    summary = pd.DataFrame(out)
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    ev.to_csv(EVENTS_CSV, index=False)
    for ex in EXITS:
        print(f"\n=== exit: {ex} ===")
        print(tabulate(summary[summary["exit"] == ex].drop(columns="exit"), headers="keys",
                       tablefmt="simple", showindex=False))
    print(f"\nWrote {SUMMARY_CSV} and {EVENTS_CSV}")


if __name__ == "__main__":
    main()
