"""
volume_breakout_backtest.py
============================
Research question (2026-10, prompted by CYPH's Aug-2026 move): does a
"volume regime shift" breakout -- a big volume expansion measured against
a LONG baseline, on a close above a multi-month high, ideally after a
quiet base -- predict a multi-week move? This is a different signal from
volume_spike_scanner.py's production rule (volume > its own 20-day
average on any up day, which fires on roughly one ticker-day in seven and
goes silent mid-trend once the 20-day average catches up with the new
volume level).

Signal on day t (everything known at day t's close, no lookahead):
  breakout   close_t > highest close of the prior BREAKOUT_LOOKBACK days
  volume     volume_t >= VOL_MULT x mean volume of the prior BASE_VOL_DAYS days
  quiet base prior 20-day mean volume <= QUIET_VOL_RATIO x that same
             long baseline, AND prior BASE_RANGE_DAYS closes' max/min
             <= BASE_RANGE_MAX
  liquidity  prior BASE_VOL_DAYS mean dollar volume >= MIN_DOLLAR_VOL and
             close >= MIN_PRICE (applied to every variant AND the baseline)
One event per ticker per COOLDOWN_DAYS (first signal only).

Entry: day t+1's OPEN (you can't act on a close-based signal at that same
close without already knowing it). Exits, each net of ROUND_TRIP_COST:
  hold_k     close k trading days after the signal
  trail      exit at the close once close <= TRAIL_PCT below the highest
             close since entry, or below the signal day's low (initial
             stop), or at MAX_TRAIL_DAYS

Baseline: every liquidity-eligible ticker-day, same entry/exit rules
(trail baseline is sampled every BASELINE_SAMPLE_STEP days for speed).
Variants are nested so the controls isolate each ingredient:
  current_rule      production rule (vol > 20d avg & up day), reference
  breakout_only     new multi-month closing high, no volume condition
  breakout_vol3     + volume >= 3x long baseline
  breakout_vol3_quiet + quiet base
  breakout_vol5_quiet same, volume >= 5x
Per-fold (event mean - baseline mean) over N_FOLDS chronological folds,
one-sample t-test, same methodology as volume_spike_daily_backtest.py.

Caveat: the ticker list is TODAY's universe, so names that broke out and
then delisted are missing -- survivorship bias flatters every variant
(baseline included, but breakout events in failed names most of all).

Usage:
    python volume_breakout_backtest.py
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
from time_utils import market_today

BREAKOUT_LOOKBACK = 120
BASE_VOL_DAYS = 100
QUIET_VOL_RATIO = 1.2
BASE_RANGE_DAYS = 60
BASE_RANGE_MAX = 1.5
MIN_DOLLAR_VOL = 100_000
MIN_PRICE = 0.20
COOLDOWN_DAYS = 60

ROUND_TRIP_COST = 0.01            # 1% -- spread + commission on small/mid caps
HOLD_DAYS = [5, 20, 60]
TRAIL_PCT = 0.20
MAX_TRAIL_DAYS = 120
BASELINE_SAMPLE_STEP = 5
N_FOLDS = 16

FETCH_START = "2020-09-01"
START = "2021-03-01"              # after BREAKOUT_LOOKBACK warm-up
END = market_today().strftime("%Y-%m-%d")

SUMMARY_CSV = OUT_PATH / "volume_breakout_backtest_summary.csv"
EVENTS_CSV = OUT_PATH / "volume_breakout_backtest_events.csv"

EXITS = [f"hold_{k}" for k in HOLD_DAYS] + ["trail"]


def _trail_return(i: int, opens, closes, lows) -> float:
    """Entry at opens[i+1], trailing-stop exit at a close. NaN if the
    trade can't complete inside the available history (open trade)."""
    n = len(closes)
    if i + 1 >= n or not opens[i + 1] > 0:
        return np.nan
    entry = opens[i + 1]
    stop0 = lows[i]
    peak = closes[i + 1]
    for j in range(i + 1, min(n, i + 1 + MAX_TRAIL_DAYS)):
        c = closes[j]
        peak = max(peak, c)
        if c <= peak * (1 - TRAIL_PCT) or c < stop0:
            return c / entry - 1 - ROUND_TRIP_COST
    if i + MAX_TRAIL_DAYS < n:
        return closes[i + MAX_TRAIL_DAYS] / entry - 1 - ROUND_TRIP_COST
    return np.nan


def _cooldown(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    last = -10**9
    for i in np.flatnonzero(mask):
        if i - last >= COOLDOWN_DAYS:
            out[i] = True
            last = i
    return out


def signal_masks(df: pd.DataFrame):
    """(eligible, volx, {variant: bool mask}) for one ticker's sorted
    daily bars -- shared with volume_breakout_target_backtest.py."""
    c, v = df["Close"], df["Volume"].astype(float)
    base_vol = v.rolling(BASE_VOL_DAYS, min_periods=int(BASE_VOL_DAYS * 0.8)).mean().shift(1)
    vol20 = v.rolling(20).mean().shift(1)
    dollar = (c * v).rolling(BASE_VOL_DAYS, min_periods=int(BASE_VOL_DAYS * 0.8)).mean().shift(1)
    hi = c.shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=int(BREAKOUT_LOOKBACK * 0.8)).max()
    rng = c.shift(1).rolling(BASE_RANGE_DAYS).max() / c.shift(1).rolling(BASE_RANGE_DAYS).min()
    up = c > c.shift(1)

    eligible = ((dollar >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)).to_numpy()
    breakout = (c > hi).to_numpy() & eligible
    volx = (v / base_vol).to_numpy()
    quiet = ((vol20 <= QUIET_VOL_RATIO * base_vol) & (rng <= BASE_RANGE_MAX)).to_numpy()
    variants = {
        "current_rule": (v > vol20).to_numpy() & up.to_numpy() & eligible,
        "breakout_only": breakout,
        "breakout_vol3": breakout & (volx >= 3),
        "breakout_vol3_quiet": breakout & (volx >= 3) & quiet,
        "breakout_vol5_quiet": breakout & (volx >= 5) & quiet,
    }
    return eligible, volx, variants


def _ticker_frames(ticker: str, df: pd.DataFrame):
    df = df.sort_index()
    c, o, l = df["Close"], df["Open"], df["Low"]
    eligible, volx, variants = signal_masks(df)

    opens, closes, lows = o.to_numpy(), c.to_numpy(), l.to_numpy()
    n = len(closes)
    fixed = {}
    entry = np.r_[opens[1:], np.nan]
    for k in HOLD_DAYS:
        exit_c = np.r_[closes[k:], np.full(k, np.nan)]
        fixed[f"hold_{k}"] = np.where(entry > 0, exit_c / entry - 1 - ROUND_TRIP_COST, np.nan)

    dates = df.index
    in_window = np.asarray(dates >= pd.Timestamp(START))
    rows = []
    for name, mask in variants.items():
        idx = np.flatnonzero(_cooldown(mask & in_window)) if name != "current_rule" else np.flatnonzero(mask & in_window)
        for i in idx:
            r = {"ticker": ticker, "date": dates[i], "variant": name, "volx": volx[i]}
            for e in HOLD_DAYS:
                r[f"hold_{e}"] = fixed[f"hold_{e}"][i]
            # current_rule fires too often for a per-event trail loop to matter; fixed holds suffice
            r["trail"] = _trail_return(i, opens, closes, lows) if name != "current_rule" else np.nan
            rows.append(r)

    base_idx = np.flatnonzero(eligible & in_window)
    base = {"date": dates[base_idx]}
    for k in HOLD_DAYS:
        base[f"hold_{k}"] = fixed[f"hold_{k}"][base_idx]
    base["trail"] = np.full(len(base_idx), np.nan)
    for j, i in enumerate(base_idx):
        if j % BASELINE_SAMPLE_STEP == 0:
            base["trail"][j] = _trail_return(i, opens, closes, lows)
    return rows, pd.DataFrame(base)


def main() -> None:
    tickers = [t for t in load_tickers(VOLUME_SPIKE_TICKERS_URL) if not t.endswith(".NE")]
    print(f"Universe: {len(tickers)} tickers  |  window {START} .. {END}  |  {N_FOLDS} folds")
    t0 = time.perf_counter()
    data = sync_and_load(tickers, start=FETCH_START, end=END, quiet=True)
    print(f"Daily bars loaded in {time.perf_counter() - t0:.1f}s")

    event_rows, base_frames = [], []
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < BREAKOUT_LOOKBACK + 50:
            continue
        rows, base = _ticker_frames(t, df)
        event_rows.extend(rows)
        base_frames.append(base)
    events = pd.DataFrame(event_rows)
    base = pd.concat(base_frames, ignore_index=True)

    bounds = pd.date_range(start=START, end=END, periods=N_FOLDS + 1)
    events["fold"] = np.clip(np.searchsorted(bounds, events["date"], side="right") - 1, 0, N_FOLDS - 1)
    base["fold"] = np.clip(np.searchsorted(bounds, base["date"], side="right") - 1, 0, N_FOLDS - 1)

    out = []
    for variant in events["variant"].unique():
        ev = events[events["variant"] == variant]
        for ex in EXITS:
            r = ev[ex].dropna()
            if r.empty:
                continue
            diffs = []
            for f in range(N_FOLDS):
                e = ev.loc[ev["fold"] == f, ex].dropna()
                b = base.loc[base["fold"] == f, ex].dropna()
                if len(e) >= 3 and len(b):
                    diffs.append(e.mean() - b.mean())
            t_stat, p_val = stats.ttest_1samp(diffs, 0.0) if len(diffs) >= 2 else (np.nan, np.nan)
            b_all = base[ex].dropna()
            out.append({
                "variant": variant, "exit": ex, "n": len(r),
                "mean%": round(r.mean() * 100, 2), "median%": round(r.median() * 100, 2),
                "win%": round((r > 0).mean() * 100, 1),
                "base_mean%": round(b_all.mean() * 100, 2), "base_median%": round(b_all.median() * 100, 2),
                "folds+": f"{sum(d > 0 for d in diffs)}/{len(diffs)}",
                "t": round(t_stat, 2), "p": round(p_val, 3),
            })
    summary = pd.DataFrame(out)
    print()
    print(tabulate(summary, headers="keys", tablefmt="simple", showindex=False))
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    events[events["variant"] != "current_rule"].to_csv(EVENTS_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV} and {EVENTS_CSV}")


if __name__ == "__main__":
    main()
