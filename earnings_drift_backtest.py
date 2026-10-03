"""
earnings_drift_backtest.py
===========================
Research question (2026-10, prompted by MATR.TO: its May-14-2026 +27% /
5x-volume day was the morning after a +136% EPS beat, and the stock kept
climbing for months): does post-earnings-announcement drift (PEAD) exist
on the TSX universe -- and is a volume spike ON an earnings report worth
more than a volume spike on any other day? The earlier volume-only tests
(volume_breakout_backtest.py, volume_persistence_backtest.py) mixed both
kinds of spike together.

Earnings data: LiveDataProvider.get_earnings_dates() (Yahoo), cached to
EARNINGS_CACHE_CSV so re-runs don't refetch. Yahoo has little or nothing
for most TSXV/CSE juniors, so only tickers with >= MIN_REPORTS reports in
the window are tested (the "covered" universe -- baseline included).

Event timing, with no lookahead: report date d (Yahoo's timestamp is
often only approximate about before-open vs after-close), so the
reaction window is trading days d and d+1 and the trade is bought at the
OPEN of d+2. Everything a group condition uses is known by d+1's close:
  surprise   Yahoo's Surprise(%) = (reported - estimate) / |estimate|
  reaction   close(d+1) / close(d-1) - 1
  volx       max volume of d, d+1 / mean volume of the prior BASE_VOL_DAYS
Groups are fixed in advance from the PEAD literature, not tuned here:
  earn_all          every covered report
  beat / miss       surprise > 0 / < 0
  big_beat          surprise >= BIG_SURPRISE_PCT
  react_up / down   reaction >= +REACT_PCT / <= -REACT_PCT
  beat_up_vol       beat AND react_up AND volx >= VOLX_MIN  (the MATR shape)
Spike comparison (same covered tickers): a "spike day" t is volume >=
SPIKE_VOLX x baseline AND close >= +REACT_PCT vs the prior close (one per
ticker per SPIKE_COOLDOWN days), bought at t+1's open, split by whether t
falls in an earnings reaction window (d or d+1) or not.

Exits, liquidity filter, baseline (random eligible days), 16 chronological
folds and the per-fold (event - baseline) t-test all reuse
volume_breakout_backtest.py.

SURVIVORSHIP: today's ticker list only -- see volume_breakout_backtest.py.

Usage:
    python earnings_drift_backtest.py
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH, VOLUME_SPIKE_TICKERS_URL
from market_data import LiveDataProvider
from market_data_cache import sync_and_load
from research.triple_screen.batch import load_tickers
from volume_breakout_backtest import (BASE_VOL_DAYS, BASELINE_SAMPLE_STEP, END, FETCH_START, HOLD_DAYS,
                                      MIN_DOLLAR_VOL, MIN_PRICE, N_FOLDS, ROUND_TRIP_COST, START,
                                      _trail_return)

MIN_REPORTS = 4
BIG_SURPRISE_PCT = 20.0
REACT_PCT = 5.0
VOLX_MIN = 2.0
SPIKE_VOLX = 3.0
SPIKE_COOLDOWN = 20
FETCH_WORKERS = 4

EARNINGS_CACHE_CSV = OUT_PATH / "earnings_dates_cache.csv"
SUMMARY_CSV = OUT_PATH / "earnings_drift_summary.csv"
EVENTS_CSV = OUT_PATH / "earnings_drift_events.csv"
EXITS = [f"hold_{k}" for k in HOLD_DAYS] + ["trail"]
GROUPS = ["earn_all", "beat", "big_beat", "miss", "react_up", "react_down", "beat_up_vol",
          "spike_on_earnings", "spike_no_earnings"]


def _fetch_one(ticker: str):
    try:
        d = LiveDataProvider().get_earnings_dates(ticker, limit=40)
    except Exception:
        d = None
    if d is None or d.empty:
        return pd.DataFrame({"ticker": [ticker], "report_ts": [pd.NaT]})
    d = d.reset_index()
    return pd.DataFrame({
        "ticker": ticker,
        "report_ts": pd.to_datetime(d.iloc[:, 0], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None),
        "eps_est": d["EPS Estimate"].to_numpy(),
        "eps_rep": d["Reported EPS"].to_numpy(),
        "surprise_pct": d["Surprise(%)"].to_numpy(),
    })


def load_earnings(tickers: list[str]) -> pd.DataFrame:
    cached = pd.read_csv(EARNINGS_CACHE_CSV, parse_dates=["report_ts"]) if EARNINGS_CACHE_CSV.exists() else pd.DataFrame(
        columns=["ticker", "report_ts", "eps_est", "eps_rep", "surprise_pct"])
    missing = sorted(set(tickers) - set(cached["ticker"]))
    if missing:
        print(f"Fetching earnings dates for {len(missing)} tickers ...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(FETCH_WORKERS) as ex:
            new = list(ex.map(_fetch_one, missing))
        cached = pd.concat([cached, *new], ignore_index=True)
        OUT_PATH.mkdir(parents=True, exist_ok=True)
        cached.to_csv(EARNINGS_CACHE_CSV, index=False)
        print(f"  done in {time.perf_counter() - t0:.0f}s")
    e = cached.dropna(subset=["report_ts", "eps_rep"]).copy()
    e["report_ts"] = pd.to_datetime(e["report_ts"])
    return e[e["ticker"].isin(tickers)]


def _ticker(ticker: str, df: pd.DataFrame, earn: pd.DataFrame):
    df = df.sort_index()
    o, l, c = df["Open"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    v = df["Volume"].astype(float)
    base = v.rolling(BASE_VOL_DAYS, min_periods=int(BASE_VOL_DAYS * 0.8)).mean().shift(1).to_numpy()
    dollar = (df["Close"] * v).rolling(BASE_VOL_DAYS, min_periods=int(BASE_VOL_DAYS * 0.8)).mean().shift(1)
    eligible = ((dollar >= MIN_DOLLAR_VOL) & (df["Close"] >= MIN_PRICE)).to_numpy()
    in_window = np.asarray(df.index >= pd.Timestamp(START))
    v_a = v.to_numpy()
    n = len(c)
    entry = np.r_[o[1:], np.nan]
    fixed = {k: np.where(entry > 0, np.r_[c[k:], np.full(k, np.nan)] / entry - 1 - ROUND_TRIP_COST, np.nan)
             for k in HOLD_DAYS}

    def returns(i: int) -> dict:
        r = {f"hold_{k}": fixed[k][i] for k in HOLD_DAYS}
        r["trail"] = _trail_return(i, o, c, l)
        return r

    rows, react_days = [], set()
    dates = df.index
    for _, e in earn.iterrows():
        i0 = int(np.searchsorted(dates, e["report_ts"].normalize()))
        i1 = i0 + 1
        if i0 < 1 or i1 >= n:
            continue
        react_days.update((i0, i1))
        if not (eligible[i0 - 1] and in_window[i0]) or not base[i0] > 0:
            continue
        reaction = (c[i1] / c[i0 - 1] - 1) * 100
        volx = max(v_a[i0], v_a[i1]) / base[i0]
        s = e["surprise_pct"]
        tags = ["earn_all"]
        if s > 0:
            tags.append("beat")
        if s >= BIG_SURPRISE_PCT:
            tags.append("big_beat")
        if s < 0:
            tags.append("miss")
        if reaction >= REACT_PCT:
            tags.append("react_up")
        if reaction <= -REACT_PCT:
            tags.append("react_down")
        if s > 0 and reaction >= REACT_PCT and volx >= VOLX_MIN:
            tags.append("beat_up_vol")
        r = returns(i1)
        for g in tags:
            rows.append({"ticker": ticker, "date": dates[i1], "group": g, "surprise_pct": s,
                         "reaction_pct": reaction, "volx": volx, **r})

    up = np.r_[False, c[1:] / c[:-1] - 1 >= REACT_PCT / 100]
    spike = up & (v_a >= SPIKE_VOLX * base) & eligible & in_window
    last = -10**9
    for t in np.flatnonzero(spike):
        if t - last < SPIKE_COOLDOWN:
            continue
        last = t
        g = "spike_on_earnings" if t in react_days else "spike_no_earnings"
        rows.append({"ticker": ticker, "date": dates[t], "group": g, "volx": v_a[t] / base[t], **returns(t)})

    brows = []
    for j, i in enumerate(np.flatnonzero(eligible & in_window)):
        b = {"date": dates[i], **{f"hold_{k}": fixed[k][i] for k in HOLD_DAYS}}
        b["trail"] = _trail_return(i, o, c, l) if j % BASELINE_SAMPLE_STEP == 0 else np.nan
        brows.append(b)
    return rows, brows


def main() -> None:
    tickers = [t for t in load_tickers(VOLUME_SPIKE_TICKERS_URL) if not t.endswith(".NE")]
    print(f"Universe: {len(tickers)} tickers  |  window {START} .. {END}")
    earnings = load_earnings(tickers)
    in_win = earnings[earnings["report_ts"] >= pd.Timestamp(START)]
    counts = in_win.groupby("ticker").size()
    covered = sorted(counts[counts >= MIN_REPORTS].index)
    print(f"Covered (>= {MIN_REPORTS} reports since {START}): {len(covered)} tickers, "
          f"{int(counts[counts >= MIN_REPORTS].sum())} reports")
    data = sync_and_load(covered, start=FETCH_START, end=END, quiet=True)

    ev, bs = [], []
    for t in covered:
        df = data.get(t)
        if df is None or len(df) < BASE_VOL_DAYS + 50:
            continue
        r, b = _ticker(t, df, in_win[in_win["ticker"] == t])
        ev.extend(r)
        bs.extend(b)
    ev, bs = pd.DataFrame(ev), pd.DataFrame(bs)

    bounds = pd.date_range(start=START, end=END, periods=N_FOLDS + 1)
    for f in (ev, bs):
        f["fold"] = np.clip(np.searchsorted(bounds, f["date"], side="right") - 1, 0, N_FOLDS - 1)
    base_fold = {ex: bs.groupby("fold")[ex].mean() for ex in EXITS}

    out = []
    for g in GROUPS:
        e = ev[ev["group"] == g]
        if e.empty:
            continue
        for ex in EXITS:
            r = e[ex].dropna()
            ef = e.groupby("fold")[ex].agg(["mean", "count"])
            ef = ef[ef["count"] >= 3]
            diffs = (ef["mean"] - base_fold[ex].reindex(ef.index)).dropna()
            t_stat, p_val = stats.ttest_1samp(diffs, 0.0) if len(diffs) >= 2 else (np.nan, np.nan)
            srt = r.sort_values(ascending=False)
            out.append({
                "group": g, "exit": ex, "n": len(r), "mean%": round(r.mean() * 100, 2),
                "mean_ex_top5%": round(srt.iloc[5:].mean() * 100, 2) if len(srt) > 5 else np.nan,
                "median%": round(r.median() * 100, 2), "win%": round((r > 0).mean() * 100, 1),
                "base_mean%": round(bs[ex].dropna().mean() * 100, 2),
                "folds+": f"{int((diffs > 0).sum())}/{len(diffs)}", "t": round(t_stat, 2), "p": round(p_val, 3),
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
