"""
Barometer Phase 0 -- the stock-day panel: past-only features, future-only
labels (see PLAN.md for the targets and known biases).

Every feature on row (ticker, t) is computed from bars at or before t's
close; every label from bars t+1..t+HORIZON. Both properties are locked by
tests/test_barometer_dataset.py and re-checked on real data by
`--check-lookahead` (rebuild on data cut at a date, compare). The one
deliberate exception is days_to_earnings -- see its docstring.

Each ticker is aligned to the benchmark's (XIU.TO's) trading calendar first,
so every rolling window and every H-session label spans the same sessions
for every ticker. A missing session is forward-filled as a flat, zero-volume
bar for at most MAX_FFILL_SESSIONS; a longer gap (halt, delisting) leaves
NaN, which drops the row's features and the labels whose window crosses it.

Reuses auto_pipeline's own Wilder ATR (_atr) and PipelineConfig's stop
parameters, so the stop behind y_stop_hit is exactly the one the live core
pipeline would set (compute_levels(); parity is a test).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from auto_pipeline import PipelineConfig, _atr

BENCHMARK = "XIU.TO"
HORIZON = 20                      # trading days ahead the labels look
MIN_PRICE = 2.0                   # liquidity floor: close at t
MIN_DOLLAR_VOL_20 = 1_000_000.0   # liquidity floor: 20-day avg close*volume through t
MAX_FFILL_SESSIONS = 5            # longest missing-session run bridged as a flat bar
EARNINGS_CLIP = 30                # days_to_earnings is clipped here (see its docstring)
UP_DOWN_THRESHOLD = 0.10          # y_up10 / y_dn10

# compute_levels() hard-codes these two (not on PipelineConfig); mirrored
# here and locked by the parity test against compute_levels() itself.
SWING_LOW_BARS = 20
SWING_LOW_BUFFER = 0.99

PANEL_START = "2020-09-01"        # first cached bar (features need ~1y warm-up)
PANEL_END = "2026-10-02"          # last cached bar at Phase 0 time
PANEL_PATH = Path(__file__).resolve().parents[2] / "cache" / "barometer_panel.pkl"
EARNINGS_CSV = Path(__file__).resolve().parents[2] / "out" / "earnings_dates_cache.csv"

FEATURE_COLS: List[str] = [
    "ret_5", "ret_21", "ret_63", "mom_12_1",
    "vol_5", "vol_20", "vol_60", "atr_pct", "stop_dist_pct",
    "dist_52w_high", "dist_52w_low",
    "vol_ratio_20", "vol_ratio_100", "gap_abs_20", "beta_60",
]
MARKET_COLS: List[str] = ["mkt_above_200", "mkt_vol_20", "mkt_ret_21"]
OPTIONAL_COLS: List[str] = ["days_to_earnings"]
LABEL_COLS: List[str] = ["fwd_ret", "fwd_rvol", "y_stop_hit", "y_up10", "y_dn10", "y_beat_mkt"]


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks (all per ticker, on the benchmark calendar)
# ─────────────────────────────────────────────────────────────────────────────

def align_to_calendar(df: pd.DataFrame, calendar: pd.DatetimeIndex,
                      max_ffill: int = MAX_FFILL_SESSIONS) -> pd.DataFrame:
    """Reindex OHLCV onto `calendar`. A missing session becomes a flat bar at
    the last close with zero volume, for at most `max_ffill` sessions in a
    row; beyond that (and before the first bar) everything is NaN.
    Adds a boolean `traded` column (a real bar existed that session)."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = df.reindex(calendar)
    traded = out["Close"].notna()
    out["Close"] = out["Close"].ffill(limit=max_ffill)
    for col in ("Open", "High", "Low"):
        out[col] = out[col].where(traded, out["Close"])
    out["Volume"] = out["Volume"].where(traded, 0.0).where(out["Close"].notna())
    out["traded"] = traded
    return out


def compute_stop(px: pd.DataFrame) -> pd.Series:
    """Vectorised auto_pipeline.compute_levels() stop, with entry = close at t:
    max(min(close - atr_stop_mult*ATR, 20-bar low * 0.99), close * (1 - max_stop_pct/100))."""
    cfg = PipelineConfig()
    c, h, l = px["Close"], px["High"], px["Low"]
    atr = _atr(h, l, c, cfg.atr_period)
    atr_stop = c - cfg.atr_stop_mult * atr
    swing_low = l.rolling(SWING_LOW_BARS).min() * SWING_LOW_BUFFER
    raw_stop = np.minimum(atr_stop, swing_low)
    min_stop = c * (1 - cfg.max_stop_pct / 100)
    return np.maximum(raw_stop, min_stop)


def compute_features(px: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    """Per-ticker features; row t uses bars <= t only."""
    c, h, l, o, v = px["Close"], px["High"], px["Low"], px["Open"], px["Volume"]
    lr = np.log(c).diff()
    blr = np.log(bench_close).diff()
    f = pd.DataFrame(index=px.index)
    f["ret_5"] = c / c.shift(5) - 1
    f["ret_21"] = c / c.shift(21) - 1
    f["ret_63"] = c / c.shift(63) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["vol_5"] = lr.rolling(5).std()
    f["vol_20"] = lr.rolling(20).std()
    f["vol_60"] = lr.rolling(60).std()
    f["atr_pct"] = _atr(h, l, c, PipelineConfig().atr_period) / c
    f["stop_dist_pct"] = compute_stop(px) / c - 1
    f["dist_52w_high"] = c / h.rolling(252).max() - 1
    f["dist_52w_low"] = c / l.rolling(252).min() - 1
    f["vol_ratio_20"] = v / v.shift(1).rolling(20).mean()
    f["vol_ratio_100"] = v / v.shift(1).rolling(100).mean()
    f["gap_abs_20"] = (o / c.shift(1) - 1).abs().rolling(20).mean()
    f["beta_60"] = lr.rolling(60).cov(blr) / blr.rolling(60).var()
    f["dollar_vol_20"] = (c * v).rolling(20).mean()
    return f.replace([np.inf, -np.inf], np.nan)


def compute_market_features(bench_close: pd.Series) -> pd.DataFrame:
    """Benchmark-level context, identical for every ticker on a date."""
    blr = np.log(bench_close).diff()
    sma200 = bench_close.rolling(200).mean()
    m = pd.DataFrame(index=bench_close.index)
    m["mkt_above_200"] = (bench_close > sma200).astype(float).where(sma200.notna())
    m["mkt_vol_20"] = blr.rolling(20).std()
    m["mkt_ret_21"] = bench_close / bench_close.shift(21) - 1
    return m


def _forward_window(s: pd.Series, horizon: int, fn: str) -> pd.Series:
    """`fn` ('min' or 'std') over s[t+1 .. t+horizon]; NaN unless all present."""
    rev = s.iloc[::-1].rolling(horizon, min_periods=horizon)
    agg = getattr(rev, fn)().iloc[::-1]
    return agg.shift(-1)


def compute_labels(px: pd.DataFrame, bench_close: pd.Series, stop: pd.Series,
                   horizon: int = HORIZON) -> pd.DataFrame:
    """Labels from bars t+1..t+horizon only (entry = close at t)."""
    c = px["Close"]
    lab = pd.DataFrame(index=px.index)
    fwd = c.shift(-horizon) / c - 1
    bench_fwd = bench_close.shift(-horizon) / bench_close - 1
    fut_min = _forward_window(c, horizon, "min")
    fut_rvol = _forward_window(np.log(c).diff(), horizon, "std")
    ok = fwd.notna() & fut_min.notna() & stop.notna()
    lab["fwd_ret"] = fwd.where(ok)
    lab["fwd_rvol"] = fut_rvol.where(ok)
    lab["y_stop_hit"] = (fut_min < stop).astype(float).where(ok)
    lab["y_up10"] = (fwd > UP_DOWN_THRESHOLD).astype(float).where(ok)
    lab["y_dn10"] = (fwd < -UP_DOWN_THRESHOLD).astype(float).where(ok)
    lab["y_beat_mkt"] = (fwd > bench_fwd).astype(float).where(ok & bench_fwd.notna())
    return lab


def days_to_earnings(calendar: pd.DatetimeIndex, report_dates: List[pd.Timestamp],
                     clip: int = EARNINGS_CLIP) -> pd.Series:
    """Sessions from t to the next report date strictly after t (1 = the next
    session), clipped at `clip`; NaN when no later report date is known.

    Deliberate, documented lookahead: this uses the ACTUAL report date. Firms
    usually announce it a few weeks ahead, so it is near-known within ~30
    sessions but not beyond -- hence the clip, its exclusion from the
    truncation check, and PLAN.md's with/without comparison in Phase 3."""
    out = pd.Series(np.nan, index=calendar)
    if not report_dates:
        return out
    cal_days = calendar.normalize().values
    rep = np.sort(pd.DatetimeIndex(report_dates).normalize().values)
    rep_pos = np.searchsorted(cal_days, rep, side="left")   # first session on/after report day
    nxt = np.searchsorted(rep, cal_days, side="right")      # next report strictly after t
    have = nxt < len(rep)
    pos = np.arange(len(calendar))
    sess = np.full(len(calendar), np.nan)
    sess[have] = rep_pos[nxt[have]] - pos[have]
    return pd.Series(np.minimum(sess, clip), index=calendar)


def load_earnings_dates(path: Path = EARNINGS_CSV) -> Dict[str, List[pd.Timestamp]]:
    """{ticker: [report dates]} from earnings_drift_backtest.py's cache CSV."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, usecols=["ticker", "report_ts"]).dropna()
    df["d"] = pd.to_datetime(df["report_ts"].str[:10], errors="coerce")
    df = df.dropna(subset=["d"])
    return {t: sorted(g["d"].unique()) for t, g in df.groupby("ticker")}


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

def build_ticker_rows(ticker: str, df: pd.DataFrame, bench_close: pd.Series,
                      earnings: Optional[List[pd.Timestamp]] = None,
                      horizon: int = HORIZON) -> pd.DataFrame:
    """All panel rows for one ticker: liquid sessions with complete features."""
    px = align_to_calendar(df, bench_close.index)
    feats = compute_features(px, bench_close)
    labels = compute_labels(px, bench_close, compute_stop(px), horizon)
    liquid = (px["traded"] & (px["Close"] >= MIN_PRICE)
              & (feats["dollar_vol_20"] >= MIN_DOLLAR_VOL_20))
    keep = liquid & feats[FEATURE_COLS].notna().all(axis=1)
    rows = feats.loc[keep].join(labels.loc[keep])
    rows["days_to_earnings"] = days_to_earnings(bench_close.index, earnings or [])[keep]
    rows["close"] = px["Close"][keep]
    rows.insert(0, "ticker", ticker)
    return rows


def build_panel(data: Dict[str, pd.DataFrame], benchmark: str = BENCHMARK,
                earnings: Optional[Dict[str, List[pd.Timestamp]]] = None,
                horizon: int = HORIZON) -> pd.DataFrame:
    """Long panel indexed by (date, ticker); the benchmark itself gets no rows."""
    bench = data[benchmark].sort_index()
    bench = bench[~bench.index.duplicated(keep="last")]
    bench_close = bench["Close"]
    earnings = earnings or {}
    frames = [build_ticker_rows(t, df, bench_close, earnings.get(t), horizon)
              for t, df in data.items() if t != benchmark and df is not None and not df.empty]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.join(compute_market_features(bench_close), on="date")
    return panel.reset_index().set_index(["date", "ticker"]).sort_index()


def load_cached_data(start: str = PANEL_START, end: str = PANEL_END) -> Dict[str, pd.DataFrame]:
    """Raw-list tickers + benchmark straight from data/market_cache.db --
    load_from_cache(), never sync_and_load(): no Yahoo calls (see the
    yahoo-rate-limit discipline)."""
    from config import BACKTEST_RAW_TICKERS_URL
    from market_data_cache import load_from_cache
    from swing_tickers import read_tickers
    tickers = list(dict.fromkeys(read_tickers(BACKTEST_RAW_TICKERS_URL) + [BENCHMARK]))
    return load_from_cache(tickers, start=start, end=end)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _summary(panel: pd.DataFrame) -> None:
    dates = panel.index.get_level_values("date")
    print(f"rows {len(panel):,}  tickers {panel.index.get_level_values('ticker').nunique()}  "
          f"dates {dates.min().date()} .. {dates.max().date()}")
    by_year = panel.groupby(dates.year)
    print("year   rows    tickers/day  stop_hit  up10   dn10   beat_mkt  labelled")
    for y, g in by_year:
        per_day = g.groupby(level="date").size().mean()
        lab = g["y_stop_hit"].notna().mean()
        print(f"{y}  {len(g):7,}   {per_day:6.0f}      {g['y_stop_hit'].mean():.3f}   "
              f"{g['y_up10'].mean():.3f}  {g['y_dn10'].mean():.3f}  {g['y_beat_mkt'].mean():.3f}     {lab:.0%}")
    print(f"days_to_earnings known on {panel['days_to_earnings'].notna().mean():.0%} of rows")


def check_lookahead(data: Dict[str, pd.DataFrame], cut: str, horizon: int = HORIZON) -> bool:
    """Rebuild on data cut at `cut` and compare with the full build: every
    row and feature on dates <= cut must be identical, and every label whose
    window ends by `cut` too. Returns True on a clean pass."""
    cut_ts = pd.Timestamp(cut)
    full = build_panel(data, horizon=horizon)
    trunc = build_panel({t: df.loc[:cut_ts] for t, df in data.items()}, horizon=horizon)
    f_old = full[full.index.get_level_values("date") <= cut_ts]
    ok = True
    if not f_old.index.equals(trunc.index):
        print(f"FAIL row sets differ: full {len(f_old)} vs truncated {len(trunc)}")
        return False
    cols = FEATURE_COLS + MARKET_COLS
    diff = (f_old[cols] - trunc[cols]).abs().max().max()
    print(f"rows compared {len(trunc):,}; max |feature diff| = {diff:.3g}")
    ok &= bool(diff < 1e-9) or bool(np.isnan(diff))
    sessions = data[BENCHMARK].sort_index().loc[:cut_ts].index
    label_end = sessions[-horizon - 1]
    lf = f_old[f_old.index.get_level_values("date") <= label_end][LABEL_COLS]
    lt = trunc[trunc.index.get_level_values("date") <= label_end][LABEL_COLS]
    ldiff = (lf - lt).abs().max().max()
    tail_nan = trunc[trunc.index.get_level_values("date") > label_end]["y_stop_hit"].notna().sum()
    print(f"labels compared through {label_end.date()}: max |diff| = {ldiff:.3g}; "
          f"labelled rows in the last {horizon} sessions before the cut: {tail_nan} (must be 0)")
    ok &= (bool(ldiff < 1e-9) or bool(np.isnan(ldiff))) and tail_nan == 0
    print("PASS" if ok else "FAIL")
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Barometer Phase 0 panel builder")
    ap.add_argument("--check-lookahead", metavar="CUT_DATE", nargs="?", const="2024-06-28",
                    help="rebuild on data cut at CUT_DATE and verify nothing before it changes")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    data = load_cached_data()
    print(f"loaded {len(data)} cached tickers in {time.perf_counter() - t0:.0f}s")
    # The panel runs on the benchmark's calendar, so a stale benchmark silently
    # truncates every ticker (seen 2026-10-12: XIU.TO cached to 08-26, the rest to 10-02).
    bench_end = data[BENCHMARK].index.max()
    latest = pd.Series({t: df.index.max() for t, df in data.items() if not df.empty}).median()
    if bench_end < latest - pd.Timedelta(days=5):
        print(f"WARNING: {BENCHMARK} cached only to {bench_end.date()} vs median ticker "
              f"{latest.date()} -- the panel ends there; top up {BENCHMARK} in the cache")
    if args.check_lookahead:
        return 0 if check_lookahead(data, args.check_lookahead) else 1
    panel = build_panel(data, earnings=load_earnings_dates())
    _summary(panel)
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(PANEL_PATH)
    print(f"saved {PANEL_PATH} ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
