"""
VCP score backtest: does a high score lead to better forward returns?

Weekly, every liquid name is scored with data up to that date only (vcp.py's
own score_at), then followed forward. Rules, groups and the pass rule are fixed
in PLAN.md.

    python -m research.vcp.backtest_vcp --phase dev          # data cut before HOLDOUT
    python -m research.vcp.backtest_vcp --phase sensitivity  # same cut, +/-20% variants
    python -m research.vcp.backtest_vcp --phase holdout      # only after the dev verdict exists

Prices come from cache/vcp_prices.db (prices.py); nothing is downloaded here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from canadian_stock_screener import CONFIG as SCREENER_CONFIG, ScoreCalculator
from research.vcp import prices, vcp
from research.vcp.vcp import DEFAULT, Bars, VcpConfig

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "vcp"
PANEL_DIR = Path(__file__).resolve().parents[2] / "cache"          # gitignored
HOLDOUT = pd.Timestamp("2025-05-20")
HORIZONS = (20, 60, 120)
MAIN_H = 60
ROUND_TRIP_COST = 0.005
MAX_ENTRY_DELAY = 5                # sessions to wait for the first trade after the signal
BREAKOUT_DAYS = 30
BREAKOUT_VOL_MULT = 1.4
COOLDOWN = 60                      # sessions between separate events of one ticker in one group
BLOCK_WEEKS = 13
N_BOOT = 5000
HIGH_SCORE = 71

# A split Yahoo left unadjusted (PLAN.md, Phase 4): up this much in a day, on
# this little of the usual volume, in a name that really trades.
SPLIT_MIN_JUMP, SPLIT_MAX_VOLUME, SPLIT_MIN_DOLLAR_VOL, SPLIT_DAYS = 2.0, 0.5, 100_000.0, 20

BUCKETS = (("1-40", 1, 40), ("41-70", 41, 70), ("71-85", 71, 85), ("86-100", 86, 100))

# Pass rule (PLAN.md)
MIN_EVENTS, MIN_TICKERS, MIN_YEAR_SHARE, MIN_SENS_SHARE = 100, 50, 0.60, 0.80


@dataclass
class Market:
    calendar: pd.DatetimeIndex
    bench: Bars
    bars: Dict[str, Bars]


# ── Preparing ────────────────────────────────────────────────────────────────

def fix_split_errors(bars: Bars) -> Bars:
    """Put the bars before an unadjusted reverse split on the later price scale."""
    for k in range(1, len(bars)):
        jump = bars.close[k] / bars.close[k - 1]
        if jump < SPLIT_MIN_JUMP:
            continue
        before = slice(max(0, k - SPLIT_DAYS), k)
        usual = bars.volume[before].mean()
        if (bars.volume[k] < SPLIT_MAX_VOLUME * usual
                and (bars.close[before] * bars.volume[before]).mean() >= SPLIT_MIN_DOLLAR_VOL):
            f = np.where(np.arange(len(bars)) < k, jump, 1.0)
            bars = replace(bars, open=bars.open * f, high=bars.high * f, low=bars.low * f,
                           close=bars.close * f, volume=bars.volume / f)
    return bars


def build_market(data: Dict[str, pd.DataFrame], cut: Optional[pd.Timestamp] = None) -> Market:
    """Clean bars on the benchmark's calendar; with `cut`, only dates before it."""
    if cut is not None:
        data = {t: d[d.index < cut] for t, d in data.items()}
    bench_df = data[prices.BENCHMARK]
    calendar = bench_df.index.sort_values().unique()
    bars = {t: fix_split_errors(vcp.clean_bars(d, calendar)) for t, d in data.items() if t != prices.BENCHMARK and len(d)}
    return Market(calendar, vcp.clean_bars(bench_df, calendar), {t: b for t, b in bars.items() if len(b)})


def weekly_sessions(calendar: pd.DatetimeIndex, start_session: int) -> List[int]:
    """Positions of the last session of each week, from start_session on."""
    weeks = calendar.to_period("W-FRI")
    last = pd.Series(np.arange(len(calendar))).groupby(weeks.asi8).max()
    return [int(s) for s in last if s >= start_session]


def bar_at(bars: Bars, session: int, max_stale: int) -> int:
    """Index of the last trade on or before `session`, or -1 if too old."""
    i = int(np.searchsorted(bars.session, session, side="right")) - 1
    return i if i >= 0 and session - bars.session[i] <= max_stale else -1


# ── Scoring ──────────────────────────────────────────────────────────────────

_FIELDS = ("vcp_score", "valid_vcp", "gate_passed", "gate_fail_reason", "no_score_reason",
           "n_contractions", "final_depth_pct", "vol_dryup_ratio", "pivot_price", "pct_from_pivot",
           "base_length_days", "base_depth_pct", "base_start")


def score_panel(market: Market, sessions: List[int], cfg: VcpConfig = DEFAULT) -> pd.DataFrame:
    """One row per liquid ticker per scoring session."""
    gates = {t: vcp.gate_table(b, cfg) for t, b in market.bars.items() if len(b) > cfg.year_days}
    rows: List[Dict[str, Any]] = []
    for s in sessions:
        at: Dict[str, int] = {}
        for t, g in gates.items():
            i = bar_at(market.bars[t], s, cfg.max_stale_sessions)
            if i >= 0 and g["ready"].iat[i] and vcp.is_liquid(g, i):
                at[t] = i
        ranks = vcp.rs_percentiles(pd.Series({t: gates[t]["rs_raw"].iat[i] for t, i in at.items()}, dtype=float))
        for t, i in at.items():
            r = vcp.score_at(market.bars[t], gates[t], i, ranks[t], cfg)
            rows.append({"ticker": t, "session": s, "date": market.calendar[s], "bar": i,
                         "close": market.bars[t].close[i], "rs_percentile": ranks[t],
                         **{k: r[k] for k in _FIELDS}})
    return pd.DataFrame(rows)


def stage2_flags(market: Market, panel: pd.DataFrame) -> pd.Series:
    """score_stage2 > 0 on closes up to each row's bar (the repo's own Stage II)."""
    calc = ScoreCalculator(SCREENER_CONFIG)
    out = []
    for t, i in zip(panel["ticker"], panel["bar"]):
        b = market.bars[t]
        lo = max(0, i - 299)
        out.append(calc.score_stage2(pd.Series(b.close[lo:i + 1], index=pd.DatetimeIndex(b.dates[lo:i + 1]))) > 0)
    return pd.Series(out, index=panel.index, dtype=bool)


# ── Forward outcomes ─────────────────────────────────────────────────────────

def entry_bar(bars: Bars, session: int) -> int:
    """First trade after `session`, within MAX_ENTRY_DELAY sessions; else -1."""
    e = int(np.searchsorted(bars.session, session, side="right"))
    return e if e < len(bars) and bars.session[e] - session <= MAX_ENTRY_DELAY else -1


def forward(bars: Bars, e: int, horizon: int, n_sessions: int) -> Tuple[float, float, bool]:
    """(return, worst low, stopped trading) from bar e's open to the close
    `horizon` sessions later. NaN if that session hasn't happened yet."""
    target = bars.session[e] + horizon
    if target >= n_sessions:
        return np.nan, np.nan, False
    x = int(np.searchsorted(bars.session, target, side="right")) - 1
    worst = bars.low[e:x + 1].min() / bars.open[e] - 1.0
    ended = x == len(bars) - 1 and target - bars.session[x] > MAX_ENTRY_DELAY
    return float(bars.close[x] / bars.open[e] - 1.0), float(worst), bool(ended)


def outcomes(market: Market, keys: pd.DataFrame) -> pd.DataFrame:
    """Forward returns of every (ticker, session) row and of XIU.TO over the same days."""
    n = len(market.calendar)
    rows = []
    for t, s in zip(keys["ticker"], keys["session"]):
        b = market.bars[t]
        e = entry_bar(b, s)
        row: Dict[str, Any] = {"ticker": t, "session": s, "entry_bar": e}
        for h in HORIZONS:
            row[f"ret_{h}"], row[f"mae_{h}"], row[f"ended_{h}"] = \
                forward(b, e, h, n) if e >= 0 else (np.nan, np.nan, False)
            row[f"xiu_{h}"] = np.nan
            if e >= 0 and np.isfinite(row[f"ret_{h}"]):
                row[f"xiu_{h}"] = _bench_return(market, int(b.session[e]), h)
        rows.append(row)
    return pd.DataFrame(rows)


def _bench_return(market: Market, entry_session: int, horizon: int) -> float:
    b = market.bench
    e = int(np.searchsorted(b.session, entry_session, side="left"))
    x = int(np.searchsorted(b.session, entry_session + horizon, side="right")) - 1
    return float(b.close[x] / b.open[e] - 1.0)


def breakout(bars: Bars, i: int, e: int, session: int, pivot: float, n_sessions: int) -> Tuple[float, float]:
    """(broke out 1/0, worst low from entry until then) within BREAKOUT_DAYS
    sessions: a close above the pivot on BREAKOUT_VOL_MULT x the 50-day average
    volume before it. NaN if already above the pivot, not entered, or too recent."""
    if e < 0 or not np.isfinite(pivot) or bars.close[i] > pivot or session + BREAKOUT_DAYS >= n_sessions:
        return np.nan, np.nan
    last = int(np.searchsorted(bars.session, session + BREAKOUT_DAYS, side="right")) - 1
    for k in range(i + 1, last + 1):
        average = bars.volume[max(0, k - DEFAULT.vol_avg_days):k].mean()
        if bars.close[k] > pivot and bars.volume[k] >= BREAKOUT_VOL_MULT * average:
            return 1.0, float(bars.low[e:k + 1].min() / bars.open[e] - 1.0)
    return 0.0, float(bars.low[e:last + 1].min() / bars.open[e] - 1.0)


def add_breakouts(market: Market, panel: pd.DataFrame) -> pd.DataFrame:
    n = len(market.calendar)
    got = [breakout(market.bars[t], i, e, s, p, n) for t, i, e, s, p in       # no pivot if the gate failed
           zip(panel["ticker"], panel["bar"], panel["entry_bar"], panel["session"], panel["pivot_price"])]
    return panel.assign(broke_out=[a for a, _ in got], dd_before_breakout=[b for _, b in got])


# ── Groups and statistics ────────────────────────────────────────────────────

def group_masks(p: pd.DataFrame) -> Dict[str, pd.Series]:
    score = np.floor(p["vcp_score"] + 0.5)
    gate = p["gate_passed"].astype(bool)
    masks = {"0 (gate failed)": ~gate, "0 (gate passed)": gate & (score == 0)}
    for name, lo, hi in BUCKETS:
        masks[name] = gate & (score >= lo) & (score <= hi)
    masks[f"{HIGH_SCORE}+"] = gate & (score >= HIGH_SCORE)
    masks["gate-only"] = gate & ~p["valid_vcp"].astype(bool)
    masks["all gate passers"] = gate
    if "stage2" in p:
        masks["stage 2"] = p["stage2"].astype(bool)
    masks["all liquid"] = pd.Series(True, index=p.index)
    return masks


def separate_events(rows: pd.DataFrame, cooldown: int = COOLDOWN) -> pd.DataFrame:
    """First row of each ticker, then the first at least `cooldown` sessions later, ..."""
    keep = []
    for _, g in rows.sort_values(["ticker", "session"]).groupby("ticker", sort=False):
        last = None
        for idx, s in zip(g.index, g["session"]):
            if last is None or s - last >= cooldown:
                keep.append(idx)
                last = s
    return rows.loc[sorted(keep)]


def same_date_mean(p: pd.DataFrame, mask: pd.Series, col: str) -> pd.Series:
    """For every row, the mean of `col` over the masked rows of its date."""
    return p["session"].map(p.loc[mask].groupby("session")[col].mean())


def block_bootstrap(values: np.ndarray, sessions: np.ndarray, week_of: Dict[int, int],
                    n_boot: int = N_BOOT, seed: int = 0) -> Tuple[float, float]:
    """95% interval of the mean, resampling whole BLOCK_WEEKS-week blocks."""
    ok = np.isfinite(values)
    values, sessions = values[ok], sessions[ok]
    blocks = np.array([week_of[int(s)] // BLOCK_WEEKS for s in sessions])
    ids = np.unique(blocks)
    if len(ids) < 2:
        return np.nan, np.nan
    sums = np.array([values[blocks == b].sum() for b in ids])
    counts = np.array([(blocks == b).sum() for b in ids])
    pick = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def with_excess(p: pd.DataFrame) -> pd.DataFrame:
    """Adds return minus gate-only, all gate passers, stage 2 and XIU.TO, per horizon."""
    masks = group_masks(p)
    out = p.copy()
    for h in HORIZONS:
        r = p[f"ret_{h}"]
        out[f"ex_gate_{h}"] = r - same_date_mean(p, masks["gate-only"], f"ret_{h}")
        out[f"ex_all_{h}"] = r - same_date_mean(p, masks["all gate passers"], f"ret_{h}")
        if "stage 2" in masks:
            out[f"ex_stage2_{h}"] = r - same_date_mean(p, masks["stage 2"], f"ret_{h}")
        out[f"ex_xiu_{h}"] = r - p[f"xiu_{h}"]
    return out


def summary_table(p: pd.DataFrame, h: int, week_of: Dict[int, int], separate: bool) -> pd.DataFrame:
    rows = []
    for name, mask in group_masks(p).items():
        g = p[mask & p[f"ret_{h}"].notna()]
        g = separate_events(g) if separate else g
        if g.empty:
            continue
        lo, hi = block_bootstrap(g[f"ex_gate_{h}"].to_numpy(float), g["session"].to_numpy(), week_of)
        rows.append({
            "group": name, "n": len(g), "tickers": g["ticker"].nunique(),
            "mean": g[f"ret_{h}"].mean(), "mean_net": g[f"ret_{h}"].mean() - ROUND_TRIP_COST,
            "median": g[f"ret_{h}"].median(), "hit": (g[f"ret_{h}"] > 0).mean(),
            "xiu_mean": g[f"xiu_{h}"].mean(),
            "ex_gate": g[f"ex_gate_{h}"].mean(), "ex_gate_lo": lo, "ex_gate_hi": hi,
            "ex_gate_median": g[f"ex_gate_{h}"].median(),
            "ex_all": g[f"ex_all_{h}"].mean(),
            "ex_stage2": g[f"ex_stage2_{h}"].mean() if f"ex_stage2_{h}" in g else np.nan,
            "ex_xiu": g[f"ex_xiu_{h}"].mean(),
            "broke_out": g["broke_out"].mean(), "dd_before_breakout": g["dd_before_breakout"].median(),
            "mae_median": g[f"mae_{h}"].median(), "ended": g[f"ended_{h}"].mean(),
        })
    return pd.DataFrame(rows)


def high_score_events(p: pd.DataFrame, h: int = MAIN_H) -> pd.DataFrame:
    """Separate events of the high-score group that have a return and a same-date baseline."""
    mask = group_masks(p)[f"{HIGH_SCORE}+"] & p[f"ex_gate_{h}"].notna()
    return separate_events(p[mask])


def verdict(p: pd.DataFrame, week_of: Dict[int, int], sensitivity_means: Optional[List[float]] = None) -> Dict[str, Any]:
    """PLAN.md's four conditions on the high-score group at MAIN_H sessions."""
    ev = high_score_events(p)
    ex = ev[f"ex_gate_{MAIN_H}"]
    lo, hi = block_bootstrap(ex.to_numpy(float), ev["session"].to_numpy(), week_of)
    by_year = ex.groupby(pd.DatetimeIndex(ev["date"]).year).mean()
    out: Dict[str, Any] = {
        "events": int(len(ev)), "tickers": int(ev["ticker"].nunique()),
        "mean_excess": float(ex.mean()) if len(ex) else np.nan, "ci": [lo, hi],
        "by_year": {int(y): float(v) for y, v in by_year.items()},
        "years_ahead": float((by_year > 0).mean()) if len(by_year) else np.nan,
    }
    out["c1_positive"] = bool(len(ex) and ex.mean() > 0 and lo > 0)
    out["c2_sample"] = bool(len(ev) >= MIN_EVENTS and out["tickers"] >= MIN_TICKERS)
    out["c3_years"] = bool(len(by_year) and out["years_ahead"] >= MIN_YEAR_SHARE)
    if sensitivity_means is not None:
        out["sens_share"] = float(np.mean(np.array(sensitivity_means) > 0))
        out["c4_sensitivity"] = bool(out["sens_share"] >= MIN_SENS_SHARE)
        out["passed"] = bool(all(out[c] for c in ("c1_positive", "c2_sample", "c3_years", "c4_sensitivity")))
    return out


# ── Sensitivity ──────────────────────────────────────────────────────────────

# Each entry is moved together, by -20% and +20%.
SENSITIVITY_GROUPS: Dict[str, Tuple[str, ...]] = {
    "min_above_low": ("min_above_low",), "max_below_high": ("max_below_high",),
    "min_rs_percentile": ("min_rs_percentile",), "min_dollar_vol": ("min_dollar_vol",),
    "zigzag_pct": ("zigzag_pct",), "max_shrink_ratio": ("max_shrink_ratio",),
    "min_base_days": ("min_base_days",), "max_base_days": ("max_base_days",),
    "atr_ref": ("atr_ref",), "full_shrink": ("full_shrink",),
    "tightness": ("tight_full", "tight_good", "tight_zero"),
    "dryup": ("dryup_full", "dryup_zero"), "near": ("near_full", "near_zero"),
    "max_extended": ("max_extended",), "base_depth": ("base_depth_full", "base_depth_zero"),
    "uptrend_full": ("uptrend_full",), "vol_avg_days": ("vol_avg_days",),
}
WEIGHTS = ("w_progression", "w_tightness", "w_dryup", "w_proximity", "w_base")
EXTRA_VARIANTS = {"every swing": {"merge_swings": False}, "no ATR scaling": {"scale_by_atr": False}}


def _scaled(cfg: VcpConfig, names: Tuple[str, ...], k: float) -> VcpConfig:
    new = {}
    for n in names:
        v = getattr(cfg, n)
        new[n] = int(round(v * k)) if isinstance(v, int) else v * k
    if "max_shrink_ratio" in new:
        new["max_shrink_ratio"] = min(new["max_shrink_ratio"], 1.0)     # never allow growth
    return replace(cfg, **new)


def _reweighted(cfg: VcpConfig, name: str, k: float) -> VcpConfig:
    """One weight moved, all rescaled back to a total of 100."""
    w = {n: getattr(cfg, n) * (k if n == name else 1.0) for n in WEIGHTS}
    total = sum(w.values())
    return replace(cfg, **{n: v * 100.0 / total for n, v in w.items()})


def sensitivity_configs(cfg: VcpConfig = DEFAULT) -> Dict[str, VcpConfig]:
    out = {}
    for label, k in (("-20%", 0.8), ("+20%", 1.2)):
        for name, fields in SENSITIVITY_GROUPS.items():
            out[f"{name} {label}"] = _scaled(cfg, fields, k)
        for name in WEIGHTS:
            out[f"{name} {label}"] = _reweighted(cfg, name, k)
    return out


_SHARED: Dict[str, Any] = {}


def _run_variant(item: Tuple[str, VcpConfig]) -> Dict[str, Any]:
    name, cfg = item
    market, sessions = _SHARED["market"], _SHARED["sessions"]
    panel = score_panel(market, sessions, cfg)
    panel = with_excess(panel.merge(outcomes(market, panel[["ticker", "session"]]), on=["ticker", "session"]))
    ev = high_score_events(panel)
    ex = ev[f"ex_gate_{MAIN_H}"]
    lo, hi = block_bootstrap(ex.to_numpy(float), ev["session"].to_numpy(), _SHARED["week_of"])
    return {"variant": name, "events": len(ev), "mean_excess": ex.mean(), "lo": lo, "hi": hi,
            "median_excess": ex.median()}


def run_sensitivity(market: Market, sessions: List[int], week_of: Dict[int, int],
                    configs: Dict[str, VcpConfig], processes: int = 3) -> pd.DataFrame:
    _SHARED.update(market=market, sessions=sessions, week_of=week_of)
    with mp.get_context("fork").Pool(processes) as pool:
        return pd.DataFrame(pool.map(_run_variant, configs.items(), chunksize=1))


# ── Runs ─────────────────────────────────────────────────────────────────────

def build_panel(market: Market, sessions: List[int], cfg: VcpConfig = DEFAULT) -> pd.DataFrame:
    panel = score_panel(market, sessions, cfg)
    panel["stage2"] = stage2_flags(market, panel)
    panel = panel.merge(outcomes(market, panel[["ticker", "session"]]), on=["ticker", "session"],
                        validate="one_to_one")
    return with_excess(add_breakouts(market, panel))


def scoring_sessions(market: Market, start: Optional[pd.Timestamp] = None) -> Tuple[List[int], Dict[int, int]]:
    sessions = weekly_sessions(market.calendar, DEFAULT.year_days)
    if start is not None:
        sessions = [s for s in sessions if market.calendar[s] >= start]
    return sessions, {s: k for k, s in enumerate(sessions)}


def _fmt(t: pd.DataFrame) -> str:
    t = t.copy()
    for c in t.columns:
        if c not in ("group", "n", "tickers"):
            t[c] = (t[c] * 100).round(2)
    return t.to_string(index=False)


def _report(panel: pd.DataFrame, week_of: Dict[int, int], name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for separate in (False, True):
        for h in HORIZONS:
            t = summary_table(panel, h, week_of, separate)
            kind = "events" if separate else "weekly"
            t.to_csv(OUT_DIR / f"{name}_{kind}_{h}d.csv", index=False)
            print(f"\n== {name}, {kind}, {h} sessions (percent) ==\n{_fmt(t)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="VCP score backtest")
    ap.add_argument("--phase", choices=("dev", "sensitivity", "holdout"), required=True)
    args = ap.parse_args(argv)
    verdict_path = OUT_DIR / "dev_verdict.json"
    if args.phase == "holdout" and not verdict_path.exists():
        print("Write the development verdict first (--phase dev, then --phase sensitivity).")
        return 1
    conn = prices.connect()
    try:
        data = prices.load_all(conn)
    finally:
        conn.close()
    market = build_market(data, cut=None if args.phase == "holdout" else HOLDOUT)
    sessions, week_of = scoring_sessions(market, start=HOLDOUT if args.phase == "holdout" else None)

    if args.phase == "sensitivity":
        res = run_sensitivity(market, sessions, week_of, sensitivity_configs())
        extra = run_sensitivity(market, sessions, week_of,
                                {k: replace(DEFAULT, **v) for k, v in EXTRA_VARIANTS.items()})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        res.to_csv(OUT_DIR / "sensitivity.csv", index=False)
        extra.to_csv(OUT_DIR / "variants.csv", index=False)
        print(res.to_string(index=False), "\n\n", extra.to_string(index=False))
        v = verdict(pd.read_pickle(PANEL_DIR / "vcp_dev_panel.pkl"), week_of, res["mean_excess"].tolist())
        verdict_path.write_text(json.dumps(v, indent=2))
        print(json.dumps(v, indent=2))
        return 0

    panel = build_panel(market, sessions)
    _report(panel, week_of, args.phase)
    panel.to_pickle(PANEL_DIR / f"vcp_{args.phase}_panel.pkl")
    print(json.dumps(verdict(panel, week_of), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
