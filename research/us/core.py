"""
The core and momentum sleeves on US stocks through the project's own BacktestRunner, unchanged (PLAN.md). SPY is
given the benchmark's name, XIU.TO. Sixteen quarters, each starting fresh, as in walk_forward_ab.py; also one run.

    python -m research.us.core us|ca core|momentum
"""
from __future__ import annotations

import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_runner import BacktestConfig, BacktestRunner, _lookback_start, _run_screener_step, _trading_days
from config import (GAP_FILTER_PCT, MAX_POSITIONS, MOMENTUM_CHAND_ARM_PCT, MOMENTUM_CHAND_TRAIL_ATR_K, MOMENTUM_MAX_ATR_PCT,
                    MOMENTUM_MAX_POSITIONS, MOMENTUM_RISK_PER_TRADE_PCT, RISK_PER_TRADE_PCT)
from market_data import HistoricalSliceProvider
from position_monitor import ExitParams
from research.us import backtest as us
from swing_tickers import Thresholds
from time_utils import TSX_TZ, set_backtest_clock

START, END = "2022-10-03", "2026-10-02"
BENCH, LOOKBACK, EVERY, FOLDS, CAPITAL = "XIU.TO", 504, 5, 16, 50_000.0
WORKERS = 3
_provider: HistoricalSliceProvider = None
_cfg: Dict = {}


def settings(sleeve: str) -> Dict:
    if sleeve == "core":
        return dict(risk_pct=RISK_PER_TRADE_PCT, max_positions=MAX_POSITIONS, regime_filter=True, universe_filter=Thresholds(), exit_params=ExitParams())
    return dict(risk_pct=MOMENTUM_RISK_PER_TRADE_PCT, max_positions=MOMENTUM_MAX_POSITIONS, regime_filter=False,
                universe_filter=Thresholds(max_atr_pct_14=MOMENTUM_MAX_ATR_PCT),
                exit_params=ExitParams(chand_trail_atr_k=MOMENTUM_CHAND_TRAIL_ATR_K, chand_arm_pct=MOMENTUM_CHAND_ARM_PCT))


def base(tickers: List[str], sleeve: str) -> Dict:
    return dict(tickers=tickers, benchmark=BENCH, initial_cash=CAPITAL, top_n_buys=3, min_score=0.0, lookback_days=LOOKBACK,
                screener_frequency=EVERY, max_tracked_tickers=40, min_rr=2.0, sizing="live", sizing_basis="cash", gap_filter_pct=GAP_FILTER_PCT,
                atr_stop_mult=1.5, **settings(sleeve))


def load(tag: str) -> Dict[str, pd.DataFrame]:
    if tag == "us":
        data = us.load()
        data[BENCH] = data.pop(us.SPY)
    else:
        from research.breakouts import prices
        data = prices.load()
    first = _lookback_start(START, LOOKBACK)
    data = {t: d.loc[first:] for t, d in data.items()}
    return {t: d for t, d in data.items() if len(d) > 300}


def _init(path: str, cfg: Dict) -> None:
    global _provider, _cfg
    _provider, _cfg = HistoricalSliceProvider(pd.read_pickle(path)), cfg


def _screen(chunk: List[Tuple[int, object]]) -> Dict:
    import contextlib, io
    cfg = BacktestConfig(start_date=START, end_date=END, _provider=_provider, **_cfg)
    out = {}
    for i, day in chunk:
        set_backtest_clock(datetime(day.year, day.month, day.day, 16, 5, tzinfo=TSX_TZ))
        with contextlib.redirect_stdout(io.StringIO()):
            out[i] = _run_screener_step(cfg, _provider, pd.Timestamp(day))
    set_backtest_clock(None)
    return out


def _fold(args: Tuple[str, str, Dict]) -> Dict:
    import contextlib, io
    start, end, cache = args
    cfg = BacktestConfig(start_date=start, end_date=end, _provider=_provider, _screener_cache=cache, **_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        r = BacktestRunner(cfg).run(verbose=False)
    eq = r.equity_curve_df()["total_equity"]
    tl = r.trade_log_df()
    col = next((c for c in tl.columns if "pnl_pct" in c.lower() or c.lower() in ("return_pct", "pct")), None)
    return {"start": start, "end": end, "ret": float(eq.iloc[-1] / CAPITAL - 1), "dd": float((eq / eq.cummax() - 1).min()), "trades": len(tl),
            "win": float((tl[col] > 0).mean()) if col and len(tl) else np.nan, "avg": float(tl[col].mean()) if col and len(tl) else np.nan}


def sub_cache(full: Dict, offset: int, n: int) -> Dict:
    out = {}
    for j in range(0, n, EVERY):
        k = ((offset + j) // EVERY) * EVERY
        if k in full:
            out[j] = full[k]
    return out


def main() -> int:
    tag, sleeve = sys.argv[1], sys.argv[2]
    data = load(tag)
    names = sorted(data)
    bench = data[BENCH]["Close"]
    path = f"{tempfile.gettempdir()}/us_core_{tag}_{sleeve}.pkl"         # one writer, many readers: duckdb allows one
    pd.to_pickle(data, path)
    del data
    cfg = base(names, sleeve)
    days = _trading_days(START, END)
    todo = [(i, d) for i, d in enumerate(days) if i % EVERY == 0]
    chunks = [todo[k::WORKERS * 4] for k in range(WORKERS * 4)]
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(path, cfg)) as pool:
        full = {}
        for part in pool.map(_screen, chunks):
            full.update(part)
        print(f"{tag} {sleeve}: {len(names) - 1} names, {len(full)} weekly screens", flush=True)
        n = len(days)
        b = [round(i * n / FOLDS) for i in range(FOLDS + 1)]
        jobs = [(str(days[s]), str(days[e - 1]), sub_cache(full, s, e - s)) for s, e in zip(b[:-1], b[1:])]
        jobs.append((str(days[0]), str(days[-1]), full))
        rows = list(pool.map(_fold, jobs))
    os.unlink(path)
    t = pd.DataFrame(rows)
    t["bench"] = [float(bench.loc[a:z].iloc[-1] / bench.loc[a:z].iloc[0] - 1) for a, z in zip(t["start"], t["end"])]
    us.OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(us.OUT / f"{sleeve}_{tag}.csv", index=False)
    f, one = t.iloc[:-1], t.iloc[-1]
    from scipy import stats
    d = f["ret"] - f["bench"]
    print(t.round(4).to_string())
    print(f"{tag} {sleeve}: quarters mean {f['ret'].mean():+.2%} vs benchmark {f['bench'].mean():+.2%}, ahead {int((d > 0).sum())}/{len(d)}, "
          f"p={stats.ttest_1samp(d, 0.0).pvalue:.3f}; compounded {np.prod(1 + f['ret']) - 1:+.1%} vs {np.prod(1 + f['bench']) - 1:+.1%}; "
          f"one 4-year run {one['ret']:+.1%} (worst {one['dd']:.1%}, {one['trades']} trades) vs {one['bench']:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
