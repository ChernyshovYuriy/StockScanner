"""
analysis/run_walk_forward_trades.py -- re-run the 16-fold point-in-time
walk-forward of the live core-sleeve config and SAVE what Phase 1 of the
idle-cash plan needs: every closed trade (with setup type, planned stop and
exit reason) and each fold's daily equity curve.

Same setup as the 2026-10-09 run behind "+22% vs XIU +110%" (walk_forward_ab.py
base config, point-in-time universe, live exit rules incl. the 20d/0% time
stop), over the window the local cache covers. Bars come from
data/market_cache.db ONLY (market_data_cache.load_from_cache) -- a
HistoricalSliceProvider.from_cache() run would try to backfill ~1,000
tickers from Yahoo, since the 504-day lookback predates the cache's start.

Not a live-parity run in two respects, both inherited from that baseline:
no sector cap (no offline sector map for ~950 tickers) and top_n_buys=3.

Usage:
    python analysis/run_walk_forward_trades.py                 # Phase 1 baseline -> 01_trade_vs_xiu/
    python analysis/run_walk_forward_trades.py off all_on ...  # Phase 2 variants -> 02_baseline/<variant>/
    python analysis/run_walk_forward_trades.py realistic_A ... # Phase 3 runs -> 03_matrix/runs/<variant>/

Phase 2 variants (VARIANTS below) switch on the BacktestConfig flags one at a
time and all together; "off" is the Phase 1 config and must reproduce it.
Cash yield uses CASH.TO's daily total return from ref/xiu_cash_daily.csv.

Phase 3 runs (PHASE3_VARIANTS) are <cost scenario>_<A|B|C>: the Phase 2
all_on config with idle cash in cash (A), always in XIU (B) or in XIU above
its 200-day (C), under realistic / adverse / zero costs -- see
reports/idle_cash/03_matrix/DECISION_RULES.md. They also write
backtest_idle.csv (the real account's daily log).

Outputs (reports/idle_cash/01_trade_vs_xiu/, or 02_baseline/<variant>/):
    backtest_trades.csv   one row per closed trade, with fold number
    backtest_open.csv     positions still held when each fold ended, marked
                          at that day's close -- the fold's equity includes
                          them, so a closed-trades-only view is biased
    backtest_equity.csv   fold, date, cash, open_value, total_equity
    backtest_folds.csv    per-fold return / max DD / trades / costs / cash yield
"""
from __future__ import annotations

import sys
import time
import urllib.request
from datetime import datetime as _dt
from pathlib import Path

import pickle

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_runner import BacktestConfig, BacktestRunner, _lookback_start, _run_screener_step, _trading_days  # noqa: E402
from config import BACKTEST_RAW_TICKERS_URL  # noqa: E402
from market_data import HistoricalSliceProvider  # noqa: E402
from market_data_cache import load_from_cache  # noqa: E402
from position_monitor import ExitParams  # noqa: E402
from swing_tickers import Thresholds  # noqa: E402
from time_utils import TSX_TZ, set_backtest_clock  # noqa: E402

START = "2022-10-03"
END = "2026-10-02"          # last bar in the local cache
BENCHMARK = "XIU.TO"
CAPITAL = 50_000.0
LOOKBACK_DAYS = 504
SCREENER_FREQ = 5
N_FOLDS = 16
OUT_DIR = ROOT / "reports" / "idle_cash" / "01_trade_vs_xiu"
# Screener snapshots take ~25 min to compute; reused across re-runs.
SCREENER_PICKLE = ROOT / "cache" / f"idle_cash_screener_{START}_{END}.pkl"
REF_DIR = ROOT / "reports" / "idle_cash" / "ref"
PHASE2_DIR = ROOT / "reports" / "idle_cash" / "02_baseline"

COSTS = dict(costs_enabled=True, commission_per_order=0.0, stock_spread_bps=10.0)
YIELD = dict(cash_yield_source="cash_etf")   # cash_yield_returns filled in at run time
CONSERVATIVE = dict(same_bar_rule="conservative")
VARIANTS = {
    "off": {},
    "costs": COSTS,
    "yield": YIELD,
    "conservative": CONSERVATIVE,
    "conservative_gap": {**CONSERVATIVE, "gap_fills": True},
    "all_on": {**COSTS, **YIELD, **CONSERVATIVE, "gap_fills": True},
}
PHASE3_DIR = ROOT / "reports" / "idle_cash" / "03_matrix" / "runs"
COST_SCENARIOS = {
    "realistic": dict(costs_enabled=True, commission_per_order=0.0, stock_spread_bps=10.0, xiu_spread_bps=2.0),
    "adverse": dict(costs_enabled=True, commission_per_order=0.0, stock_spread_bps=20.0, xiu_spread_bps=4.0),
    "zero": dict(costs_enabled=False),
}
IDLE_MODES = {"A": "cash", "B": "xiu_always", "C": "xiu_gated"}
PHASE3_VARIANTS = {
    f"{scen}_{vid}": {**YIELD, **CONSERVATIVE, "gap_fills": True, **costs, "idle_cash_mode": mode}
    for scen, costs in COST_SCENARIOS.items() for vid, mode in IDLE_MODES.items()
}


def _load_tickers(source: str) -> list[str]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as resp:
            content = resp.read().decode("utf-8")
    else:
        content = Path(source).read_text(encoding="utf-8")
    return [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _offline_provider(tickers: list[str], start: str, end: str) -> HistoricalSliceProvider:
    raw = load_from_cache(tickers, start, end)
    data = {t: df for t, df in raw.items() if not df.empty and df["Close"].iloc[-1] > 0}
    return HistoricalSliceProvider(data)


def _cash_returns() -> dict:
    ref = pd.read_csv(REF_DIR / "xiu_cash_daily.csv", parse_dates=["date"])
    cash = ref[ref.ticker == "CASH.TO"].set_index("date")["Close"].sort_index()
    return {ts.date(): float(r) for ts, r in cash.pct_change().dropna().items()}


def main(variants: list[str]) -> None:
    tickers = list(dict.fromkeys(_load_tickers(BACKTEST_RAW_TICKERS_URL) + [BENCHMARK]))
    provider = _offline_provider(tickers, _lookback_start(START, LOOKBACK_DAYS), END)
    print(f"Universe: {len(tickers)} listed, {len(provider)} with cached bars", flush=True)

    base = dict(
        tickers=tickers, benchmark=BENCHMARK,
        initial_cash=CAPITAL, risk_pct=1.0, top_n_buys=3, min_score=0.0,
        lookback_days=LOOKBACK_DAYS, screener_frequency=SCREENER_FREQ,
        max_tracked_tickers=40, regime_filter=True, min_rr=2.0, atr_stop_mult=1.5,
        max_positions=8, sizing="live", sizing_basis="cash",
        gap_filter_pct=None, universe_filter=Thresholds(), _provider=provider,
        exit_params=ExitParams(),   # live defaults: 20d/0% time stop, close trigger
    )

    days = _trading_days(START, END)
    n = len(days)
    pre = BacktestConfig(start_date=START, end_date=END, **base)
    t0 = time.perf_counter()
    if SCREENER_PICKLE.exists():
        full_cache = pickle.loads(SCREENER_PICKLE.read_bytes())
    else:
        full_cache = {}
        for i, day in enumerate(days):
            if i % SCREENER_FREQ == 0:
                set_backtest_clock(_dt(day.year, day.month, day.day, 16, 5, tzinfo=TSX_TZ))
                full_cache[i] = _run_screener_step(pre, provider, pd.Timestamp(day))
        set_backtest_clock(None)
        SCREENER_PICKLE.parent.mkdir(parents=True, exist_ok=True)
        SCREENER_PICKLE.write_bytes(pickle.dumps(full_cache))
    print(f"Screener snapshots: {len(full_cache)} in {time.perf_counter() - t0:.0f}s", flush=True)

    def sub_cache(offset: int, length: int) -> dict:
        sub: dict = {}
        for j in range(length):
            key = (j // SCREENER_FREQ) * SCREENER_FREQ
            full_key = ((offset + j) // SCREENER_FREQ) * SCREENER_FREQ
            if key not in sub and full_key in full_cache:
                sub[key] = full_cache[full_key]
        return sub

    cash_returns = _cash_returns()
    for variant in variants or [None]:
        overrides = dict({**VARIANTS, **PHASE3_VARIANTS}[variant]) if variant else {}
        if overrides.get("cash_yield_source") == "cash_etf":
            overrides["cash_yield_returns"] = cash_returns
        out_dir = (PHASE3_DIR if variant in PHASE3_VARIANTS else PHASE2_DIR) / variant if variant else OUT_DIR
        print(f"\n== {variant or 'phase1'} -> {out_dir.relative_to(ROOT)}", flush=True)
        _run_folds({**base, **overrides}, days, sub_cache, provider, out_dir)


def _run_folds(base: dict, days: list, sub_cache, provider, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(days)
    bounds = [round(i * n / N_FOLDS) for i in range(N_FOLDS + 1)]
    trades, equity, folds, open_rows, idle = [], [], [], [], []
    for f, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        cfg = BacktestConfig(start_date=str(days[s]), end_date=str(days[e - 1]) if e < n else END,
                             _screener_cache=sub_cache(s, e - s), **base)
        t0 = time.perf_counter()
        res = BacktestRunner(cfg).run(verbose=False)
        tl = res.trade_log_df().assign(fold=f)
        eq = res.equity_curve_df().assign(fold=f)
        trades.append(tl)
        equity.append(eq)
        if res.idle_log is not None:
            idle.append(res.idle_log.assign(fold=f))
        mark_day = pd.Timestamp(eq["date"].iloc[-1])
        for pos in res.open_positions.values():
            bars = provider.get(pos.ticker, as_of=mark_day)
            open_rows.append(dict(fold=f, ticker=pos.ticker, entry_date=pos.entry_date, entry_price=pos.entry_price,
                                  shares=pos.shares, stop_price=pos.stop_price, pattern=pos.pattern,
                                  sell_date=mark_day.date(), sell_price=float(bars["Close"].iloc[-1]),
                                  exit_reason="OPEN_AT_FOLD_END"))
        curve = eq["total_equity"]
        ret = (curve.iloc[-1] / CAPITAL - 1) * 100
        dd = float(((curve - curve.cummax()) / curve.cummax()).min() * 100)
        folds.append(dict(fold=f, start=days[s], end=days[e - 1], ret_pct=round(ret, 2),
                          max_dd_pct=round(dd, 2), trades=len(tl),
                          costs=round(res.costs, 2), cash_yield=round(res.cash_yield, 2)))
        print(f"fold {f:>2}/{N_FOLDS} {days[s]}..{days[e - 1]}  ret={ret:+.2f}%  dd={dd:.2f}%  "
              f"trades={len(tl)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        pd.concat(trades).to_csv(out_dir / "backtest_trades.csv", index=False)
        pd.concat(equity).to_csv(out_dir / "backtest_equity.csv", index=False)
        pd.DataFrame(folds).to_csv(out_dir / "backtest_folds.csv", index=False)
        pd.DataFrame(open_rows).to_csv(out_dir / "backtest_open.csv", index=False)
        if idle:
            pd.concat(idle).to_csv(out_dir / "backtest_idle.csv", index=False)

    fdf = pd.DataFrame(folds)
    compounded = ((1 + fdf.ret_pct / 100).prod() - 1) * 100
    print(f"\nMean fold return {fdf.ret_pct.mean():+.2f}%, compounded {compounded:+.1f}%, "
          f"{int(fdf.trades.sum())} trades", flush=True)


if __name__ == "__main__":
    unknown = [v for v in sys.argv[1:] if v not in VARIANTS and v not in PHASE3_VARIANTS]
    if unknown:
        sys.exit(f"Unknown variant(s) {unknown}; choose from {list(VARIANTS) + list(PHASE3_VARIANTS)}")
    main(sys.argv[1:])
