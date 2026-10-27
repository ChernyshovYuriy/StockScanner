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
    python analysis/run_walk_forward_trades.py
Outputs (reports/idle_cash/01_trade_vs_xiu/):
    backtest_trades.csv   one row per closed trade, with fold number
    backtest_open.csv     positions still held when each fold ended, marked
                          at that day's close -- the fold's equity includes
                          them, so a closed-trades-only view is biased
    backtest_equity.csv   fold, date, cash, open_value, total_equity
    backtest_folds.csv    per-fold return / max DD / trades
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    bounds = [round(i * n / N_FOLDS) for i in range(N_FOLDS + 1)]
    trades, equity, folds, open_rows = [], [], [], []
    for f, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        cfg = BacktestConfig(start_date=str(days[s]), end_date=str(days[e - 1]) if e < n else END,
                             _screener_cache=sub_cache(s, e - s), **base)
        t0 = time.perf_counter()
        res = BacktestRunner(cfg).run(verbose=False)
        tl = res.trade_log_df().assign(fold=f)
        eq = res.equity_curve_df().assign(fold=f)
        trades.append(tl)
        equity.append(eq)
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
                          max_dd_pct=round(dd, 2), trades=len(tl)))
        print(f"fold {f:>2}/{N_FOLDS} {days[s]}..{days[e - 1]}  ret={ret:+.2f}%  dd={dd:.2f}%  "
              f"trades={len(tl)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        pd.concat(trades).to_csv(OUT_DIR / "backtest_trades.csv", index=False)
        pd.concat(equity).to_csv(OUT_DIR / "backtest_equity.csv", index=False)
        pd.DataFrame(folds).to_csv(OUT_DIR / "backtest_folds.csv", index=False)
        pd.DataFrame(open_rows).to_csv(OUT_DIR / "backtest_open.csv", index=False)

    fdf = pd.DataFrame(folds)
    compounded = ((1 + fdf.ret_pct / 100).prod() - 1) * 100
    print(f"\nMean fold return {fdf.ret_pct.mean():+.2f}%, compounded {compounded:+.1f}%, "
          f"{int(fdf.trades.sum())} trades", flush=True)


if __name__ == "__main__":
    main()
