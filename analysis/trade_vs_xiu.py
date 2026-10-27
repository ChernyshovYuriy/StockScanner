"""
analysis/trade_vs_xiu.py -- Phase 1 of the idle-cash plan: for every closed
trade, would the same money have done better in XIU.TO over the same days?

    xiu_pnl = entry_notional * (XIU_close(sell_date) / XIU_open(entry_date) - 1)
    excess  = trade_pnl - xiu_pnl - commissions - spread_cost

Same fill convention on both legs: entries at the open (backtest: D+1 open;
live: the ~09:45 quote, approximated by XIU's open), exits at the close
(backtest: D's close; live: the ~15:50 price, approximated by XIU's close).
XIU is total return (ref/xiu_cash_daily.csv, one fresh auto-adjusted
download). The stock leg is put on the same basis:
  live      recorded P&L (raw quotes) + cash dividends with ex-date in
            (entry_date, sell_date] x shares (ref/dividends.csv)
  backtest  P&L on the market cache's adjusted bars (approximately total
            return; see 00_recon §5 for the cache-tail drift caveat)

Costs: commission per order is a parameter (0 by default -- user decision,
platform-specific); spread/slippage 10 bps per side on the stock notional,
20 bps in the adverse scenario. The XIU leg is the buy-and-hold alternative
and carries no cost here (switching costs belong to Phase 3).

Positions still open are included, marked to market (status "open"): in the
walk-forward every fold ends with up to 8 of them, its equity counts them,
and winners are over-represented among them (losers exit fast on stops),
so a closed-trades-only sample is biased against the strategy. Results are
reported for both samples ("closed" and "all"); "all" matches the equity.
Live open positions are marked at the 2026-10-26 monitor prices; SES.TO is
left out (no valid price since 2026-09-02).

Live and backtest trades are processed separately. Inputs are all local:
data/trading.db, reports/idle_cash/ref/*, and
reports/idle_cash/01_trade_vs_xiu/backtest_trades.csv (from
analysis/run_walk_forward_trades.py) when present.

Usage:
    python analysis/trade_vs_xiu.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reports" / "idle_cash" / "ref"
OUT = ROOT / "reports" / "idle_cash" / "01_trade_vs_xiu"

COMMISSION_PER_ORDER = 0.0
SPREAD_BPS = {"zero": 0.0, "realistic": 10.0, "adverse": 20.0}
N_BOOT = 10_000
SEED = 20261027
REGIME_SMA = 200


# ── reference data ────────────────────────────────────────────────────────────

def load_xiu() -> pd.DataFrame:
    df = pd.read_csv(REF / "xiu_cash_daily.csv", parse_dates=["date"])
    xiu = df[df.ticker == "XIU.TO"].set_index("date").sort_index()[["Open", "Close"]]
    xiu["sma"] = xiu["Close"].rolling(REGIME_SMA).mean()
    return xiu


def _bar(xiu: pd.DataFrame, day: pd.Timestamp, col: str) -> float:
    """XIU price on `day`; a missing day (holiday mismatch) uses the next bar
    for an open and the previous bar for a close."""
    if day in xiu.index:
        return float(xiu.at[day, col])
    if col == "Open":
        later = xiu.loc[xiu.index > day]
        return float(later[col].iloc[0])
    return float(xiu.loc[xiu.index < day, col].iloc[-1])


def _regime_at_entry(xiu: pd.DataFrame, entry: pd.Timestamp) -> str:
    prior = xiu.loc[xiu.index < entry].dropna(subset=["sma"])
    if prior.empty:
        return "unknown"
    return "above_200d" if prior["Close"].iloc[-1] >= prior["sma"].iloc[-1] else "below_200d"


# ── trade sources ─────────────────────────────────────────────────────────────

def load_live_trades() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "data" / "trading.db"), read_only=True)
    trades = con.execute("""
        SELECT t.ticker, t.entry_date, t.sell_date, t.entry_price, t.sell_price, t.shares,
               t.pnl_dollars AS pnl, t.reason AS exit_reason,
               (SELECT b.reason FROM transactions b WHERE b.side = 'BUY' AND b.ticker = t.ticker
                  AND b.trade_date = t.entry_date LIMIT 1) AS pattern,
               (SELECT i.stop_price FROM intents i WHERE i.intent_status = 'EXECUTED' AND i.ticker = t.ticker
                  AND abs(i.executed_price - t.entry_price) < 1e-6
                  ORDER BY i.processed_at DESC LIMIT 1) AS stop_price
        FROM trades t ORDER BY t.sell_date
    """).df()
    con.close()
    for c in ("entry_date", "sell_date"):
        trades[c] = pd.to_datetime(trades[c])
    trades["status"] = "closed"

    marks = pd.read_csv(REF / "live_open_marks_20261026.csv", parse_dates=["entry_date", "last_date"])
    marks = marks[marks.last_date == pd.Timestamp("2026-10-26")]   # drops SES.TO (stale since 09-02)
    marks = marks[["ticker", "entry_date", "entry_price", "shares", "last_close"]]
    con = duckdb.connect(str(ROOT / "data" / "trading.db"), read_only=True)
    pos = con.execute("""
        SELECT p.ticker, p.stop_price, b.reason AS pattern FROM positions p
        LEFT JOIN transactions b ON b.side = 'BUY' AND b.ticker = p.ticker AND b.trade_date = p.entry_date
    """).df()
    con.close()
    opened = marks.merge(pos, on="ticker", how="left").assign(
        sell_date=pd.Timestamp("2026-10-26"), sell_price=marks.last_close.values,
        exit_reason="OPEN_MARKED", status="open")
    opened["pnl"] = (opened.sell_price - opened.entry_price) * opened.shares
    opened = opened[["ticker", "entry_date", "sell_date", "entry_price", "sell_price", "shares", "pnl",
                     "exit_reason", "pattern", "stop_price", "status"]]
    trades = pd.concat([trades, opened], ignore_index=True)

    div = pd.read_csv(REF / "dividends.csv", parse_dates=["ex_date"])
    def _divs(r) -> float:
        d = div[(div.ticker == r.ticker) & (div.ex_date > r.entry_date) & (div.ex_date <= r.sell_date)]
        return float(d.amount.sum() * r.shares)
    trades["dividends"] = trades.apply(_divs, axis=1)
    trades["pnl_tr"] = trades["pnl"] + trades["dividends"]

    sectors = json.loads((REF / "live_sectors.json").read_text())
    trades["sector"] = trades.ticker.map(sectors).fillna("Unknown")
    trades["source"] = "live"
    return trades


def load_backtest_trades() -> pd.DataFrame | None:
    path = OUT / "backtest_trades.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path, parse_dates=["entry_date", "sell_date"]).assign(status="closed")
    open_path = OUT / "backtest_open.csv"
    if open_path.exists():
        o = pd.read_csv(open_path, parse_dates=["entry_date", "sell_date"]).assign(status="open")
        o["pnl"] = (o.sell_price - o.entry_price) * o.shares
        t = pd.concat([t, o], ignore_index=True)
    t["dividends"] = 0.0          # already in the adjusted bars
    t["pnl_tr"] = t["pnl"]
    t["sector"] = "n/a"           # no offline sector map for the backtest universe
    t["source"] = "backtest"
    return t


# ── per-trade excess ──────────────────────────────────────────────────────────

def _exit_category(reason) -> str:
    if not isinstance(reason, str) or not reason:
        return "unknown"
    return re.split(r"[(;]", reason, maxsplit=1)[0].strip() or "unknown"


def _stop_bucket(stop_pct) -> str:
    if pd.isna(stop_pct):
        return "unknown"
    return "<6%" if stop_pct < 6 else ("6-8%" if stop_pct <= 8 else ">8%")


def add_excess(trades: pd.DataFrame, xiu: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["entry_notional"] = t.entry_price * t.shares
    t["exit_notional"] = t.sell_price * t.shares
    t["xiu_ret"] = [_bar(xiu, x, "Close") / _bar(xiu, e, "Open") - 1
                    for e, x in zip(t.entry_date, t.sell_date)]
    t["xiu_pnl"] = t.entry_notional * t.xiu_ret
    t["commissions"] = 2 * COMMISSION_PER_ORDER
    for name, bps in SPREAD_BPS.items():
        spread = (t.entry_notional + t.exit_notional) * bps / 1e4
        t[f"excess_{name}"] = t.pnl_tr - t.xiu_pnl - t.commissions - spread
        t[f"excess_{name}_pct"] = t[f"excess_{name}"] / t.entry_notional * 100
    t["stop_pct"] = (t.entry_price - t.stop_price) / t.entry_price * 100
    t["stop_bucket"] = t.stop_pct.map(_stop_bucket)
    t["exit_cat"] = t.exit_reason.map(_exit_category)
    t["pattern"] = t.pattern.fillna("unknown")
    t["regime"] = [_regime_at_entry(xiu, e) for e in t.entry_date]
    t["hold_days"] = (t.sell_date - t.entry_date).dt.days
    return t


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(t: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for name in SPREAD_BPS:
        col = f"excess_{name}_pct"
        lo, hi = bootstrap_ci(t[col].to_numpy(), rng)
        rows.append({
            "costs": name, "trades": len(t),
            "gross_pnl": t.pnl_tr.sum(), "xiu_matched_pnl": t.xiu_pnl.sum(),
            "excess_total": t[f"excess_{name}"].sum(),
            "excess_mean_$": t[f"excess_{name}"].mean(),
            "excess_mean_pct": t[col].mean(), "ci95_lo_pct": lo, "ci95_hi_pct": hi,
            "beat_xiu_rate": (t[f"excess_{name}"] > 0).mean(),
        })
    return pd.DataFrame(rows)


def breakdown(t: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for dim in ("pattern", "stop_bucket", "exit_cat", "sector", "regime"):
        for val, g in t.groupby(dim):
            x = g["excess_realistic_pct"].to_numpy()
            lo, hi = bootstrap_ci(x, rng)
            rows.append({"dimension": dim, "value": val, "trades": len(g),
                         "excess_mean_pct": x.mean(), "ci95_lo_pct": lo, "ci95_hi_pct": hi,
                         "excess_total_$": g["excess_realistic"].sum(),
                         "beat_xiu_rate": (g["excess_realistic"] > 0).mean()})
    return pd.DataFrame(rows)


def charts(t: pd.DataFrame, source: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t["excess_realistic_pct"], bins=30, color="#4a6fa5")
    ax.axvline(0, color="black", lw=1)
    ax.set_title(f"{source}: excess over XIU per trade (realistic costs)")
    ax.set_xlabel("excess, % of entry notional")
    ax.set_ylabel("trades")
    fig.tight_layout()
    fig.savefig(OUT / f"{source}_excess_hist.png", dpi=120)
    plt.close(fig)

    s = t.sort_values("sell_date")
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, style in (("zero", ":"), ("realistic", "-"), ("adverse", "--")):
        ax.plot(s.sell_date, s[f"excess_{name}"].cumsum(), style, label=f"{name} costs")
    ax.axhline(0, color="black", lw=1)
    ax.set_title(f"{source}: cumulative excess over XIU ($)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / f"{source}_cumulative_excess.png", dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xiu = load_xiu()
    sources = {"live": load_live_trades()}
    bt = load_backtest_trades()
    if bt is not None:
        sources["backtest"] = bt

    all_rows, summaries, breakdowns = [], [], []
    for source, trades in sources.items():
        t = add_excess(trades, xiu)
        all_rows.append(t)
        for sample, sub in (("closed", t[t.status == "closed"]), ("all", t)):
            rng = np.random.default_rng(SEED)
            summaries.append(summarize(sub, rng).assign(source=source, sample=sample))
            breakdowns.append(breakdown(sub, rng).assign(source=source, sample=sample))
        charts(t, source)

    keep = ["source", "status", "fold", "ticker", "pattern", "entry_date", "sell_date", "hold_days", "entry_price",
            "sell_price", "shares", "stop_price", "stop_pct", "stop_bucket", "exit_reason", "exit_cat",
            "sector", "regime", "entry_notional", "pnl", "dividends", "pnl_tr", "xiu_ret", "xiu_pnl",
            "excess_zero", "excess_realistic", "excess_adverse",
            "excess_zero_pct", "excess_realistic_pct", "excess_adverse_pct"]
    out = pd.concat(all_rows, ignore_index=True)
    out.reindex(columns=keep).to_csv(OUT / "trades_with_excess.csv", index=False)
    summ = pd.concat(summaries, ignore_index=True)
    summ.to_csv(OUT / "totals.csv", index=False)
    brk = pd.concat(breakdowns, ignore_index=True)
    brk.to_csv(OUT / "breakdowns.csv", index=False)

    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print()
    print(brk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
