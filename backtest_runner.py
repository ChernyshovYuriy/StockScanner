"""
backtest_runner.py
==================
Phase 4 — Day-by-day historical backtest runner.

Simulates the live trading workflow faithfully, one trading day at a time:

  For each day D in [start_date, end_date]:

    1. AFTER-CLOSE (D 16:05 ET)
       a. Set clock to D 16:05
       b. Run screener  — score all tickers using only data up to D
       c. Run pipeline  — detect patterns, update signal DB, emit buy intents

    2. NEXT OPEN (D+1 09:31 ET)
       a. Set clock to D+1 09:31
       b. Execute buys  — fill CONFIRMED intents at D+1 open price
       c. Deduct cost from PortfolioState.cash (equal allocation)

    3. DAILY MONITOR (every D in hold period)
       a. Run position_monitor.compute_signals on data up to D
       b. If SELL — execute at D close price, add proceeds to cash

  Capital flows:
    cash  ←  starts at initial_capital
    cash  →  decreases on buy  (D+1 open price × whole shares)
    cash  ←  increases on sell (D close price × shares)
    realized_pnl accumulated in PortfolioState

Non-goals / constraints (see architecture doc):
  - Does NOT change any scoring, pattern detection, or sizing logic
  - Does NOT alter the signal state machine
  - Existing modules (auto_pipeline, position_monitor, canadian_stock_screener)
    are called with injected dependencies — their code is not modified here
  - No lookahead: every call receives only data up to the current sim_date

Usage:
    from backtest_runner import BacktestConfig, BacktestRunner

    cfg = BacktestConfig(
        tickers       = ["RY.TO", "TD.TO", "ENB.TO", "XIU.TO"],
        start_date    = "2023-01-01",
        end_date      = "2024-01-01",
        initial_cash  = 100_000.0,
        risk_pct      = 1.0,
        top_n_buys    = 3,
    )
    runner = BacktestRunner(cfg)
    results = runner.run()
    print(results.summary())
    results.trade_log_df().to_csv("backtest_trades.csv", index=False)
    results.equity_curve_df().to_csv("backtest_equity.csv", index=False)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from market_data import HistoricalSliceProvider
from portfolio import ClosedTrade, OpenPosition, PortfolioState
from position_monitor import ExitParams
from time_utils import TSX_TZ, set_backtest_clock

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """All parameters for a single backtest run."""

    # Universe
    tickers: List[str]  # must include benchmark (e.g. "XIU.TO")
    benchmark: str = "XIU.TO"

    # Date range
    start_date: str = "2023-01-01"  # ISO string, inclusive
    end_date: str = "2024-01-01"  # ISO string, exclusive (last day analysed is end_date - 1 bday)

    # Capital
    initial_cash: float = 100_000.0
    risk_pct: float = 1.0  # % of account risked per trade

    # Screener / pipeline params
    lookback_days: int = 504  # bars fed to screener
    top_n_buys: int = 3  # max concurrent positions opened per day
    min_score: float = 0.0  # min composite_score to be eligible for buy.
    # NOTE: 55.0 is appropriate for 100+ ticker universes where RS percentile
    # ranking is meaningful. For small universes (< 20 tickers) use 0.0 and
    # let pattern detection (CONFIRMED state) be the real filter.
    min_avg_volume: int = 100_000
    min_price: float = 2.0

    # Screener frequency: run full scoring every N trading days.
    # Pattern detection still runs every day; only scoring is throttled.
    # Default=5 (weekly). Use 1 to score every day (slower, large universes).
    screener_frequency: int = 5

    # Max tickers passed to pattern detection each day.
    # Mirrors auto_pipeline.py max_tracked_tickers=40 (live default).
    # Reducing this is the single biggest speed lever for large universes.
    max_tracked_tickers: int = 40

    # Pipeline detection params (mirrors PipelineConfig defaults)
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    min_rr: float = 2.0

    # Simulation hours (ET) — only affects the clock pin, not logic
    after_close_hour: int = 16
    after_close_min: int = 5
    next_open_hour: int = 9
    next_open_min: int = 31

    # Exit rule overrides — None = use position_monitor.py defaults (live behaviour).
    # Pass an ExitParams instance to tune stops/time-stop for backtesting.
    exit_params: Optional[ExitParams] = field(default=None, repr=False)

    # Data provider override (optional — for tests or pre-loaded data)
    _provider: Optional[HistoricalSliceProvider] = field(
        default=None, repr=False
    )

    # Market regime filter — only open NEW positions when benchmark is in uptrend.
    # When True: buys are blocked whenever benchmark_close < benchmark_sma200.
    # When False (default): no filter, all confirmed signals are acted on.
    # Sells / position management are NEVER blocked regardless of this setting.
    regime_filter: bool = False

    # Pre-computed screener cache: {trading_day_index: pd.DataFrame}
    # Injected by the sweep runner to avoid recomputing scores per combo.
    # None = compute fresh each run (default, single-run mode).
    _screener_cache: Optional[Dict] = field(default=None, repr=False)

    # Gap filter: skip a buy if open price > intent_entry * (1 + gap_filter_pct/100).
    # None = no filter (buy at any open price, matches pre-2026-05 backtest behaviour).
    # Set to e.g. 2.0 to match the live GAP_FILTER_PCT=2.0 in config.py.
    gap_filter_pct: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DayLog:
    """Per-day record for the equity curve and audit log."""
    sim_date: date
    cash: float
    open_value: float  # mark-to-market of open positions at day close
    total_equity: float  # cash + open_value
    realized_pnl: float
    open_tickers: List[str]
    buys_today: List[str]  # tickers bought on this day's D+1 open
    sells_today: List[str]  # tickers sold on this day


class BacktestResults:
    """
    Holds all backtest output.  Provides convenience methods to convert
    to DataFrames for analysis and reporting.
    """

    def __init__(
            self,
            cfg: BacktestConfig,
            day_logs: List[DayLog],
            trades: List[ClosedTrade],
    ):
        self.cfg = cfg
        self.day_logs = day_logs
        self.trades = trades

    # ── DataFrames ────────────────────────────────────────────────────────────

    def equity_curve_df(self) -> pd.DataFrame:
        rows = [
            {
                "date": d.sim_date,
                "cash": round(d.cash, 2),
                "open_value": round(d.open_value, 2),
                "total_equity": round(d.total_equity, 2),
                "realized_pnl": round(d.realized_pnl, 2),
                "open_count": len(d.open_tickers),
            }
            for d in self.day_logs
        ]
        return pd.DataFrame(rows)

    def trade_log_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                "ticker", "entry_date", "sell_date", "entry_price",
                "sell_price", "shares", "pnl", "pnl_pct", "holding_days",
            ])
        return pd.DataFrame([
            {
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "sell_date": t.sell_date,
                "entry_price": t.entry_price,
                "sell_price": t.sell_price,
                "shares": t.shares,
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "holding_days": t.holding_days,
            }
            for t in self.trades
        ])

    # ── Summary stats ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        eq = self.equity_curve_df()
        tl = self.trade_log_df()

        if eq.empty:
            return "No simulation data."

        start_eq = float(eq["total_equity"].iloc[0])
        end_eq = float(eq["total_equity"].iloc[-1])
        total_ret_pct = (end_eq / start_eq - 1.0) * 100.0 if start_eq > 0 else 0.0

        # Max drawdown on equity curve
        equity = eq["total_equity"]
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max * 100
        max_dd = float(dd.min())

        # Trade stats
        n_trades = len(tl)
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_hold = 0.0

        if n_trades > 0:
            winners = tl[tl["pnl"] > 0]
            losers = tl[tl["pnl"] <= 0]
            win_rate = len(winners) / n_trades * 100
            avg_win = float(winners["pnl"].mean()) if not winners.empty else 0.0
            avg_loss = float(losers["pnl"].mean()) if not losers.empty else 0.0
            gross_profit = float(winners["pnl"].sum()) if not winners.empty else 0.0
            gross_loss = abs(float(losers["pnl"].sum())) if not losers.empty else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            avg_hold = float(tl["holding_days"].mean())

        lines = [
            f"{'─' * 55}",
            f"  Backtest Summary",
            f"{'─' * 55}",
            f"  Period          : {self.cfg.start_date} → {self.cfg.end_date}",
            f"  Initial capital : ${self.cfg.initial_cash:>12,.2f}",
            f"  Final equity    : ${end_eq:>12,.2f}",
            f"  Total return    : {total_ret_pct:>+.2f}%",
            f"  Max drawdown    : {max_dd:.2f}%",
            f"  Realized P&L    : ${float(eq['realized_pnl'].iloc[-1]):>+,.2f}",
            f"{'─' * 55}",
            f"  Total trades    : {n_trades}",
            f"  Win rate        : {win_rate:.1f}%",
            f"  Avg win         : ${avg_win:>+,.2f}",
            f"  Avg loss        : ${avg_loss:>+,.2f}",
            f"  Profit factor   : {profit_factor:.2f}",
            f"  Avg hold (days) : {avg_hold:.1f}",
            f"{'─' * 55}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _trading_days(start: str, end: str) -> List[date]:
    """Return a sorted list of business days in [start, end)."""
    idx = pd.bdate_range(start=start, end=end, inclusive="left")
    return [ts.date() for ts in idx]


def _mark_to_market(
        open_positions: Dict[str, OpenPosition],
        provider: HistoricalSliceProvider,
        as_of: pd.Timestamp,
) -> float:
    """Sum of last-close × shares for all open positions as of as_of."""
    total = 0.0
    for ticker, pos in open_positions.items():
        try:
            df = provider.get(ticker, as_of=as_of)
            if not df.empty:
                total += float(df["Close"].iloc[-1]) * pos.shares
        except (KeyError, Exception):
            # If data unavailable, use cost basis as fallback
            total += pos.cost_basis
    return total


def _next_open_price(
        ticker: str,
        provider: HistoricalSliceProvider,
        after: pd.Timestamp,
) -> Optional[float]:
    """
    Return the Open price of the first bar strictly after `after`.
    Returns None if no such bar exists in the dataset.
    """
    try:
        full = provider._data.get(ticker)
        if full is None or full.empty:
            return None
        future = full.loc[full.index > after]
        if future.empty:
            return None
        return float(future["Open"].iloc[0])
    except Exception:
        return None


def _day_close_price(
        ticker: str,
        provider: HistoricalSliceProvider,
        on: pd.Timestamp,
) -> Optional[float]:
    """Return the Close price for the bar on `on` (exact date match)."""
    try:
        df = provider.get(ticker, as_of=on)
        day = df[df.index.normalize() == on.normalize()]
        if day.empty:
            return float(df["Close"].iloc[-1])  # fallback: last available
        return float(day["Close"].iloc[-1])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER STEP  (after close D)
# ─────────────────────────────────────────────────────────────────────────────

def _run_screener_step(
        cfg: BacktestConfig,
        provider: HistoricalSliceProvider,
        sim_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Run the stock screener using only data up to sim_date.

    Returns a DataFrame of scored tickers (may be empty on data failures).
    Screener CONFIG is reused from the live module — no logic changes.
    """
    from canadian_stock_screener import (
        DataManager, ScreenerConfig, ScoreCalculator,
        StockScreener, BENCHMARK, TechnicalIndicators,
    )

    universe_tickers = [t for t in cfg.tickers if t != cfg.benchmark]

    screener_cfg = ScreenerConfig(
        lookback_days=cfg.lookback_days,
        top_n=min(len(universe_tickers), 50),  # ScreenerConfig max is 50
        min_avg_volume=cfg.min_avg_volume,
        min_price=cfg.min_price,
        weights={
            "stage2_score": 0.20,
            "rs_score": 0.20,
            "macd_score": 0.15,
            "obv_score": 0.15,
            "adx_score": 0.10,
            "vam_score": 0.10,
            "breakout_score": 0.10,
        },
    )
    dm = DataManager(
        tickers_source=universe_tickers,
        provider=provider,
    )
    screener = StockScreener(screener_cfg, dm)

    # Inject the cutoff so download_data delegates to the provider
    data = dm.download_data(cfg.lookback_days, as_of=sim_date)
    if BENCHMARK not in data or not data:
        return pd.DataFrame()

    bench_close = data[BENCHMARK]["Close"]
    score_calc = ScoreCalculator(screener_cfg)
    ti = TechnicalIndicators()

    rs_vals = []
    for tkr, df in data.items():
        if tkr == BENCHMARK:
            continue
        try:
            common = df["Close"].index.intersection(bench_close.index)
            if len(common) < 60:
                continue
            c = df["Close"].loc[common]
            b = bench_close.loc[common]
            rs = (c.iloc[-1] / c.iloc[-60] - 1 - (b.iloc[-1] / b.iloc[-60] - 1)) * 100
            rs_vals.append(float(rs))
        except Exception:
            pass

    rows = []
    for tkr in dm.tickers:
        if tkr not in data:
            continue
        df = data[tkr]
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        avg_vol = float(volume.iloc[-20:].mean())
        last_price = float(close.iloc[-1])
        if avg_vol < cfg.min_avg_volume or last_price < cfg.min_price:
            continue
        try:
            s2 = score_calc.score_stage2(close)
            rs = score_calc.score_relative_strength(close, bench_close, rs_vals)
            mac = score_calc.score_macd(close)
            ob = score_calc.score_obv(close, volume)
            adx = score_calc.score_adx(high, low, close)
            vam = score_calc.score_vam(close)
            brk = score_calc.score_breakout(close, high, volume)
            w = screener_cfg.weights
            composite = (s2 * w["stage2_score"] + rs * w["rs_score"] +
                         mac * w["macd_score"] + ob * w["obv_score"] +
                         adx * w["adx_score"] + vam * w["vam_score"] +
                         brk * w["breakout_score"])
            rows.append({
                "ticker": tkr,
                "composite_score": round(composite, 2),
                "price": round(last_price, 4),
            })
        except Exception:
            pass

    return (pd.DataFrame(rows)
            .sort_values("composite_score", ascending=False)
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP  (after close D)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_step(
        screener_df: pd.DataFrame,
        provider: HistoricalSliceProvider,
        sim_date: pd.Timestamp,
        cfg: BacktestConfig,
        signal_db: pd.DataFrame,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Run pattern detection on the screener's top candidates.

    Returns (buy_intents, updated_signal_db).
    buy_intents is a list of dicts with entry/stop/rr for CONFIRMED patterns.
    signal_db is the updated in-memory state (NOT written to disk here).
    """
    from auto_pipeline import (
        detect_all_patterns, compute_levels, expire_missing_tickers, invalidation_check,
        STATE_CONFIRMED, STATE_AT_PIVOT, STATE_ACTIVE, STATE_FAILED,
        SIGNAL_COL_TICKER, SIGNAL_COL_PATTERN, SIGNAL_COL_STATE,
        SIGNAL_COL_FIRST_SEEN, SIGNAL_COL_LAST_SEEN, SIGNAL_COL_DAYS_IN_STATE,
        SIGNAL_COL_CONSECUTIVE_SCREENER_DAYS, SIGNAL_COL_ENTRY, SIGNAL_COL_STOP,
        SIGNAL_COL_TARGET_2R, SIGNAL_COL_TARGET_3R, SIGNAL_COL_RISK_PCT,
        SIGNAL_COL_PIVOT_PRICE, SIGNAL_COL_DETAIL, SIGNAL_COL_ALERT_SENT,
    )

    if screener_df.empty:
        return [], signal_db

    tracked = screener_df["ticker"].tolist()
    db = signal_db.copy()
    db = expire_missing_tickers(db, tracked, sim_date)
    intents = []

    for _, row in screener_df.iterrows():
        ticker = row["ticker"]
        score = row["composite_score"]
        if score < cfg.min_score:
            continue

        try:
            df = provider.get(ticker, as_of=sim_date)
        except KeyError:
            continue

        if df.empty or len(df) < 60:
            continue

        df.index = pd.to_datetime(df.index).tz_localize(None)
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        last_price = float(close.iloc[-1])

        # Check invalidation for existing signals
        existing = db[db[SIGNAL_COL_TICKER] == ticker]
        for idx, ex_row in existing.iterrows():
            if ex_row[SIGNAL_COL_STATE] in (STATE_ACTIVE, STATE_AT_PIVOT, STATE_CONFIRMED):
                if invalidation_check(ticker, df, ex_row):
                    db.at[idx, SIGNAL_COL_STATE] = STATE_FAILED
                    db.at[idx, SIGNAL_COL_DETAIL] = "Invalidation rule triggered"
                    db.at[idx, SIGNAL_COL_LAST_SEEN] = sim_date

        patterns = detect_all_patterns(ticker, df)
        if not patterns:
            continue

        best = patterns[0]
        pattern = best["pattern"]
        state = best["state"]
        pivot = best.get("pivot", last_price)
        detail = best["detail"]

        entry = round(pivot * 1.005, 2)
        levels = compute_levels(close, high, low, entry,
                                cfg.atr_period, cfg.atr_stop_mult)
        rr = ((levels["target_2r"] - levels["entry"]) /
              max(levels["entry"] - levels["stop"], 0.01))

        # Update signal DB
        match = db[(db[SIGNAL_COL_TICKER] == ticker) &
                   (db[SIGNAL_COL_PATTERN] == pattern)]
        if match.empty:
            new_row = {
                SIGNAL_COL_TICKER: ticker,
                SIGNAL_COL_PATTERN: pattern,
                SIGNAL_COL_STATE: state,
                SIGNAL_COL_FIRST_SEEN: sim_date,
                SIGNAL_COL_LAST_SEEN: sim_date,
                SIGNAL_COL_DAYS_IN_STATE: 1,
                SIGNAL_COL_CONSECUTIVE_SCREENER_DAYS: 1,
                SIGNAL_COL_ENTRY: levels["entry"],
                SIGNAL_COL_STOP: levels["stop"],
                SIGNAL_COL_TARGET_2R: levels["target_2r"],
                SIGNAL_COL_TARGET_3R: levels["target_3r"],
                SIGNAL_COL_RISK_PCT: levels["risk_pct"],
                SIGNAL_COL_PIVOT_PRICE: pivot,
                SIGNAL_COL_DETAIL: detail,
                SIGNAL_COL_ALERT_SENT: False,
            }
            db = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
        else:
            i = match.index[0]
            old_state = db.at[i, SIGNAL_COL_STATE]
            db.at[i, SIGNAL_COL_STATE] = state
            db.at[i, SIGNAL_COL_LAST_SEEN] = sim_date
            db.at[i, SIGNAL_COL_DETAIL] = detail
            db.at[i, SIGNAL_COL_ENTRY] = levels["entry"]
            db.at[i, SIGNAL_COL_STOP] = levels["stop"]
            db.at[i, SIGNAL_COL_TARGET_2R] = levels["target_2r"]
            db.at[i, SIGNAL_COL_TARGET_3R] = levels["target_3r"]
            db.at[i, SIGNAL_COL_RISK_PCT] = levels["risk_pct"]
            db.at[i, SIGNAL_COL_PIVOT_PRICE] = pivot
            if old_state == state:
                db.at[i, SIGNAL_COL_DAYS_IN_STATE] = (
                        int(db.at[i, SIGNAL_COL_DAYS_IN_STATE] or 1) + 1
                )
            else:
                db.at[i, SIGNAL_COL_DAYS_IN_STATE] = 1
                db.at[i, SIGNAL_COL_ALERT_SENT] = False

        # Emit buy intent for CONFIRMED patterns with acceptable R:R
        if state == STATE_CONFIRMED and rr >= cfg.min_rr:
            intents.append({
                "ticker": ticker,
                "pattern": pattern,
                "entry": levels["entry"],
                "stop": levels["stop"],
                "rr": round(rr, 2),
                "score": round(score, 2),
            })

    return intents, db


# ─────────────────────────────────────────────────────────────────────────────
# BUY STEP  (next open D+1)
# ─────────────────────────────────────────────────────────────────────────────

def _is_market_in_uptrend(
        benchmark: str,
        provider: HistoricalSliceProvider,
        as_of: pd.Timestamp,
        sma_period: int = 200,
) -> bool:
    """
    Return True when benchmark last close >= its sma_period-day SMA.
    Used by the regime filter to block new buys in downtrending markets.
    Returns True (permissive) on any data error so live mode is unaffected.
    """
    try:
        df = provider.get(benchmark, as_of=as_of)
        if df.empty or len(df) < sma_period:
            return True  # not enough data — allow trades
        close = df["Close"].squeeze()
        sma   = float(close.rolling(sma_period).mean().iloc[-1])
        last  = float(close.iloc[-1])
        return last >= sma
    except Exception:
        return True  # fail open — do not block trades on data errors


def _execute_buys(
        intents: List[Dict],
        portfolio: PortfolioState,
        provider: HistoricalSliceProvider,
        after: pd.Timestamp,  # D's last bar timestamp — buy at next bar's open
        buy_date: date,
        cfg: BacktestConfig,
) -> List[str]:
    """
    Execute buy intents at the D+1 open price.

    Applies the same equal-allocation rule as virtual_buy.py:
      allocation_per_ticker = cash / n_actionable
      shares = int(allocation / price)   — whole shares only

    If cfg.regime_filter is True, buys are blocked when benchmark is below
    its 200-day SMA.  Sells and position management are never blocked.

    Returns list of tickers actually bought.
    """
    if not intents or portfolio.cash <= 0:
        return []

    # Regime filter — block new buys in downtrending markets
    if cfg.regime_filter:
        if not _is_market_in_uptrend(cfg.benchmark, provider, after):
            return []   # market below 200d SMA — no new longs

    # Filter: skip tickers already held
    owned = set(portfolio.open_positions.keys())
    actionable = [i for i in intents
                  if i["ticker"] not in owned][:cfg.top_n_buys]

    if not actionable:
        return []

    allocation = portfolio.cash / len(actionable)
    bought = []

    for intent in actionable:
        ticker = intent["ticker"]
        price = _next_open_price(ticker, provider, after)
        if price is None or price <= 0:
            continue
        if cfg.gap_filter_pct is not None:
            planned = intent.get("entry", 0.0)
            if planned and planned > 0 and price > planned * (1 + cfg.gap_filter_pct / 100):
                continue
        shares = int(allocation / price)
        if shares <= 0:
            continue
        cost = price * shares
        if cost > portfolio.cash:
            continue
        portfolio.buy(ticker, buy_date, price, shares, stop_price=intent.get("stop"))
        bought.append(ticker)

    return bought


# ─────────────────────────────────────────────────────────────────────────────
# MONITOR STEP  (each day D while positions are held)
# ─────────────────────────────────────────────────────────────────────────────

def _run_monitor_step(
        portfolio: PortfolioState,
        provider: HistoricalSliceProvider,
        sim_date: pd.Timestamp,
        cfg: BacktestConfig,
        exit_params: Optional["ExitParams"] = None,
) -> List[str]:
    """
    Run position_monitor.compute_signals on each open position using
    data up to sim_date.  Execute sells at today's close price.

    Returns list of tickers sold.
    """
    from position_monitor import compute_signals, Position

    sold = []

    for ticker, pos in list(portfolio.open_positions.items()):
        try:
            df = provider.get(ticker, as_of=sim_date)
        except KeyError:
            continue

        if df.empty or len(df) < 25:
            continue

        df.index = pd.to_datetime(df.index).tz_localize(None)
        pm_pos = Position(
            ticker=pos.ticker,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            shares=float(pos.shares),
        )

        result = compute_signals(pm_pos, df, exit_params=exit_params,
                                 planned_stop=pos.stop_price)  # no today_bar
        if result.get("status") == "SELL":
            sell_price = _day_close_price(ticker, provider, sim_date)
            if sell_price is None or sell_price <= 0:
                sell_price = pos.entry_price  # fallback: flat exit
            portfolio.sell(ticker, sim_date.date(), sell_price)
            sold.append(ticker)

    return sold


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class BacktestRunner:
    """
    Runs the full day-by-day backtest simulation.

    Parameters
    ----------
    cfg : BacktestConfig
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def run(self, verbose: bool = True) -> BacktestResults:
        cfg = self.cfg

        # ── 1. Pre-load all historical data once ─────────────────────────────
        if cfg._provider is not None:
            provider = cfg._provider
        else:
            if verbose:
                print(f"[Backtest] Pre-loading data "
                      f"{cfg.start_date} → {cfg.end_date} "
                      f"({len(cfg.tickers)} tickers)...")
            provider = HistoricalSliceProvider.from_yfinance(
                tickers=cfg.tickers,
                start=_lookback_start(cfg.start_date, cfg.lookback_days),
                end=cfg.end_date,
            )
            if verbose:
                print(f"[Backtest] Loaded: {provider}")

        # ── 2. Initialise state ───────────────────────────────────────────────
        portfolio = PortfolioState(initial_cash=cfg.initial_cash)
        from schema_keys import SIGNAL_DB_COLS
        signal_db = pd.DataFrame(columns=SIGNAL_DB_COLS)  # in-memory signal history
        day_logs: List[DayLog] = []
        all_trades: List[ClosedTrade] = []

        trading_days = _trading_days(cfg.start_date, cfg.end_date)
        if verbose:
            print(f"[Backtest] Simulating {len(trading_days)} trading days...\n")

        # ── 3. Day loop ───────────────────────────────────────────────────────
        pending_intents: List[Dict] = []  # intents from D, executed at D+1 open
        cached_screener_df: pd.DataFrame = pd.DataFrame()  # reused between screener runs
        last_screener_day: int = -cfg.screener_frequency  # force run on day 0
        n_days = len(trading_days)
        progress_every = max(1, n_days // 20)  # ~20 progress lines across the run

        for i, day in enumerate(trading_days):
            sim_ts = pd.Timestamp(day)

            # ── Progress output ───────────────────────────────────────────────
            if verbose and (i % progress_every == 0 or i == n_days - 1):
                pct = (i + 1) / n_days * 100
                open_val_now = _mark_to_market(
                    portfolio.open_positions, provider, sim_ts)
                eq_now = portfolio.cash + open_val_now
                ret = (eq_now / cfg.initial_cash - 1) * 100
                print(f"  [{pct:5.1f}%] {day}  "
                      f"equity=${eq_now:>10,.0f}  ret={ret:+.1f}%  "
                      f"open={len(portfolio.open_positions)}  "
                      f"trades={len(all_trades)}")

            # ── 3a. Execute pending buy intents from yesterday ────────────────
            if pending_intents:
                set_backtest_clock(
                    datetime(day.year, day.month, day.day,
                             cfg.next_open_hour, cfg.next_open_min,
                             tzinfo=TSX_TZ)
                )
                prev_day = trading_days[i - 1] if i > 0 else day
                prev_ts = pd.Timestamp(prev_day)

                bought = _execute_buys(
                    intents=pending_intents,
                    portfolio=portfolio,
                    provider=provider,
                    after=prev_ts,
                    buy_date=day,
                    cfg=cfg,
                )
                if verbose and bought:
                    print(f"    -> BUY  {bought}")
                pending_intents = []

            # ── 3b. Monitor existing positions (EOD) ─────────────────────────
            set_backtest_clock(
                datetime(day.year, day.month, day.day,
                         cfg.after_close_hour, cfg.after_close_min,
                         tzinfo=TSX_TZ)
            )

            sold = _run_monitor_step(portfolio, provider, sim_ts, cfg, cfg.exit_params)
            if verbose and sold:
                print(f"    -> SELL {sold}")

            for trade in portfolio.trade_log[-len(sold):] if sold else []:
                all_trades.append(trade)

            # ── 3c. Screener (every screener_frequency days) ──────────────────
            if cfg._screener_cache is not None:
                # Sweep mode: use pre-computed scores (key = day index rounded
                # down to nearest screener_frequency boundary)
                cache_key = (i // cfg.screener_frequency) * cfg.screener_frequency
                cached_screener_df = cfg._screener_cache.get(
                    cache_key, cached_screener_df)
            elif i - last_screener_day >= cfg.screener_frequency:
                cached_screener_df = _run_screener_step(cfg, provider, sim_ts)
                last_screener_day = i

            # ── 3d. Pipeline: pattern detection (every day) ───────────────────
            # Limit to top-N by score to match live system (max_tracked_tickers)
            pipeline_df = (
                cached_screener_df.head(cfg.max_tracked_tickers)
                if not cached_screener_df.empty else cached_screener_df
            )
            new_intents, signal_db = _run_pipeline_step(
                pipeline_df, provider, sim_ts, cfg, signal_db
            )
            pending_intents = new_intents

            if verbose and new_intents:
                confirmed = [intent["ticker"] for intent in new_intents]
                print(f"    -> SIGNAL confirmed={confirmed}")

            # ── 3e. Log equity ────────────────────────────────────────────────
            open_val = _mark_to_market(portfolio.open_positions, provider, sim_ts)
            day_logs.append(DayLog(
                sim_date=day,
                cash=portfolio.cash,
                open_value=open_val,
                total_equity=portfolio.cash + open_val,
                realized_pnl=portfolio.realized_pnl,
                open_tickers=list(portfolio.open_positions.keys()),
                buys_today=[],
                sells_today=sold,
            ))

        # ── 4. Restore live clock ─────────────────────────────────────────────
        set_backtest_clock(None)

        if verbose:
            print()

        return BacktestResults(cfg=cfg, day_logs=day_logs, trades=all_trades)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lookback_start(start_date: str, lookback_days: int) -> str:
    """Return the date we need to fetch from to have `lookback_days` bars at start_date."""
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    # Add a 20% buffer for weekends/holidays
    fetch_from = dt - timedelta(days=int(lookback_days * 1.4))
    return fetch_from.strftime("%Y-%m-%d")
