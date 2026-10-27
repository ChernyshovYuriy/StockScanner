"""
Phase 2 of the idle-cash plan (reports/idle_cash/): BacktestConfig switches
for trading costs, cash yield, the conservative same-bar rule and gap fills.
Every default must leave the backtest unchanged; each switch is checked on
small synthetic bars.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest_runner import BacktestConfig, BacktestRunner, _execute_buys, _run_monitor_step
from market_data import HistoricalSliceProvider
from portfolio import PortfolioState
from position_monitor import ExitParams

DAYS = pd.bdate_range("2026-05-01", periods=40)


def _flat(price: float = 40.0) -> pd.DataFrame:
    return pd.DataFrame({"Open": price, "High": price + 0.5, "Low": price - 0.5, "Close": price,
                         "Volume": 100_000.0}, index=DAYS)


def _cfg(**kw) -> BacktestConfig:
    return BacktestConfig(tickers=["AAA"], sizing="live", max_positions=1, risk_pct=1.0, **kw)


def _buy(cfg: BacktestConfig, open_price: float = 40.0) -> PortfolioState:
    df = _flat()
    df.iloc[-1, df.columns.get_loc("Open")] = open_price
    portfolio = PortfolioState(10_000.0)
    bought = _execute_buys([{"ticker": "AAA", "entry": 40.0, "stop": 36.0, "pattern": "BASE"}], portfolio,
                           HistoricalSliceProvider({"AAA": df}), DAYS[-2], DAYS[-1].date(), cfg)
    assert bought == ["AAA"]
    return portfolio


# ── defaults ─────────────────────────────────────────────────────────────────

def test_defaults_are_the_old_behaviour():
    cfg = BacktestConfig(tickers=["AAA"])
    assert (cfg.costs_enabled, cfg.cash_yield_source, cfg.same_bar_rule, cfg.gap_fills) == \
           (False, "none", "current", False)
    p = _buy(_cfg())
    assert p.open_positions["AAA"].entry_price == 40.0
    assert p.costs == 0.0


# ── costs ────────────────────────────────────────────────────────────────────

def test_buy_pays_spread_in_price_and_commission_from_cash():
    p = _buy(_cfg(costs_enabled=True, stock_spread_bps=10.0, commission_per_order=5.0))
    pos = p.open_positions["AAA"]
    assert pos.entry_price == pytest.approx(40.04)              # 40 * (1 + 10 bps)
    assert p.cash == pytest.approx(10_000.0 - pos.shares * 40.04 - 5.0)
    assert p.costs == pytest.approx(pos.shares * 0.04 + 5.0)


def test_sell_receives_less_by_the_spread_and_pays_commission():
    cfg = _cfg(costs_enabled=True, stock_spread_bps=10.0, commission_per_order=5.0)
    p = PortfolioState(10_000.0)
    p.buy("AAA", DAYS[30].date(), 40.0, 100, stop_price=45.0)    # close 40 is below the stop -> SELL
    sold = _run_monitor_step(p, HistoricalSliceProvider({"AAA": _flat()}), DAYS[-1], cfg)
    assert sold == ["AAA"]
    trade = p.trade_log[-1]
    assert trade.sell_price == pytest.approx(40.0 * 0.999)
    assert p.cash == pytest.approx(10_000.0 - 4_000.0 + 100 * 39.96 - 5.0)
    assert p.costs == pytest.approx(100 * 0.04 + 5.0)


# ── cash yield ───────────────────────────────────────────────────────────────

def _run_idle(cfg_kw: dict) -> "BacktestResults":  # noqa: F821
    provider = HistoricalSliceProvider({"XIU.TO": _flat()})
    cfg = BacktestConfig(tickers=["XIU.TO"], start_date=str(DAYS[30].date()), end_date=str(DAYS[35].date()),
                         initial_cash=10_000.0, _provider=provider,
                         _screener_cache={0: pd.DataFrame()},   # no candidates: cash stays idle
                         **cfg_kw)
    return BacktestRunner(cfg).run(verbose=False)


def test_cash_yield_accrues_daily_on_uninvested_cash():
    returns = {d.date(): 0.001 for d in DAYS}
    res = _run_idle(dict(cash_yield_source="cash_etf", cash_yield_returns=returns))
    eq = res.equity_curve_df()
    assert len(eq) == 5
    assert eq["cash"].iloc[-1] == pytest.approx(10_000.0 * 1.001 ** 5, abs=0.01)   # logged to the cent
    assert res.cash_yield == pytest.approx(10_000.0 * (1.001 ** 5 - 1), rel=1e-9)


def test_cash_yield_off_by_default():
    res = _run_idle({})
    assert res.equity_curve_df()["cash"].iloc[-1] == 10_000.0
    assert res.cash_yield == 0.0


def test_unknown_option_rejected():
    with pytest.raises(ValueError):
        _run_idle(dict(cash_yield_source="bonds"))
    with pytest.raises(ValueError):
        _run_idle(dict(same_bar_rule="optimistic"))


# ── conservative same-bar rule, gap fills ────────────────────────────────────

def _held(today_open: float, today_low: float, today_high: float, today_close: float) -> pd.DataFrame:
    df = _flat()
    df.iloc[-1] = [today_open, today_high, today_low, today_close, 100_000.0]
    return df


def _monitor(df: pd.DataFrame, **kw) -> PortfolioState:
    p = PortfolioState(10_000.0)
    p.buy("AAA", DAYS[30].date(), 40.0, 100, stop_price=38.0)
    _run_monitor_step(p, HistoricalSliceProvider({"AAA": df}), DAYS[-1], _cfg(**kw), ExitParams())
    return p


def test_current_rule_ignores_an_intraday_breach_that_closes_above():
    p = _monitor(_held(39.0, 37.0, 40.0, 39.5))
    assert "AAA" in p.open_positions


def test_conservative_rule_fills_the_resting_stop_on_an_intraday_breach():
    p = _monitor(_held(39.0, 37.0, 40.0, 39.5), same_bar_rule="conservative")
    trade = p.trade_log[-1]
    assert trade.sell_price == pytest.approx(38.0)
    assert trade.exit_reason.startswith("STOP_RESTING")


def test_conservative_stop_level_ignores_todays_high():
    # Today's spike to 60 would arm the chandelier and lift the stop near 58
    # if today's bar counted; the resting order was placed before it, at 38.
    p = _monitor(_held(40.0, 39.0, 60.0, 40.0), same_bar_rule="conservative")
    assert "AAA" in p.open_positions


def test_gap_through_the_stop_fills_at_the_open_only_with_gap_fills():
    gapped = _held(35.0, 34.0, 36.0, 35.5)
    assert _monitor(gapped, same_bar_rule="conservative").trade_log[-1].sell_price == pytest.approx(38.0)
    assert _monitor(gapped, same_bar_rule="conservative", gap_fills=True).trade_log[-1].sell_price == \
        pytest.approx(35.0)


def test_conservative_rule_still_applies_close_based_exits():
    # Low never reaches the resting stop, but the time stop is due (20 days, not profitable).
    df = _flat()
    p = PortfolioState(10_000.0)
    p.buy("AAA", DAYS[10].date(), 41.0, 100, stop_price=30.0)
    _run_monitor_step(p, HistoricalSliceProvider({"AAA": df}), DAYS[-1],
                      _cfg(same_bar_rule="conservative"), ExitParams())
    assert p.trade_log[-1].exit_reason.startswith("TIME_STOP")
    assert p.trade_log[-1].sell_price == pytest.approx(40.0)
