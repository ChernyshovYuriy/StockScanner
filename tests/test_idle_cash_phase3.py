"""
Phase 3 of the idle-cash plan (reports/idle_cash/03_matrix/DECISION_RULES.md):
BacktestConfig.idle_cash_mode parks uninvested money in the benchmark while
the swing book keeps sizing from the shadow ledger (PortfolioState.cash).
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest_runner import BacktestConfig, BacktestRunner, _IdleCashAccount, _execute_buys
from market_data import HistoricalSliceProvider
from portfolio import PortfolioState

DAYS = pd.bdate_range("2025-01-01", periods=260)


def _bars(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close,
                         "Volume": 100_000.0}, index=close.index)


def _xiu(trend: float) -> pd.DataFrame:
    """XIU.TO bars rising (trend > 0) or falling (trend < 0) by `trend` a day from 40."""
    return _bars(pd.Series([40.0 + trend * i for i in range(len(DAYS))], index=DAYS))


def _run(mode: str, xiu: pd.DataFrame, **kw):
    provider = HistoricalSliceProvider({"XIU.TO": xiu})
    cfg = BacktestConfig(tickers=["XIU.TO"], start_date=str(DAYS[250].date()), end_date=str(DAYS[255].date()),
                         initial_cash=10_000.0, _provider=provider, _screener_cache={0: pd.DataFrame()},
                         idle_cash_mode=mode, **kw)
    return BacktestRunner(cfg).run(verbose=False)


# ── defaults ─────────────────────────────────────────────────────────────────

def test_default_mode_keeps_everything_in_cash():
    res = _run("cash", _xiu(0.05))
    eq = res.equity_curve_df()
    assert BacktestConfig(tickers=["X"]).idle_cash_mode == "cash"
    assert (eq["xiu_value"] == 0).all() and (eq["cash"] == 10_000.0).all()
    assert res.idle_log is None


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        _run("bonds", _xiu(0.05))


# ── parking ──────────────────────────────────────────────────────────────────

def test_xiu_always_parks_all_but_the_buffer_at_the_first_open():
    xiu = _xiu(0.05)
    res = _run("xiu_always", xiu, costs_enabled=True, stock_spread_bps=10.0, xiu_spread_bps=2.0)
    first_open = float(xiu["Open"].iloc[250])
    shares = int((10_000.0 - 100.0) / (first_open * 1.0002))
    log = res.idle_log
    assert log["xiu_shares"].iloc[0] == shares
    assert res.costs == pytest.approx(shares * first_open * 0.0002)
    eq = res.equity_curve_df()
    last_close = float(xiu["Close"].iloc[254])
    assert eq["total_equity"].iloc[-1] == pytest.approx(eq["cash"].iloc[-1] + shares * last_close, abs=0.01)


def test_xiu_gated_stays_in_cash_below_the_200_day_and_earns_yield():
    returns = {d.date(): 0.001 for d in DAYS}
    res = _run("xiu_gated", _xiu(-0.05), cash_yield_source="cash_etf", cash_yield_returns=returns)
    assert (res.idle_log["xiu_shares"] == 0).all()
    assert res.equity_curve_df()["cash"].iloc[-1] == pytest.approx(10_000.0 * 1.001 ** 5, abs=0.01)


def test_xiu_gated_parks_above_the_200_day():
    res = _run("xiu_gated", _xiu(0.05))
    assert res.idle_log["xiu_shares"].iloc[0] > 0


# ── funding from the parked benchmark ────────────────────────────────────────

def _account(cash: float, shares: int, open_price: float, **kw) -> _IdleCashAccount:
    acct = _IdleCashAccount(cash, BacktestConfig(tickers=["X"], **kw))
    acct.shares = shares
    acct.begin_morning(DAYS[0].date(), open_price, True, cash + shares * open_price)
    return acct


def test_morning_buys_sell_just_enough_xiu_once():
    acct = _account(100.0, 300, 40.0)                 # equity 12,100, buffer 121
    assert acct.can_fund(5_000.0)
    acct.reserve(3_000.0)
    acct.reserve(2_000.0)
    acct.settle_morning()
    sold = 300 - acct.shares
    assert sold == 126                                 # ceil((4,900 + 121) / 40)
    assert acct.cash == pytest.approx(100.0 + sold * 40.0 - 5_000.0)
    assert acct.log == [] and acct.traded == pytest.approx(sold * 40.0)   # one order


def test_a_buy_the_real_account_cant_fund_is_skipped():
    acct = _account(100.0, 10, 40.0)                  # real funds 500
    shadow = PortfolioState(10_000.0)                  # shadow ledger still thinks 10k
    stock = _bars(pd.Series(50.0, index=DAYS[:40]))
    cfg = BacktestConfig(tickers=["AAA"], sizing="live", max_positions=1, risk_pct=1.0)
    bought = _execute_buys([{"ticker": "AAA", "entry": 50.0, "stop": 45.0, "pattern": "BASE"}], shadow,
                           HistoricalSliceProvider({"AAA": stock}), DAYS[38], DAYS[39].date(), cfg, idle=acct)
    assert bought == [] and acct.skipped_buys == 1
    assert shadow.cash == 10_000.0


def test_sizing_comes_from_the_shadow_ledger_not_the_xiu_value():
    stock = _bars(pd.Series(50.0, index=DAYS[:40]))
    intent = [{"ticker": "AAA", "entry": 50.0, "stop": 45.0, "pattern": "BASE"}]
    cfg = BacktestConfig(tickers=["AAA"], sizing="live", max_positions=1, risk_pct=1.0)
    sizes = []
    for xiu_shares in (300, 1_000):                    # real account richer in the second case
        shadow = PortfolioState(10_000.0)
        acct = _account(0.0, xiu_shares, 40.0)
        _execute_buys(intent, shadow, HistoricalSliceProvider({"AAA": stock}), DAYS[38], DAYS[39].date(),
                      cfg, idle=acct)
        sizes.append(shadow.open_positions["AAA"].shares)
    assert sizes[0] == sizes[1] == 20                  # 1% of 10k / (50 - 45)


def test_exit_proceeds_are_parked_at_the_close_above_the_threshold():
    acct = _account(0.0, 0, 40.0)
    acct.settle_morning()
    acct.receive(5_000.0)
    acct.end_of_day(40.0, 0.0, 5_000.0)                # equity 10,000: buffer 100, threshold 300
    assert acct.shares == int(4_900 / 40.0)
    small = _account(0.0, 0, 40.0)
    small.settle_morning()
    small.receive(250.0)
    small.end_of_day(40.0, 0.0, 9_750.0)               # 150 excess < 300 threshold
    assert small.shares == 0 and small.cash == 250.0


def test_closed_park_gate_sells_everything_at_the_open():
    acct = _IdleCashAccount(0.0, BacktestConfig(tickers=["X"]))
    acct.shares = 100
    acct.begin_morning(DAYS[0].date(), 40.0, False, 4_000.0)
    acct.settle_morning()
    assert acct.shares == 0 and acct.cash == pytest.approx(4_000.0)
