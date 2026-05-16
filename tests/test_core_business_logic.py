"""
tests/test_core_business_logic.py
==================================
Tests for core business logic:
  - Funds summary HTML (gain vs loss labelling)
  - compute_signals exit-rule engine (stop hit, chandelier, giveback, time stop, hold)
  - execute_virtual_sells P&L accounting
  - append_positions_report ordering (newest day first)

Run with:
    pytest tests/test_core_business_logic.py -v
"""

from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that aren't installed in the test env
# (yfinance requires multitasking which cannot be built on this platform).
# We only need the pure-Python business logic; no live market data is fetched.
# ---------------------------------------------------------------------------

def _stub_modules():
    """Stub heavy/unavailable optional dependencies before any project import."""
    # yfinance — requires multitasking which can't be built here
    yf = types.ModuleType("yfinance")
    yf.download = MagicMock(return_value=pd.DataFrame())
    sys.modules.setdefault("yfinance", yf)

    # python-dotenv — optional at runtime, not installed in test env
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = MagicMock()
    sys.modules.setdefault("dotenv", dotenv)

    # colorama — may not be installed
    colorama = types.ModuleType("colorama")
    colorama.Fore = MagicMock()
    colorama.Style = MagicMock()
    colorama.init = MagicMock()
    sys.modules.setdefault("colorama", colorama)


_stub_modules()


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_daily_ohlcv(
        n: int = 60,
        start: str = "2025-01-01",
        close_prices: list[float] | None = None,
        atr_approx: float = 1.0,
) -> pd.DataFrame:
    """
    Build a minimal daily OHLCV DataFrame.

    If *close_prices* is given it must have exactly *n* elements.
    Otherwise a flat series at 100.0 is used.
    Highs/Lows are ±atr_approx/2 around close so ATR ≈ atr_approx.
    """
    idx = pd.bdate_range(start, periods=n)
    if close_prices is not None:
        assert len(close_prices) == n
        close = pd.Series(close_prices, index=idx, dtype=float)
    else:
        close = pd.Series([100.0] * n, index=idx, dtype=float)
    spread = atr_approx / 2
    high = close + spread
    low = close - spread
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": 300_000.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# 1. Funds summary HTML — sell proceeds / realised P&L / totals
# ---------------------------------------------------------------------------

class TestFundsSummaryHtml:
    """
    Original bug: 'Sold (gained)' label appeared even when positions were closed
    at a loss because it was based on gross proceeds, not actual P&L.

    Current behaviour:
    - 'Sold proceeds'   → gross cash returned (always positive when sells occurred)
    - 'Realised P&L'    → separate line with actual profit or loss (can be negative)
    - 'Total funds available' → cash + open-position market value (always shown)
    - 'Total gain/loss'       → realised P&L + unrealised P&L (always shown)
    """

    def _call(self, funds_before, funds_after, funds_gained, realized_pnl=0.0,
              unrealised_pnl=0.0, unrealised_position_value=0.0):
        from position_monitor import _funds_summary_html
        return _funds_summary_html(
            funds_before=funds_before,
            funds_after=funds_after,
            funds_gained=funds_gained,
            realized_pnl=realized_pnl,
            unrealised_pnl=unrealised_pnl,
            unrealised_position_value=unrealised_position_value,
        )

    def test_profitable_sell_shows_positive_pnl(self):
        """Profitable sell: 'Sold proceeds' and positive 'Realised P&L' must appear."""
        html = self._call(
            funds_before=1000.0,
            funds_after=1150.0,
            funds_gained=1150.0,
            realized_pnl=+150.0,
        )
        assert "Sold proceeds" in html
        assert "Realised P&L" in html
        assert "+$150.00" in html

    def test_losing_sell_shows_negative_pnl(self):
        """
        BUG REGRESSION: previously 'Sold (gained)' appeared even for losing trades.
        Now 'Realised P&L' must show a negative value and the word 'gained' must
        not appear anywhere.
        """
        html = self._call(
            funds_before=1000.0,
            funds_after=850.0,
            funds_gained=850.0,  # gross proceeds (always > 0)
            realized_pnl=-150.0,  # actual loss
        )
        assert "Sold proceeds" in html
        assert "Realised P&L" in html
        assert "-$150.00" in html
        # The old buggy label must never appear
        assert "gained" not in html.lower()

    def test_no_sells_shows_funds_remaining(self):
        html = self._call(
            funds_before=1000.0,
            funds_after=1000.0,
            funds_gained=0.0,
            realized_pnl=0.0,
        )
        assert "Funds remaining" in html
        # Should not show any sell line
        assert "Sold" not in html

    def test_breakeven_sell_shows_realised_pnl_label(self):
        """Exactly zero P&L: 'Realised P&L' must appear with +$0.00."""
        html = self._call(
            funds_before=500.0,
            funds_after=500.0,
            funds_gained=500.0,
            realized_pnl=0.0,
        )
        assert "Realised P&L" in html
        assert "+$0.00" in html

    def test_total_funds_available_always_shown(self):
        """'Total funds available' must appear for both sells and no-sells cases."""
        html_sells = self._call(
            funds_before=1000.0,
            funds_after=1200.0,
            funds_gained=1200.0,
            realized_pnl=+200.0,
            unrealised_position_value=500.0,
        )
        assert "Total funds available" in html_sells
        assert "1,700.00" in html_sells  # 1200 cash + 500 open positions

        html_no_sells = self._call(
            funds_before=1000.0,
            funds_after=1000.0,
            funds_gained=0.0,
            unrealised_position_value=300.0,
        )
        assert "Total funds available" in html_no_sells
        assert "1,300.00" in html_no_sells  # 1000 cash + 300 open positions

    def test_total_gain_loss_always_shown(self):
        """'Total gain/loss' must combine realised and unrealised P&L."""
        html = self._call(
            funds_before=1000.0,
            funds_after=1100.0,
            funds_gained=1100.0,
            realized_pnl=+100.0,
            unrealised_pnl=+50.0,
        )
        assert "Total gain/loss" in html
        assert "+$150.00" in html  # 100 realised + 50 unrealised

    def test_proceeds_dollar_amount_in_html(self):
        """The gross proceeds figure must appear in the output."""
        html = self._call(
            funds_before=5000.0,
            funds_after=4500.0,
            funds_gained=4500.0,
            realized_pnl=-500.0,
        )
        assert "4,500.00" in html

    def test_realized_pnl_dollar_amount_in_html(self):
        """The realized P&L dollar figure must appear in the output."""
        html = self._call(
            funds_before=1000.0,
            funds_after=1200.0,
            funds_gained=1200.0,
            realized_pnl=+200.0,
        )
        assert "200.00" in html

    def test_funds_before_always_shown(self):
        html = self._call(1234.56, 1234.56, 0.0)
        assert "1,234.56" in html


# ---------------------------------------------------------------------------
# 2. compute_signals — exit-rule engine
# ---------------------------------------------------------------------------

class TestComputeSignals:
    """Unit tests for position_monitor.compute_signals."""

    @pytest.fixture(autouse=True)
    def _imports(self):
        from position_monitor import compute_signals, Position, ExitParams
        self.compute_signals = compute_signals
        self.Position = Position
        self.ExitParams = ExitParams

    def _pos(self, entry_price: float = 100.0, entry_date: str = "2025-01-02",
             shares: float = 100.0, ticker: str = "TEST.TO") -> object:
        return self.Position(
            ticker=ticker,
            entry_date=date.fromisoformat(entry_date),
            entry_price=entry_price,
            shares=shares,
        )

    # ── 2a. HOLD — no exit triggered ─────────────────────────────────────────

    def test_hold_no_exit_triggered(self):
        """Rising price, no stop, no giveback, well inside time limit → HOLD."""
        df = _make_daily_ohlcv(n=30, close_prices=[100.0 + i * 0.5 for i in range(30)],
                               atr_approx=1.0)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(
            initial_stop_atr_k=1.5,
            chand_trail_atr_k=2.5,
            giveback_activate_pct=3.0,
            giveback_allow_pct=2.0,
            time_stop_days=30,
            time_stop_min_profit=0.5,
            stop_trigger="close",
        )
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["status"] == "HOLD"
        assert result["reason"] == "OK"

    # ── 2b. Initial stop hit ──────────────────────────────────────────────────

    def test_initial_stop_hit(self):
        """
        Price drops sharply below entry - 1.5×ATR → stop triggered.
        ATR ≈ 1.0 so stop ≈ 100 - 1.5 = 98.5.
        Last close at 97 should fire.
        """
        closes = [100.0] * 29 + [97.0]
        df = _make_daily_ohlcv(n=30, close_prices=closes, atr_approx=1.0)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(initial_stop_atr_k=1.5, stop_trigger="close")
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["status"] == "SELL"
        assert "STOP_HIT" in result["reason"]

    def test_price_above_stop_does_not_sell(self):
        """Close just above initial stop → HOLD."""
        closes = [100.0] * 29 + [99.0]  # 99 > 98.5 stop
        df = _make_daily_ohlcv(n=30, close_prices=closes, atr_approx=1.0)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(initial_stop_atr_k=1.5, stop_trigger="close",
                             time_stop_days=200)
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["status"] == "HOLD"

    # ── 2c. Profit giveback ───────────────────────────────────────────────────

    def test_giveback_triggers_sell(self):
        """
        Price runs to +5% then drops back to +2% → giveback fires
        (activated at 3%, allows 2% retracement, peak-now = 3% > 2%).
        """
        # 29 bars rising from 100 → 105, then last bar at 102
        rising = [100.0 + i * (5.0 / 28) for i in range(29)]
        closes = rising + [102.0]
        df = _make_daily_ohlcv(n=30, close_prices=closes, atr_approx=0.5)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(
            giveback_activate_pct=3.0,
            giveback_allow_pct=2.0,
            initial_stop_atr_k=10.0,  # disable stop hit for this test
            time_stop_days=200,
            stop_trigger="close",
        )
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["status"] == "SELL"
        assert "GIVEBACK" in result["reason"]

    def test_giveback_not_triggered_if_below_threshold(self):
        """
        Peak only reaches +2.5%, below the 3% activation threshold → no giveback.
        Both stop types are disabled (very large k) so only the giveback rule is
        under test.
        """
        rising = [100.0 + i * (2.5 / 28) for i in range(29)]
        closes = rising + [101.0]
        df = _make_daily_ohlcv(n=30, close_prices=closes, atr_approx=0.5)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(
            giveback_activate_pct=3.0,
            giveback_allow_pct=2.0,
            initial_stop_atr_k=100.0,  # far below price — stop disabled
            chand_trail_atr_k=100.0,  # chandelier far below — disabled
            time_stop_days=200,
            stop_trigger="close",
        )
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["status"] == "HOLD"

    # ── 2d. Time stop ─────────────────────────────────────────────────────────

    def test_time_stop_triggers(self):
        """
        After time_stop_days with profit below min → time stop fires.
        """
        closes = [100.0] * 25  # 25 bars, flat (0% profit)
        df = _make_daily_ohlcv(n=25, close_prices=closes, atr_approx=0.5)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(
            time_stop_days=20,
            time_stop_min_profit=0.5,
            initial_stop_atr_k=10.0,
            giveback_activate_pct=50.0,  # disable giveback
            stop_trigger="close",
        )
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["status"] == "SELL"
        assert "TIME_STOP" in result["reason"]

    def test_time_stop_not_triggered_if_profitable(self):
        """Profit above min threshold → time stop does NOT fire."""
        closes = [100.0] * 24 + [101.0]  # last bar +1%
        df = _make_daily_ohlcv(n=25, close_prices=closes, atr_approx=0.5)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(
            time_stop_days=20,
            time_stop_min_profit=0.5,
            initial_stop_atr_k=10.0,
            giveback_activate_pct=50.0,
            stop_trigger="close",
        )
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["status"] == "HOLD"

    # ── 2e. P&L values ────────────────────────────────────────────────────────

    def test_pnl_dollar_calculation(self):
        """pnl_$ = (last_close - entry_price) * shares."""
        closes = [100.0] * 29 + [110.0]
        df = _make_daily_ohlcv(n=30, close_prices=closes, atr_approx=1.0)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat(),
                        shares=50.0)
        ep = self.ExitParams(initial_stop_atr_k=10.0, time_stop_days=200,
                             giveback_activate_pct=50.0)
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["pnl_$"] == pytest.approx(500.0, abs=0.01)  # (110-100)*50

    def test_pnl_pct_calculation(self):
        """pnl_% = (last_close / entry_price - 1) * 100."""
        closes = [100.0] * 29 + [115.0]
        df = _make_daily_ohlcv(n=30, close_prices=closes, atr_approx=1.0)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat())
        ep = self.ExitParams(initial_stop_atr_k=10.0, time_stop_days=200,
                             giveback_activate_pct=50.0)
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["pnl_%"] == pytest.approx(15.0, abs=0.01)

    def test_negative_pnl_when_price_below_entry(self):
        closes = [100.0] * 29 + [90.0]
        df = _make_daily_ohlcv(n=30, close_prices=closes, atr_approx=0.3)
        pos = self._pos(entry_price=100.0, entry_date=df.index[0].date().isoformat(),
                        shares=10.0)
        ep = self.ExitParams(initial_stop_atr_k=100.0, time_stop_days=200,
                             giveback_activate_pct=50.0, stop_trigger="close")
        result = self.compute_signals(pos, df, exit_params=ep)
        assert result["pnl_$"] < 0
        assert result["pnl_%"] < 0

    # ── 2f. No data / edge cases ──────────────────────────────────────────────

    def test_no_data_after_entry(self):
        """If there are no bars on or after entry date the result is NO_DATA."""
        df = _make_daily_ohlcv(n=30, start="2024-01-01")
        pos = self._pos(entry_date="2025-06-01")  # after all bars
        result = self.compute_signals(pos, df)
        assert result["status"] == "NO_DATA"

    def test_empty_dataframe_returns_no_data(self):
        df = pd.DataFrame()
        pos = self._pos()
        result = self.compute_signals(pos, df)
        assert result["status"] == "NO_DATA"


# ---------------------------------------------------------------------------
# 3. execute_virtual_sells — P&L accounting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4. append_positions_report — newest day first ordering
# ---------------------------------------------------------------------------

class TestAppendPositionsReportOrdering:
    """
    Verifies that when multiple days are appended to the same report file the
    newest day appears BEFORE (higher in the HTML) than the older day.
    """

    def _make_row(self, ticker: str = "TST.TO") -> dict:
        return {
            "ticker": ticker,
            "entry_date": "2025-01-02",
            "entry_price": 10.0,
            "last_close": 10.5,
            "pnl_%": 5.0,
            "pnl_$": 50.0,
            "max_pnl_%": 5.0,
            "stop_price": 9.0,
            "ATR14": 0.3,
            "R_mult": 1.7,
            "tdays": 3,
            "status": "HOLD",
            "reason": "OK",
        }

    def test_newest_day_appears_before_oldest(self, tmp_path):
        from report_html import append_positions_report

        report = tmp_path / "report.html"

        # Simulate pipeline writing a skeleton with the placeholder
        report.write_text(
            "<html><body><!-- POSITION_MONITOR_PLACEHOLDER --></body></html>",
            encoding="utf-8",
        )

        # Day 1
        append_positions_report(str(report), "20250102", [self._make_row("AAA.TO")])
        # Day 2
        append_positions_report(str(report), "20250103", [self._make_row("BBB.TO")])
        # Day 3 (newest)
        append_positions_report(str(report), "20250104", [self._make_row("CCC.TO")])

        content = report.read_text(encoding="utf-8")

        pos_day1 = content.index("20250102")
        pos_day2 = content.index("20250103")
        pos_day3 = content.index("20250104")

        # Newest (day 3) must appear first (smallest index in the HTML string)
        assert pos_day3 < pos_day2 < pos_day1, (
            f"Expected newest first but got positions: "
            f"day3={pos_day3}, day2={pos_day2}, day1={pos_day1}"
        )

    def test_single_day_renders_without_placeholder_error(self, tmp_path):
        from report_html import append_positions_report

        report = tmp_path / "report.html"
        report.write_text(
            "<html><body><!-- POSITION_MONITOR_PLACEHOLDER --></body></html>",
            encoding="utf-8",
        )
        # Should not raise
        append_positions_report(str(report), "20250101", [self._make_row()])
        content = report.read_text(encoding="utf-8")
        assert "20250101" in content

    def test_creates_new_file_when_missing(self, tmp_path):
        from report_html import append_positions_report

        report = tmp_path / "new_report.html"
        assert not report.exists()
        append_positions_report(str(report), "20250201", [self._make_row()])
        assert report.exists()
        content = report.read_text(encoding="utf-8")
        assert "20250201" in content
