"""
tests/test_entry_session_label.py
==================================
The pipeline report names the actual next TSX session for a CONFIRMED
entry instead of "tomorrow" (wrong on a Friday or before a holiday).
"""

from __future__ import annotations

from datetime import date

from report_html import _entry_session, write_pipeline_report
from time_utils import next_trading_day


def test_next_trading_day_skips_weekend_and_thanksgiving():
    # Fri 2026-10-09 -> Sat, Sun, Mon 10-12 Thanksgiving -> Tue 10-13
    assert next_trading_day(date(2026, 10, 9)) == date(2026, 10, 13)


def test_next_trading_day_ordinary_weekday():
    assert next_trading_day(date(2026, 10, 14)) == date(2026, 10, 15)


def test_next_trading_day_plain_friday():
    assert next_trading_day(date(2026, 10, 16)) == date(2026, 10, 19)


def test_entry_session_label():
    assert _entry_session("2026-10-09") == "Tue Oct 13"
    assert _entry_session("not-a-date") == "Next Session"


def test_pipeline_report_names_the_session(tmp_path):
    path = tmp_path / "report.html"
    alert = {"ticker": "CAS.TO", "state": "CONFIRMED", "pattern": "PB-EMA21"}
    write_pipeline_report(path=str(path), date_str="2026-10-09", account_size=4770, risk_pct=1.0,
                          n_tracked=40, alerts=[alert], db_records=[])
    html = path.read_text()
    assert "Enter Tue Oct 13 Open" in html
    assert "Tomorrow" not in html
