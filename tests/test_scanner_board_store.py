"""
tests/test_scanner_board_store.py
===================================
Tests for scanner_board.store — see scanner_board/PLAN.md Phase 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner_board import row, store
from scanner_board.weekly import resample_ohlcv_weekly


def _synthetic_daily(n: int, start="2023-01-02", seed: int = 3) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(0.05 + rng.normal(0, 0.3, n))
    high = close + np.abs(rng.normal(0.2, 0.1, n))
    low = close - np.abs(rng.normal(0.2, 0.1, n))
    open_ = close - rng.normal(0, 0.1, n)
    volume = np.abs(rng.normal(100_000, 10_000, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "scanner_board_test.db")
    yield c
    c.close()


def _real_row(ticker="TEST") -> dict:
    daily = _synthetic_daily(260)
    weekly = resample_ohlcv_weekly(daily)
    return row.compute_row(ticker, daily, weekly)


class TestConnectSchema:
    def test_connect_creates_table_idempotently(self, tmp_path):
        db_path = tmp_path / "idempotent.db"
        c1 = store.connect(db_path)
        c1.close()
        c2 = store.connect(db_path)  # must not raise on an already-migrated file
        c2.close()

    def test_row_value_columns_has_no_duplicates_between_numeric_and_label(self):
        assert set(store.NUMERIC_COLUMNS).isdisjoint(set(store.LABEL_COLUMNS))
        assert len(store.ROW_VALUE_COLUMNS) == len(store.NUMERIC_COLUMNS) + len(store.LABEL_COLUMNS)


class TestUpsertAndRead:
    def test_round_trip_a_real_computed_row(self, conn):
        r = _real_row("AAA.TO")
        store.upsert_rows(conn, "2026-09-15", [r], updated_at="2026-09-15T17:15:00")
        got = store.rows_for_run_date(conn, "2026-09-15")
        assert len(got) == 1
        assert got[0]["ticker"] == "AAA.TO"
        assert got[0]["ma22"] == pytest.approx(r["ma22"])
        assert got[0]["trend_health"] == r["trend_health"]

    def test_none_values_round_trip_as_none(self, conn):
        """A short-history ticker's NaN-derived columns store as SQL NULL
        and read back as None, not 0.0 or the string 'None'."""
        daily = _synthetic_daily(5)
        weekly = resample_ohlcv_weekly(daily)
        r = row.compute_row("SHORT.TO", daily, weekly)
        store.upsert_rows(conn, "2026-09-15", [r], updated_at="2026-09-15T17:15:00")
        got = store.rows_for_run_date(conn, "2026-09-15")[0]
        for col, value in r.items():
            if col == "ticker":
                continue
            if value is None:
                assert got[col] is None

    def test_rerun_same_day_replaces_not_duplicates(self, conn):
        r1 = _real_row("BBB.TO")
        r2 = dict(r1)
        r2["price"] = 12345.0
        store.upsert_rows(conn, "2026-09-15", [r1], updated_at="t1")
        store.upsert_rows(conn, "2026-09-15", [r2], updated_at="t2")
        got = store.rows_for_run_date(conn, "2026-09-15")
        assert len(got) == 1
        assert got[0]["price"] == 12345.0
        assert got[0]["updated_at"] == "t2"

    def test_different_days_both_kept(self, conn):
        r = _real_row("CCC.TO")
        store.upsert_rows(conn, "2026-09-14", [r], updated_at="t1")
        store.upsert_rows(conn, "2026-09-15", [r], updated_at="t2")
        assert len(store.rows_for_run_date(conn, "2026-09-14")) == 1
        assert len(store.rows_for_run_date(conn, "2026-09-15")) == 1

    def test_full_transaction_rolls_back_on_a_bad_row(self, conn):
        """If any row in the batch is malformed, none of the batch should
        land -- upsert_rows() is one transaction, not row-by-row
        autocommit."""
        good = _real_row("DDD.TO")
        bad = {"ticker": "EEE.TO"}  # missing every value column -> .get()
        # returns None for each, which IS valid (stores as NULL) -- use a
        # genuinely malformed row instead: wrong column name that can't
        # bind, forcing an sqlite3 error.
        bad_type_row = dict(good)
        bad_type_row["ticker"] = object()  # unbindable sqlite3 type
        with pytest.raises(Exception):
            store.upsert_rows(conn, "2026-09-15", [good, bad_type_row], updated_at="t1")
        assert store.rows_for_run_date(conn, "2026-09-15") == []


class TestLatest:
    def test_latest_run_date_and_rows_on_empty_db(self, conn):
        assert store.latest_run_date(conn) is None
        assert store.latest_rows(conn) == []

    def test_latest_rows_picks_the_max_run_date(self, conn):
        r = _real_row("FFF.TO")
        store.upsert_rows(conn, "2026-09-10", [r], updated_at="t0")
        store.upsert_rows(conn, "2026-09-15", [r], updated_at="t1")
        store.upsert_rows(conn, "2026-09-12", [r], updated_at="t2")
        assert store.latest_run_date(conn) == "2026-09-15"
        got = store.latest_rows(conn)
        assert len(got) == 1
        assert got[0]["updated_at"] == "t1"

    def test_latest_rows_returns_every_ticker_for_that_date(self, conn):
        r1, r2 = _real_row("GGG.TO"), _real_row("HHH.TO")
        store.upsert_rows(conn, "2026-09-15", [r1, r2], updated_at="t1")
        got = store.latest_rows(conn)
        assert {row_["ticker"] for row_ in got} == {"GGG.TO", "HHH.TO"}
