"""
tests/test_scanner_dashboard_data.py
======================================
Offline tests for scanner_dashboard_data (no network) — see
scanner_board/PLAN.md Phase 5.
"""
from __future__ import annotations

import pytest

import scanner_dashboard_data as sdd
from scanner_board import store


def _seed_row(ticker: str, **overrides) -> dict:
    """A minimal board_snapshot row: every column defaults to None except
    `ticker` and whatever the test overrides — store.upsert_rows() is
    tolerant of a dict missing keys (row.get(c) for c in
    ROW_VALUE_COLUMNS), so tests only need to specify the columns they
    actually care about."""
    row = {"ticker": ticker}
    row.update(overrides)
    return row


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "scanner_dashboard_test.db")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _reset_cache():
    sdd._cache["ts"] = 0.0
    sdd._cache["state"] = None
    yield
    sdd._cache["ts"] = 0.0
    sdd._cache["state"] = None


class TestBadgeClass:
    @pytest.mark.parametrize("label,expected", [
        ("Rising", "badge-bullish"), ("Falling", "badge-bearish"), ("Flat", "badge-neutral"),
        ("Up", "badge-bullish"), ("Down", "badge-bearish"),
        ("Above", "badge-bullish"), ("Below", "badge-bearish"), ("At", "badge-neutral"),
        ("Bull", "badge-bullish"), ("Bear", "badge-bearish"), ("Neutral", "badge-neutral"),
        ("Safe", "badge-bullish"), ("Weak", "badge-bearish"), ("Unclear", "badge-neutral"),
        ("New High", "badge-bullish"), ("New Low", "badge-bearish"), ("No", "badge-neutral"),
        ("High", "badge-bullish"), ("Low", "badge-bearish"), ("Normal", "badge-neutral"),
        ("Positive", "badge-bullish"), ("Negative", "badge-bearish"), ("Zero", "badge-neutral"),
        ("Bullish", "badge-bullish"), ("Bearish", "badge-bearish"), ("None", "badge-neutral"),
        ("Overbought", "badge-mild_bearish"), ("Oversold", "badge-mild_bullish"),
        ("Choppy", "badge-neutral"), ("Trending", "badge-neutral"), ("Unknown", "badge-neutral"),
        ("Waking Up", "badge-highlight"), ("Overheated", "badge-highlight"),
        ("Green", "badge-bullish"), ("Red", "badge-bearish"), ("Blue", "badge-impulse-blue"),
        ("Stand aside", "badge-neutral"), ("Go long setup", "badge-bullish"), ("Go short setup", "badge-bearish"),
    ])
    def test_every_known_word(self, label, expected):
        assert sdd.badge_class(label) == expected

    def test_none_is_neutral(self):
        assert sdd.badge_class(None) == "badge-neutral"

    def test_unknown_word_is_neutral_not_a_crash(self):
        assert sdd.badge_class("some future Enum value") == "badge-neutral"

    def test_every_actual_enum_value_across_scanner_board_is_covered(self):
        """Cross-checks _WORD_BADGE against every real Enum value the
        Board's own modules can actually produce (excluding value_zone's
        ValueZonePosition, deliberately unbadged — see module docstring),
        so a newly added Enum member can't silently fall through to the
        generic neutral default unnoticed."""
        from scanner_board.slope import SlopeDirection
        from scanner_board.divergence import DivergenceType
        from scanner_board.thesis_rules import (
            PriceVsMA, MACDCross, TrendHealth, ExtremeReading, DIBias, ADXRegime,
            OscillatorZone, VolumeLevel, ForceZone, ForceBias,
        )
        from scanner_board.triple_screen import ImpulseColor, TripleScreenAlignment

        enums = [
            SlopeDirection, DivergenceType, PriceVsMA, MACDCross, TrendHealth,
            ExtremeReading, DIBias, ADXRegime, OscillatorZone, VolumeLevel,
            ForceZone, ForceBias, ImpulseColor, TripleScreenAlignment,
        ]
        for enum_cls in enums:
            for member in enum_cls:
                assert member.value in sdd._WORD_BADGE, \
                    f"{enum_cls.__name__}.{member.name} ({member.value!r}) has no badge mapping"


class TestSlopeGlyph:
    @pytest.mark.parametrize("label,expected", [
        ("Rising", "▲"), ("Falling", "▼"), ("Flat", "–"),
        ("Up", "▲"), ("Down", "▼"),
    ])
    def test_known_directions(self, label, expected):
        assert sdd.slope_glyph(label) == expected

    def test_none_and_unrelated_word_are_blank_not_a_crash(self):
        assert sdd.slope_glyph(None) == ""
        assert sdd.slope_glyph("Bull") == ""


class TestScannerCriteriaColumns:
    """The /scanner "Screen & sort" panel's column metadata (Phase 7
    add-on) -- see scanner_board.js for how the panel consumes this."""

    def test_every_entry_has_the_expected_shape(self):
        cols = sdd.scanner_criteria_columns()
        assert len(cols) > 0
        for c in cols:
            assert set(c.keys()) == {"key", "label", "kind", "group", "options"}
            assert c["kind"] in ("number", "label", "text")
            if c["kind"] == "label":
                assert isinstance(c["options"], list) and len(c["options"]) >= 2
                assert all(isinstance(o, str) for o in c["options"])
            else:
                assert c["options"] is None

    def test_no_duplicate_keys(self):
        cols = sdd.scanner_criteria_columns()
        keys = [c["key"] for c in cols]
        assert len(keys) == len(set(keys))

    def test_every_key_besides_ticker_is_a_real_row_column(self):
        """Every criterion the panel can offer must correspond to a column
        scanner_board.row.compute_row()/store.py actually produce -- so the
        panel can never filter/sort on a column that doesn't exist."""
        cols = sdd.scanner_criteria_columns()
        for c in cols:
            if c["key"] == "ticker":
                continue
            assert c["key"] in store.ROW_VALUE_COLUMNS, c["key"]

    def test_ticker_is_a_text_column(self):
        cols = {c["key"]: c for c in sdd.scanner_criteria_columns()}
        assert cols["ticker"]["kind"] == "text"
        assert cols["ticker"]["options"] is None

    def test_label_options_match_the_enum_that_produces_them(self):
        cols = {c["key"]: c for c in sdd.scanner_criteria_columns()}
        assert cols["triple_screen"]["options"] == ["Stand aside", "Go long setup", "Go short setup"]
        assert cols["ma_slope"]["options"] == ["Rising", "Falling", "Flat"]

    def test_numeric_column_has_no_options(self):
        cols = {c["key"]: c for c in sdd.scanner_criteria_columns()}
        assert cols["rsi"]["kind"] == "number"
        assert cols["rsi"]["options"] is None

    def test_available_without_a_db(self, tmp_path, monkeypatch):
        """Static column metadata, unlike build_scanner_state() -- must
        stay callable even when the board's DB is missing/unavailable."""
        monkeypatch.setattr(sdd, "SCANNER_BOARD_DB_PATH", tmp_path / "does_not_exist.db")
        assert len(sdd.scanner_criteria_columns()) > 0


class TestReadLatest:
    def test_returns_empty_when_db_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sdd, "SCANNER_BOARD_DB_PATH", tmp_path / "does_not_exist.db")
        assert sdd._read_latest() == []

    def test_returns_only_the_latest_run_date(self, tmp_path, monkeypatch, conn):
        store.upsert_rows(conn, "2026-09-10", [_seed_row("OLD.TO")], updated_at="t0")
        store.upsert_rows(conn, "2026-09-15", [_seed_row("NEW.TO")], updated_at="t1")
        db_path = [p for p in tmp_path.iterdir() if p.suffix == ".db"][0]
        monkeypatch.setattr(sdd, "SCANNER_BOARD_DB_PATH", db_path)

        got = sdd._read_latest()
        assert {r["ticker"] for r in got} == {"NEW.TO"}


class TestBuildScannerState:
    def test_attaches_a_badge_per_label_column(self, tmp_path, monkeypatch, conn):
        row = _seed_row("AAA.TO", ma_slope="Rising", macd_cross="Bear",
                         adx_regime="Waking Up", impulse_daily="Blue", value_zone="Above")
        store.upsert_rows(conn, "2026-09-15", [row], updated_at="t0")
        db_path = [p for p in tmp_path.iterdir() if p.suffix == ".db"][0]
        monkeypatch.setattr(sdd, "SCANNER_BOARD_DB_PATH", db_path)

        state = sdd._build_scanner_state()
        badges = state["rows"][0]["badges"]
        assert badges["ma_slope"] == "badge-bullish"
        assert badges["macd_cross"] == "badge-bearish"
        assert badges["adx_regime"] == "badge-highlight"
        assert badges["impulse_daily"] == "badge-impulse-blue"
        assert "value_zone" not in badges  # deliberately unbadged, see module docstring

        glyphs = state["rows"][0]["glyphs"]
        assert glyphs["ma_slope"] == "▲"

    def test_run_date_surfaced_at_top_level(self, tmp_path, monkeypatch, conn):
        store.upsert_rows(conn, "2026-09-15", [_seed_row("AAA.TO")], updated_at="t0")
        db_path = [p for p in tmp_path.iterdir() if p.suffix == ".db"][0]
        monkeypatch.setattr(sdd, "SCANNER_BOARD_DB_PATH", db_path)

        state = sdd._build_scanner_state()
        assert state["run_date"] == "2026-09-15"

    def test_empty_db_gives_none_run_date_and_empty_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sdd, "SCANNER_BOARD_DB_PATH", tmp_path / "does_not_exist.db")
        state = sdd._build_scanner_state()
        assert state == {"rows": [], "run_date": None}


class TestTtlCache:
    def test_serves_stale_data_within_ttl(self, tmp_path, monkeypatch, conn):
        store.upsert_rows(conn, "2026-09-15", [_seed_row("AAA.TO")], updated_at="t0")
        db_path = [p for p in tmp_path.iterdir() if p.suffix == ".db"][0]
        monkeypatch.setattr(sdd, "SCANNER_BOARD_DB_PATH", db_path)
        monkeypatch.setattr(sdd, "DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS", 999)

        first = sdd.build_scanner_state()
        store.upsert_rows(conn, "2026-09-16", [_seed_row("BBB.TO")], updated_at="t1")
        second = sdd.build_scanner_state()

        assert first == second
        assert first["run_date"] == "2026-09-15"
