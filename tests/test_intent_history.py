"""
tests/test_intent_history.py
============================
Regression tests for the "lost intent history" bug (found 2026-10).

db.save_intents() starts every pipeline run with
``DELETE FROM intents WHERE intent_status = 'PENDING'``. So any intent a buy
runner loads but leaves PENDING is erased the same evening with no record it
ever existed. Live 2026-09-11: the core book was full (8/8), virtual_buy.py
returned early without touching its intents, and the macro sleeve's
read-only buy of one of them (PHX.TO) became untraceable.

The invariant locked here, for both virtual_buy.py and momentum_buy.py:
after a real (non-dry-run, market-open) run, every intent the runner loaded
ends EXECUTED or SKIPPED-with-a-reason; none is left PENDING. The two
deliberate exceptions are also locked: --dry-run writes nothing, and the
market-closed guard (which runs before intents are read) leaves the queue
for the real 9:45 run.

kangaroo_buy.py is intentionally NOT covered by this invariant: its
untriggered breakout intents are meant to stay PENDING across days and are
carried forward by kangaroo_pipeline.py.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import duckdb
import pytest

import db as db_module
from config import MAX_POSITIONS, MOMENTUM_MAX_POSITIONS
from db import init_db, insert_position, load_pending_intents, save_intents, set_cash
from momentum_buy import run_momentum_buy
from time_utils import TSX_TZ, set_backtest_clock
from virtual_buy import run_virtual_buy


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES / HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Both runners share db.py's functions, so each is driven against the same
# temp DB; only the runner, its module (for patching) and its slot cap differ.
RUNNERS = [
    pytest.param(run_virtual_buy, "virtual_buy", MAX_POSITIONS, id="virtual_buy"),
    pytest.param(run_momentum_buy, "momentum_buy", MOMENTUM_MAX_POSITIONS, id="momentum_buy"),
]


@pytest.fixture(autouse=True)
def db(tmp_path):
    path = tmp_path / "trading.db"
    init_db(path)
    yield path
    db_module.DB_PATH = tmp_path / "reset.db"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No emails, no yfinance sector lookups. Every ticker is its own sector,
    so MAX_POSITIONS_PER_SECTOR never interferes unless a test opts in."""
    monkeypatch.setattr("virtual_buy.send_transaction_email", lambda **_: None)
    monkeypatch.setattr("momentum_buy.send_transaction_email", lambda **_: None)
    monkeypatch.setattr("virtual_buy.get_sector", lambda ticker: ticker)


@pytest.fixture(autouse=True)
def market_open():
    """2026-05-14 is a Thursday, 11:00 ET, not a TSX holiday."""
    set_backtest_clock(datetime(2026, 5, 14, 11, 0, tzinfo=TSX_TZ))
    yield
    set_backtest_clock(None)


def _intent(ticker: str, priority: int = 1, **overrides) -> dict:
    base = {
        "ticker": ticker,
        "signal_date": "2026-05-14",
        "alert_state": "CONFIRMED",
        "priority": priority,
        "pattern": "PB-EMA21",
        "entry_price_planned": 42.50,
        "stop_price": 40.00,
        "target_price": 47.50,
        "rr": 2.0,
    }
    base.update(overrides)
    return base


def _all_intents() -> dict[str, tuple[str, str | None]]:
    """{ticker: (intent_status, intent_reason)} for every intent row."""
    conn = duckdb.connect(str(db_module.DB_PATH), read_only=True)
    try:
        rows = conn.execute(
            "SELECT ticker, intent_status, intent_reason FROM intents ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return {t: (s, r) for t, s, r in rows}


def _assert_nothing_left_pending() -> None:
    assert load_pending_intents().empty, "intent left PENDING would be deleted by the next save_intents()"
    for ticker, (status, reason) in _all_intents().items():
        assert status in ("EXECUTED", "SKIPPED"), (ticker, status)
        if status == "SKIPPED":
            assert reason, f"{ticker} SKIPPED with no reason"


def _fill_book(n: int) -> None:
    for i in range(n):
        insert_position(f"HELD{i}.TO", "2026-05-01", 10.0, 10)


def _run(runner, module, top_n=None, dry_run=False, price=42.50):
    with patch(f"{module}.fetch_latest_price", return_value=price):
        runner(top_n=top_n, dry_run=dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# Each early-exit / leftover path now records a SKIPPED reason
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("runner,module,cap", RUNNERS)
def test_portfolio_full_marks_intents_skipped(runner, module, cap):
    set_cash(100_000.0)
    _fill_book(cap)
    save_intents([_intent("PHX.TO", 1), _intent("BNS.TO", 2)])

    _run(runner, module)

    assert _all_intents() == {
        "PHX.TO": ("SKIPPED", "portfolio_full"),
        "BNS.TO": ("SKIPPED", "portfolio_full"),
    }


@pytest.mark.parametrize("runner,module,cap", RUNNERS)
def test_no_funds_marks_intents_skipped(runner, module, cap):
    set_cash(0.0)
    save_intents([_intent("RY.TO")])

    _run(runner, module)

    assert _all_intents() == {"RY.TO": ("SKIPPED", "no_funds")}


@pytest.mark.parametrize("runner,module,cap", RUNNERS)
def test_candidates_beyond_open_slots_marked_no_slot_left(runner, module, cap):
    set_cash(100_000.0)
    _fill_book(cap - 1)  # exactly one slot open
    save_intents([_intent("AAA.TO", 1), _intent("BBB.TO", 2), _intent("CCC.TO", 3)])

    _run(runner, module)

    assert _all_intents() == {
        "AAA.TO": ("EXECUTED", None),
        "BBB.TO": ("SKIPPED", "no_slot_left"),
        "CCC.TO": ("SKIPPED", "no_slot_left"),
    }


@pytest.mark.parametrize("runner,module,cap", RUNNERS)
def test_intents_beyond_top_n_marked_skipped(runner, module, cap):
    set_cash(10_000.0)
    save_intents([_intent("AAA.TO", 1), _intent("BBB.TO", 2), _intent("CCC.TO", 3)])

    _run(runner, module, top_n=1)

    assert _all_intents() == {
        "AAA.TO": ("EXECUTED", None),
        "BBB.TO": ("SKIPPED", "beyond_top_n"),
        "CCC.TO": ("SKIPPED", "beyond_top_n"),
    }


def test_sector_cap_then_no_slot_left_both_recorded(monkeypatch):
    """virtual_buy only: a sector-capped candidate and a slot-starved one each
    get their own reason, and the slot goes to the next eligible candidate."""
    sectors = {"HELD0.TO": "Fin", "HELD1.TO": "Fin", "BNK.TO": "Fin", "TEC.TO": "Tech", "NRG.TO": "Energy"}
    monkeypatch.setattr("virtual_buy.get_sector", lambda t: sectors.get(t, "Unknown"))
    set_cash(100_000.0)
    _fill_book(MAX_POSITIONS - 1)  # HELD0/HELD1 are Fin -> Fin at the cap; one slot open
    save_intents([_intent("BNK.TO", 1), _intent("TEC.TO", 2), _intent("NRG.TO", 3)])

    _run(run_virtual_buy, "virtual_buy")

    assert _all_intents() == {
        "BNK.TO": ("SKIPPED", "sector_cap_Fin"),
        "TEC.TO": ("EXECUTED", None),
        "NRG.TO": ("SKIPPED", "no_slot_left"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The invariant across a mixed queue that hits every per-intent skip path
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("runner,module,cap", RUNNERS)
@pytest.mark.parametrize("held", [0, 1, "full"])
@pytest.mark.parametrize("top_n", [None, 2])
def test_no_intent_ever_left_pending(runner, module, cap, held, top_n):
    set_cash(10_000.0)
    if held == "full":
        _fill_book(cap - 1)
    else:
        _fill_book(held)
    insert_position("OWN.TO", "2026-05-01", 10.0, 1)
    save_intents([
        _intent("GOOD.TO", 1),
        _intent("OLD.TO", 2, signal_date="2026-05-01"),           # stale_intent
        _intent("GOOD.TO", 3),                                    # duplicate_pending
        _intent("OWN.TO", 4),                                     # already_owned
        _intent("BADSTOP.TO", 5, stop_price=43.00),               # invalid_stop_data
        _intent("NEXT.TO", 6),
        _intent("LAST.TO", 7),
    ])

    _run(runner, module, top_n=top_n)

    _assert_nothing_left_pending()


@pytest.mark.parametrize("runner,module,cap", RUNNERS)
def test_no_price_data_still_recorded(runner, module, cap):
    set_cash(10_000.0)
    save_intents([_intent("NOPX.TO")])

    _run(runner, module, price=None)

    assert _all_intents() == {"NOPX.TO": ("SKIPPED", "no_price_data")}


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end regression: the exact 2026-09-11 sequence
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("runner,module,cap", RUNNERS)
def test_intent_survives_next_pipeline_run_when_book_full(runner, module, cap):
    """Full book at 9:45 -> next evening's pipeline writes a fresh batch.
    Before the fix the first intent vanished entirely; it must now remain
    as history."""
    set_cash(100_000.0)
    _fill_book(cap)
    save_intents([_intent("PHX.TO", stop_price=11.94, entry_price_planned=12.41)])

    _run(runner, module)
    save_intents([_intent("NEW.TO")])  # the 16:30 pipeline run

    intents = _all_intents()
    assert intents["PHX.TO"] == ("SKIPPED", "portfolio_full")
    assert intents["NEW.TO"] == ("PENDING", None)


# ─────────────────────────────────────────────────────────────────────────────
# Deliberate exceptions: these must NOT consume the queue
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("runner,module,cap", RUNNERS)
@pytest.mark.parametrize("scenario", ["full", "no_funds", "top_n", "slots"])
def test_dry_run_leaves_every_intent_pending(runner, module, cap, scenario):
    set_cash(0.0 if scenario == "no_funds" else 100_000.0)
    _fill_book({"full": cap, "slots": cap - 1}.get(scenario, 0))
    save_intents([_intent("AAA.TO", 1), _intent("BBB.TO", 2), _intent("CCC.TO", 3)])

    _run(runner, module, top_n=1 if scenario == "top_n" else None, dry_run=True)

    assert {s for s, _ in _all_intents().values()} == {"PENDING"}


@pytest.mark.parametrize("runner,module,cap", RUNNERS)
def test_market_closed_leaves_queue_for_the_real_run(runner, module, cap):
    """A manual off-hours run must not burn the intents the 9:45 run needs."""
    set_backtest_clock(datetime(2026, 5, 14, 8, 0, tzinfo=TSX_TZ))
    set_cash(100_000.0)
    _fill_book(cap)
    save_intents([_intent("AAA.TO")])

    _run(runner, module)

    assert _all_intents() == {"AAA.TO": ("PENDING", None)}
