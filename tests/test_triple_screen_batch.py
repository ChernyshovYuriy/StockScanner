"""
Batch layer test: a list of tickers, one FakeDataProvider stubbed with known
per-ticker series, asserting the per-ticker Signal map -- end-to-end through
the real reference_impl screens/engine (not fakes), so this also doubles as
a real-world sanity check that a fully-formed BUY setup, an unconditional
trend-flip SELL, and an indecisive WAIT all resolve correctly when wired
together, not just in the engine's isolated truth-table test.

Planned Phase 3 batch API (research.triple_screen.batch):
    run_batch(tickers: Sequence[str], provider: DataProvider,
              engine: SignalEngine, positions: Mapping[str, bool] | None = None,
              timeframes: TimeframeConfig = TimeframeConfig()) -> dict[str, SignalResult]
`positions` defaults to "flat" (False) for any ticker not listed in it.
"""
from research.triple_screen.batch import format_results_table, load_tickers, run_batch
from research.triple_screen.reference_impl import (
    EMASlopeTrendScreen, ForceIndexEntryScreen, PriorBarTriggerScreen, TripleScreenEngine,
)
from research.triple_screen.types import Signal

from triple_screen_fixtures import (
    FakeDataProvider, aligned, buy_ready_entry_bars, downtrend_bars, flat_bars, uptrend_bars,
)


def _buy_ready_entry_bars():
    """The one daily bar shape that satisfies Screen 2's pullback AND
    Screen 3's true-breakout confirmation (2026-09: price cross alone is no
    longer enough) simultaneously -- see
    triple_screen_fixtures.buy_ready_entry_bars, numerically verified since
    the two screens are structurally in tension on the shared latest bar.
    """
    return buy_ready_entry_bars(trend="UP", ticker="BUY_READY")


def _engine() -> TripleScreenEngine:
    return TripleScreenEngine(
        trend_screen=EMASlopeTrendScreen(),
        entry_screen=ForceIndexEntryScreen(),
        trigger_screen=PriorBarTriggerScreen(),
    )


def test_run_batch_returns_correct_signal_per_ticker():
    provider = FakeDataProvider({
        "BUY_CANDIDATE": aligned(trend=uptrend_bars(timeframe="weekly"), entry=_buy_ready_entry_bars()),
        "SELL_CANDIDATE": aligned(trend=downtrend_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily")),
        "WAIT_CANDIDATE": aligned(trend=flat_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily")),
    })
    results = run_batch(
        tickers=["BUY_CANDIDATE", "SELL_CANDIDATE", "WAIT_CANDIDATE"],
        provider=provider,
        engine=_engine(),
        positions={"SELL_CANDIDATE": True},   # only this one has an open position
    )

    assert set(results.keys()) == {"BUY_CANDIDATE", "SELL_CANDIDATE", "WAIT_CANDIDATE"}
    assert results["BUY_CANDIDATE"].signal == Signal.BUY
    assert results["SELL_CANDIDATE"].signal == Signal.SELL   # DOWN trend + open position -> unconditional exit
    assert results["WAIT_CANDIDATE"].signal == Signal.WAIT


def test_run_batch_result_is_auditable_per_ticker():
    """Every ticker's result carries its own three screen verdicts, not just
    the final signal -- required by the "explainable, not just trusted"
    constraint, at the batch layer too."""
    provider = FakeDataProvider({
        "T": aligned(trend=uptrend_bars(timeframe="weekly"), entry=_buy_ready_entry_bars()),
    })
    results = run_batch(tickers=["T"], provider=provider, engine=_engine())
    result = results["T"]
    assert result.trend is not None
    assert result.entry is not None
    assert result.trigger is not None


def test_run_batch_defaults_unlisted_tickers_to_flat_position():
    """No `positions` entry for a ticker -> treated as flat, matching the
    documented default -- proven here via a DOWN-trend ticker with no
    position entry at all: unconditional-SELL-on-open never fires, so it
    must resolve WAIT (long-only, nothing to exit), not SELL."""
    provider = FakeDataProvider({
        "NO_POSITION_INFO": aligned(trend=downtrend_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily")),
    })
    results = run_batch(tickers=["NO_POSITION_INFO"], provider=provider, engine=_engine())
    assert results["NO_POSITION_INFO"].signal == Signal.WAIT


# ── CLI table formatting ──────────────────────────────────────────────────────

def test_format_results_table_includes_ticker_and_signal():
    provider = FakeDataProvider({
        "T": aligned(trend=uptrend_bars(timeframe="weekly"), entry=_buy_ready_entry_bars()),
    })
    results = run_batch(tickers=["T"], provider=provider, engine=_engine())
    table = format_results_table(results)
    assert "T" in table
    assert "BUY" in table


def test_format_results_table_groups_by_signal_buy_sell_hold_wait():
    """Rows are ordered BUY, SELL, HOLD, WAIT regardless of input order --
    fed in the opposite order here (WAIT_T, HOLD_T, SELL_T, BUY_T) to prove
    the table itself re-sorts rather than just preserving input order."""
    provider = FakeDataProvider({
        "WAIT_T": aligned(trend=flat_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily")),
        "HOLD_T": aligned(trend=uptrend_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily")),
        "SELL_T": aligned(trend=downtrend_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily")),
        "BUY_T": aligned(trend=uptrend_bars(timeframe="weekly"), entry=_buy_ready_entry_bars()),
    })
    results = run_batch(
        tickers=["WAIT_T", "HOLD_T", "SELL_T", "BUY_T"],
        provider=provider,
        engine=_engine(),
        positions={"HOLD_T": True, "SELL_T": True},
    )
    # sanity: confirm each fixture actually produced the signal this test relies on
    assert results["BUY_T"].signal == Signal.BUY
    assert results["SELL_T"].signal == Signal.SELL
    assert results["HOLD_T"].signal == Signal.HOLD
    assert results["WAIT_T"].signal == Signal.WAIT

    table = format_results_table(results)
    order = [t for t in ["BUY_T", "SELL_T", "HOLD_T", "WAIT_T"] if t in table]
    positions_in_table = [table.index(t) for t in order]
    assert positions_in_table == sorted(positions_in_table), (
        f"expected BUY, SELL, HOLD, WAIT order in:\n{table}"
    )


# ── load_tickers (CAN_TICKERS_URL source) ────────────────────────────────────

class _FakeResponse:
    def __init__(self, text: str):
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._text.encode("utf-8")


def test_load_tickers_from_url_skips_blanks_and_comments(monkeypatch):
    import urllib.request
    text = "AAPL\n# a comment\n\nMSFT\n  SLF.TO  \n"
    monkeypatch.setattr(urllib.request, "urlopen", lambda url: _FakeResponse(text))
    assert load_tickers("https://example.com/tickers.txt") == ["AAPL", "MSFT", "SLF.TO"]


def test_load_tickers_from_local_file(tmp_path):
    f = tmp_path / "tickers.txt"
    f.write_text("RY.TO\nTD.TO\n# skip me\n")
    assert load_tickers(str(f)) == ["RY.TO", "TD.TO"]
