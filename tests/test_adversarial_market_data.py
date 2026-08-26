"""
tests/test_adversarial_market_data.py
======================================
Adversarial test suite for the pure-logic methods that consume this
codebase's custom market-data structures. Every method here is assumed
BROKEN until it survives boundary, semantic, numerical, and property-based
attacks — a passing ground-truth test is the minimum bar, not the proof.

Scope
-----
Per instructions: data ingestion (yfinance) is out of scope. Every fixture
below constructs the data structure directly — no network call, no mock of
yfinance, anywhere in this file.

Covering nine methods across five modules, chosen because they are the
methods that actually decide entries, stops, sizing, and mark-to-market
value from the structures below (i.e. where a silent wrong answer moves
real — paper — money):

  market_data.validate_ohlcv
  market_data.HistoricalSliceProvider.get / .download
  position_monitor.wilder_atr
  position_monitor.compute_signals
  auto_pipeline.compute_levels
  auto_pipeline.compute_position_size
  portfolio.PortfolioState.buy / .sell
  backtest_runner._mark_to_market / ._day_close_price

Not every public method in the codebase is covered — an exhaustive suite
across all ~9 modules previously audited would not fit one file. These nine
are the highest-financial-risk, structure-consuming methods; the rest were
covered narratively in the preceding audit passes (see session).

─────────────────────────────────────────────────────────────────────────────
THE CUSTOM MARKET-DATA STRUCTURE — CONTRACT INFERRED FROM THE CODE
─────────────────────────────────────────────────────────────────────────────

1. OHLCV DataFrame (market_data.py's owned contract, not "a DataFrame" by
   accident — validate_ohlcv() is the acceptance test):
     - a pandas.DataFrame
     - index: pandas.DatetimeIndex (validate_ohlcv raises ValueError if not)
     - columns: exactly ["Open", "High", "Low", "Close", "Volume"] (order
       enforced on output; extra columns are silently dropped, not rejected)
     - dtype: every column cast to float64 (Volume included — a share count
       is stored as a float, not an int)
     - units: Open/High/Low/Close in the ticker's trading currency (CAD for
       .TO/.V/.CN symbols); Volume in shares/session
     - an EMPTY DataFrame is a valid, explicitly-supported instance and is
       returned unchanged (identity, not a copy) — "no data" is a state
       many callers branch on, not an error
     - validate_ohlcv is a STRUCTURAL contract only: it does not reject
       NaN-filled columns, zero prices, or negative prices/volume. Any value
       validation (price > 0, no NaN close, etc.) is each caller's own job.
     - NOT guaranteed by construction: sorted index, unique index. Some
       constructors (HistoricalSliceProvider.__init__) sort but do not
       deduplicate; validate_ohlcv itself does neither.

2. TodayBar (market_data.py) — live intraday snapshot, dataclass:
     - low: float   (session low so far)
     - close: float (latest traded price)
     - high: float  (session high so far)
     - source: str  (e.g. "5m-intraday")
   No invariant is enforced between low <= close <= high at construction —
   callers must not assume it holds.

3. Position (position_monitor.py) — one open position, dataclass:
     - ticker: str
     - entry_date: datetime.date
     - entry_price: float (CAD, > 0 expected but NOT enforced at
       construction — compute_signals is the one place that guards it)
     - shares: float
     - stop_price: Optional[float] — planned exit stop carried from the
       buy intent; None means "recompute from ATR"

ASSUMPTIONS these tests bake in (state explicitly per instructions):
  A1. "Look-ahead bias" for compute_signals means: a bar dated strictly
      before pos.entry_date must not influence hh_since_entry / peak_price /
      max_pnl_pct. Bars dated on/after entry_date, INCLUDING the very last
      bar of df ("today"), are legitimately known and must be used — this is
      not look-ahead, it's "after close."
  A2. ATR (both wilder_atr and auto_pipeline._atr) is a whole-history
      indicator by design — a pre-entry volatility spike legitimately
      inflates atr_latest. That is not a look-ahead violation; it's tested
      separately from the hh_since_entry guard (A1) precisely so the two
      don't get conflated.
  A3. "Ground truth" prices are engineered so True Range is an exact,
      hand-computable constant every bar (see _build_flat_range_series
      below) — this isolates the exit-rule branching logic from ATR's own
      correctness, which is separately hand-verified in TestWilderAtr.
"""

from __future__ import annotations

import smtplib
from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from auto_pipeline import compute_levels, compute_position_size
from backtest_runner import _day_close_price, _mark_to_market
from market_data import HistoricalSliceProvider, OHLCV_COLUMNS, validate_ohlcv
from portfolio import OpenPosition, PortfolioState
from position_monitor import ExitParams, Position, compute_signals, wilder_atr


# ─────────────────────────────────────────────────────────────────────────────
# HARD EMAIL SAFETY NET — no test in this file may ever dispatch a real email.
# Two layers: (1) neuter every send_report entry point, (2) neuter smtplib
# itself so even a future test added to this file that forgets to mock at the
# send_report layer still cannot open a real SMTP connection.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_email_ever(monkeypatch):
    import send_report
    monkeypatch.setattr(send_report, "send_transaction_email", lambda *a, **k: None)
    monkeypatch.setattr(send_report, "send_report", lambda *a, **k: None)
    monkeypatch.setattr(send_report, "send_text_email", lambda *a, **k: False)

    def _no_smtp(*args, **kwargs):
        raise AssertionError(
            "smtplib.SMTP() was called during a test run — email must never "
            "be dispatched under tests. Mock the caller, don't reach here."
        )
    monkeypatch.setattr(smtplib, "SMTP", _no_smtp)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _ohlcv(index, opens, highs, lows, closes, volumes) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=index,
    )


def _flat_range_series(closes: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    """
    OHLCV where High = Close+1, Low = Close-1 and |Close[i]-Close[i-1]| <= 1
    for every bar. Under those constraints True Range is EXACTLY 2.0 on every
    single bar (proof: TR = max(H-L, |H-prevC|, |L-prevC|); H-L=2 always; for
    a +1 move, H-prevC=2 and L-prevC=0; for a -1 move, H-prevC=0 and
    L-prevC=2; for a 0 move, both are 1 — max is always 2). Since Wilder's
    EMA of a constant series equals that constant, wilder_atr(df, 14) is
    exactly 2.0 on every bar, letting exit-rule tests hand-compute stop
    levels without separately re-deriving ATR each time.
    """
    idx = pd.bdate_range(start, periods=len(closes))
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    return _ohlcv(idx, closes, highs, lows, closes, [500_000] * len(closes)), idx


# ─────────────────────────────────────────────────────────────────────────────
# 0. SELF-CHECK — proves the email guard above actually intercepts, rather
#    than trusting the monkeypatch silently. Not one of the nine audited
#    methods; a safety-net test for this file's own no_email_ever fixture.
# ─────────────────────────────────────────────────────────────────────────────

class TestEmailSafetyNetItself:
    def test_smtp_construction_is_blocked_during_tests(self):
        with pytest.raises(AssertionError, match="must never be dispatched"):
            smtplib.SMTP("smtp.gmail.com", 587)

    def test_send_report_entry_points_are_neutered(self):
        import send_report
        assert send_report.send_transaction_email(buys=[], sells=[], cash_before=0,
                                                    cash_after=0, open_positions_count=0) is None
        assert send_report.send_text_email("subject", "body") is False


# ─────────────────────────────────────────────────────────────────────────────
# 1. market_data.validate_ohlcv
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateOhlcv:
    """Hypothesis under test: the contract enforcer silently lets something
    through it shouldn't, or silently mangles well-formed data."""

    def test_ground_truth_column_order_and_dtype(self):
        """Columns must be reordered/renamed to exactly OHLCV_COLUMNS and
        cast to float64, dropping any extra column, regardless of input
        column order."""
        idx = pd.date_range("2024-01-01", periods=1)
        df = pd.DataFrame(
            {"Volume": [100], "Close": [10], "Open": [9], "Extra": ["x"],
             "High": [11], "Low": [8]},
            index=idx,
        )
        out = validate_ohlcv(df)
        assert list(out.columns) == OHLCV_COLUMNS
        assert out["Open"].iloc[0] == 9.0 and isinstance(out["Open"].iloc[0], float)
        assert out["Volume"].dtype == np.float64
        assert "Extra" not in out.columns

    def test_empty_dataframe_returned_unchanged_by_identity(self):
        """Hypothesis: 'empty passes through' might mean an equal-but-copied
        frame, not the literal same object — worth pinning since callers may
        rely on identity for a cheap 'was this touched' check."""
        empty = pd.DataFrame()
        out = validate_ohlcv(empty)
        assert out is empty

    def test_missing_column_raises_valueerror_naming_the_column(self):
        """Hypothesis: a missing column is silently filled with NaN instead
        of raising."""
        idx = pd.date_range("2024-01-01", periods=1)
        df = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0]}, index=idx)
        with pytest.raises(ValueError, match=r"missing columns.*Volume"):
            validate_ohlcv(df)

    def test_non_datetimeindex_raises_valueerror(self):
        """Hypothesis: a plain RangeIndex silently slips through, and every
        downstream as_of-cutoff comparison then breaks silently."""
        df = pd.DataFrame({c: [1.0] for c in OHLCV_COLUMNS}, index=[0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_ohlcv(df)

    def test_nan_filled_column_is_not_rejected(self):
        """Documents the contract explicitly: this is a STRUCTURAL validator
        only. A fully-NaN Close column must pass through unchanged, not
        raise — callers (e.g. compute_signals) are responsible for their own
        NaN handling."""
        idx = pd.date_range("2024-01-01", periods=2)
        df = pd.DataFrame({c: [np.nan, np.nan] for c in OHLCV_COLUMNS}, index=idx)
        out = validate_ohlcv(df)
        assert out["Close"].isna().all()

    def test_negative_and_zero_prices_are_not_rejected(self):
        """Documents the contract explicitly: no value validation. A
        negative Close must survive unchanged — this is intentional per the
        docstring, not an oversight, but every caller of validate_ohlcv must
        assume it can receive one."""
        idx = pd.date_range("2024-01-01", periods=2)
        df = pd.DataFrame({c: [-5.0, 0.0] for c in OHLCV_COLUMNS}, index=idx)
        out = validate_ohlcv(df)
        assert out["Close"].tolist() == [-5.0, 0.0]

    @given(
        n=st.integers(min_value=1, max_value=30),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_length_and_column_invariants_hold(self, n, seed):
        """Property: for ANY well-shaped random float frame with a
        DatetimeIndex, output always has the same row count and exactly
        OHLCV_COLUMNS in order — regardless of the random values inside."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n)
        df = pd.DataFrame(
            {c: rng.standard_normal(n) * 100 for c in OHLCV_COLUMNS}, index=idx
        )
        out = validate_ohlcv(df)
        assert len(out) == n
        assert list(out.columns) == OHLCV_COLUMNS

    def test_property_idempotent(self):
        """Property: validate_ohlcv(validate_ohlcv(df)) == validate_ohlcv(df)."""
        idx = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({c: [1.0, 2, 3, 4, 5] for c in OHLCV_COLUMNS}, index=idx)
        once = validate_ohlcv(df)
        twice = validate_ohlcv(once)
        pd.testing.assert_frame_equal(once, twice)


# ─────────────────────────────────────────────────────────────────────────────
# 2. market_data.HistoricalSliceProvider.get — the look-ahead guard itself
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoricalSliceProviderGet:
    """Hypothesis under test: the as_of cutoff leaks a future bar, or an
    unsorted/duplicate constructor input silently corrupts the slice."""

    def test_ground_truth_cutoff_returns_exactly_bars_on_or_before(self):
        df, idx = _flat_range_series([100, 101, 102, 103, 104])
        p = HistoricalSliceProvider({"T": df})
        out = p.get("T", as_of=idx[2])
        assert list(out.index) == list(idx[:3])
        assert out["Close"].tolist() == [100, 101, 102]

    def test_lookahead_poison_value_after_cutoff_never_leaks(self):
        """Hypothesis: a poisoned/extreme value one bar past as_of leaks into
        the returned slice's last row."""
        closes = [100, 101, 102, 103, 999_999]  # index 4 is the poison bar
        df, idx = _flat_range_series(closes)
        p = HistoricalSliceProvider({"T": df})
        out = p.get("T", as_of=idx[3])
        assert out["Close"].iloc[-1] == 103.0
        assert 999_999 not in out["Close"].values

    def test_boundary_as_of_before_first_bar_returns_empty(self):
        df, idx = _flat_range_series([100, 101, 102])
        p = HistoricalSliceProvider({"T": df})
        out = p.get("T", as_of=idx[0] - pd.Timedelta(days=1))
        assert out.empty

    def test_unknown_ticker_raises_keyerror(self):
        df, idx = _flat_range_series([100])
        p = HistoricalSliceProvider({"T": df})
        with pytest.raises(KeyError):
            p.get("NOPE", as_of=idx[0])

    def test_unsorted_constructor_input_is_sorted(self):
        """Ground truth: constructor input given out of order must come back
        sorted, and value-to-date mapping must be preserved through the sort
        (not just the dates reordered while values stay in original slots)."""
        idx_unsorted = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        df = _ohlcv(idx_unsorted, [30, 10, 20], [31, 11, 21], [29, 9, 19], [30, 10, 20], [1, 1, 1])
        p = HistoricalSliceProvider({"T": df})
        out = p.get("T", as_of=pd.Timestamp("2024-01-03"))
        assert list(out.index) == sorted(idx_unsorted)
        assert out["Close"].tolist() == [10.0, 20.0, 30.0]  # value follows its own date

    def test_duplicate_timestamps_are_deduplicated_last_wins(self):
        """FIXED (was xfail): duplicate index timestamps are now deduplicated
        at construction, keeping the last row — matching the convention
        position_monitor.load_or_fetch_data already used for its own
        cache-merge path (`d[~d.index.duplicated(keep='last')]`)."""
        idx_dup = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
        df = _ohlcv(idx_dup, [1, 2, 99, 3], [2, 3, 100, 4], [0, 1, 98, 2], [1, 2, 99, 3], [1, 1, 1, 1])
        p = HistoricalSliceProvider({"T": df})
        out = p.get("T", as_of=pd.Timestamp("2024-01-02"))
        assert len(out) == 2
        assert out["Close"].iloc[-1] == 99.0  # last-wins semantics

    def test_tz_naive_construction_does_not_raise(self):
        """Boundary: a tz-naive index at construction must not hit
        `.tz_localize(None)` on an already-naive index and raise."""
        df, idx = _flat_range_series([100, 101])
        p = HistoricalSliceProvider({"T": df})
        assert p._data["T"].index.tz is None

    @given(
        as_of_offset=st.integers(min_value=-5, max_value=15),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_no_returned_bar_is_ever_after_as_of(self, as_of_offset):
        """The core look-ahead invariant, fuzzed: for ANY as_of cutoff
        (even before the first bar or after the last), every returned row's
        index must be <= as_of. This is the property the whole backtest's
        lookahead-bias guarantee rests on."""
        df, idx = _flat_range_series(list(range(100, 110)))
        p = HistoricalSliceProvider({"T": df})
        as_of = idx[0] + pd.Timedelta(days=as_of_offset)
        out = p.get("T", as_of=as_of)
        assert (out.index <= as_of).all()


class TestHistoricalSliceProviderDownload:
    """Hypothesis under test: the >200-bar quality gate is off-by-one, or the
    earliest-cutoff window arithmetic silently excludes/includes the wrong
    bars."""

    def _long_provider(self, n=1000, start="2020-01-01"):
        idx = pd.date_range(start, periods=n, freq="D")  # calendar-daily: exact day-count math
        closes = list(range(n))
        df = _ohlcv(idx, closes, [c + 1 for c in closes], [c - 1 for c in closes], closes, [1000] * n)
        return HistoricalSliceProvider({"T": df}), idx

    def test_ground_truth_quality_gate_boundary_200_excluded_201_included(self):
        """window row-count = days + 60 + 1 (inclusive endpoints, daily bars).
        days=139 -> 200 rows -> must be EXCLUDED (len(sliced) > 200 is False).
        days=140 -> 201 rows -> must be INCLUDED."""
        p, idx = self._long_provider()
        as_of = idx[900]
        res_200 = p.download(["T"], days=139, as_of=as_of)
        res_201 = p.download(["T"], days=140, as_of=as_of)
        assert "T" not in res_200
        assert "T" in res_201
        assert len(res_201["T"]) == 201

    def test_missing_ticker_silently_skipped_not_keyerror(self):
        p, idx = self._long_provider()
        res = p.download(["NOPE"], days=140, as_of=idx[900])
        assert res == {}

    def test_property_every_included_ticker_respects_cutoff(self):
        p, idx = self._long_provider()
        as_of = idx[900]
        res = p.download(["T"], days=300, as_of=as_of)
        assert (res["T"].index <= as_of).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. position_monitor.wilder_atr
# ─────────────────────────────────────────────────────────────────────────────

class TestWilderAtr:
    """Hypothesis under test: the smoothing constant is wrong (span vs
    alpha=1/period — the exact historical bug this module's own comment
    documents having been fixed once already elsewhere)."""

    def test_ground_truth_matches_hand_computed_wilder_ewm(self):
        df = _ohlcv(
            pd.date_range("2024-01-01", periods=5),
            opens=[10, 11, 12, 11, 13], highs=[10, 11, 12, 11, 13],
            lows=[9, 10, 10, 9, 11], closes=[9.5, 10.5, 11.5, 10, 12],
            volumes=[1] * 5,
        )
        out = wilder_atr(df, period=3)
        # Hand computation (see audit): TR = [1, 1.5, 2, 2.5, 3],
        # EMA(alpha=1/3): 1, 1.166667, 1.444444, 1.796296, 2.197531
        expected = [1.0, 1.166667, 1.444444, 1.796296, 2.197531]
        assert out.round(6).tolist() == expected

    def test_flat_range_series_atr_is_exactly_two(self):
        df, _ = _flat_range_series([100.0] * 20)
        out = wilder_atr(df, period=14)
        assert (out.dropna() == 2.0).all()

    def test_boundary_single_bar_atr_equals_that_bars_high_low(self):
        df = _ohlcv(pd.date_range("2024-01-01", periods=1), [10], [11], [9], [10], [1])
        out = wilder_atr(df, period=14)
        assert out.iloc[0] == 2.0  # TR[0] = H-L (prev close NaN, skipped by max)

    def test_zero_volatility_flat_price_does_not_divide_by_zero(self):
        """Boundary: High==Low==Close for every bar (a halted, unmoving
        print). No ZeroDivisionError, and ATR is exactly 0.0, not NaN."""
        idx = pd.date_range("2024-01-01", periods=20)
        df = _ohlcv(idx, [50.0] * 20, [50.0] * 20, [50.0] * 20, [50.0] * 20, [0] * 20)
        out = wilder_atr(df, period=14)
        assert (out.dropna() == 0.0).all()

    @given(
        n=st.integers(min_value=2, max_value=60),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_atr_never_negative(self, n, seed):
        """Property: ATR is an average of non-negative True Range values —
        it can never be negative for any valid (High >= Low) input."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n)
        low = rng.uniform(1, 100, n)
        high = low + rng.uniform(0, 20, n)  # enforce High >= Low
        close = low + rng.uniform(0, (high - low))
        df = _ohlcv(idx, close, high, low, close, [1000] * n)
        out = wilder_atr(df, period=14)
        assert (out.dropna() >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. position_monitor.compute_signals
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSignals:
    """Hypothesis under test: a stop/giveback/time-stop threshold is
    off-by-one, a pre-entry bar leaks into the position's own high-water
    mark, or entry_price<=0 crashes the whole per-position loop in main()."""

    # Explicit trajectory used across several tests: flat 100 for 6 bars
    # (indices 0-5), entry at index 5 (entry_price=100.0), then +1/day to a
    # peak of 115 at index 20, then -1/day back down. True Range is exactly
    # 2.0 on every bar (see _flat_range_series), so ATR(14)=2.0 always and
    # initial_stop = 100 - 2.0*2.0 = 96.0 whenever no planned_stop is given.
    TRAJECTORY = (
        [100, 100, 100, 100, 100, 100] +           # 0-5 (entry at index 5)
        list(range(101, 116)) +                    # 6-20 climb to peak 115
        list(range(114, 108, -1))                   # 21-26 decline 114..109
    )

    def _pos_and_df(self, entry_price=100.0, shares=10.0, stop_price=None, n=None):
        closes = self.TRAJECTORY if n is None else self.TRAJECTORY[:n]
        df, idx = _flat_range_series(closes)
        pos = Position(ticker="T", entry_date=idx[5].date(), entry_price=entry_price,
                        shares=shares, stop_price=stop_price)
        return pos, df

    def test_ground_truth_hold_at_peak(self):
        """last_close=115 (index 20): chandelier armed (15% >= 8% arm),
        stop=max(96, 115+1-2.5*2)=max(96,111)=111. STOP_HIT: 115<111? no.
        GIVEBACK: pnl=15.0 <= 15.0-3.0=12.0? no. => HOLD."""
        pos, df = self._pos_and_df(n=21)
        r = compute_signals(pos, df)
        assert r["status"] == "HOLD"
        assert r["stop_price"] == 111.0
        assert r["initial_stop"] == 96.0
        assert r["max_pnl_%"] == 15.0
        assert r["pnl_%"] == 15.0
        assert r["R_mult"] == 3.75  # (115-100)/(100-96)

    def test_ground_truth_giveback_fires_at_close_111(self):
        """last_close=111 (index 24): pnl=11.0 <= 15.0-3.0=12.0 -> GIVEBACK.
        STOP_HIT: 111<111 is False (strict <) -> stop alone does not fire."""
        pos, df = self._pos_and_df(n=25)
        r = compute_signals(pos, df)
        assert r["status"] == "SELL"
        assert "GIVEBACK" in r["reason"]
        assert "STOP_HIT" not in r["reason"]

    def test_ground_truth_stop_hit_and_giveback_both_fire_at_close_109(self):
        """last_close=109 (index 26): 109 < stop(111) -> STOP_HIT.
        pnl=9.0 <= 12.0 -> GIVEBACK too. Both reasons must be present."""
        pos, df = self._pos_and_df(n=27)
        r = compute_signals(pos, df)
        assert r["status"] == "SELL"
        assert "STOP_HIT(close 109.00 < stop 111.00)" in r["reason"]
        assert "GIVEBACK" in r["reason"]

    def test_ground_truth_stop_hit_alone_before_chandelier_arms(self):
        """A shallow +5% run (never reaches the 8% arm threshold, so
        chandelier/giveback stay fully inactive), then declines through the
        plain ATR-based initial_stop=96.0. Only STOP_HIT should fire."""
        closes = [100] * 6 + [101, 102, 103, 104, 105] + list(range(104, 94, -1))
        df, idx = _flat_range_series(closes)
        pos = Position(ticker="T", entry_date=idx[5].date(), entry_price=100.0, shares=10)
        r = compute_signals(pos, df)
        assert r["last_close"] == 95.0
        assert r["max_pnl_%"] == 5.0  # never armed giveback (needs >=6%) or chandelier (>=8%)
        assert r["status"] == "SELL"
        assert r["reason"] == "STOP_HIT(close 95.00 < stop 96.00)"

    def test_ground_truth_time_stop_fires_at_exactly_20_trading_days_unprofitable(self):
        """20 flat bars (tdays==20, boundary is >=) then a $1 decline making
        pnl=-1.0% (< 0.0% threshold). Chandelier/giveback never arm (max
        profit stays 0%). Only TIME_STOP should fire."""
        closes = [100] * 24 + [99]
        df, idx = _flat_range_series(closes)
        pos = Position(ticker="T", entry_date=idx[5].date(), entry_price=100.0, shares=10)
        r = compute_signals(pos, df)
        assert r["tdays"] == 20
        assert r["status"] == "SELL"
        assert r["reason"] == "TIME_STOP(20d, pnl -1.0%)"

    def test_ground_truth_time_stop_does_not_fire_at_19_days(self):
        """One bar short of the 20-day threshold: must still HOLD even
        though pnl is negative — pins the boundary from the other side."""
        closes = [100] * 23 + [99]
        df, idx = _flat_range_series(closes)
        pos = Position(ticker="T", entry_date=idx[5].date(), entry_price=100.0, shares=10)
        r = compute_signals(pos, df)
        assert r["tdays"] == 19
        assert r["status"] == "HOLD"

    def test_entry_price_zero_returns_bad_data_not_a_crash(self):
        """Documented guard: entry_price<=0 must short-circuit before any
        division, since main() has no per-position try/except."""
        pos, df = self._pos_and_df(entry_price=0.0, n=10)
        r = compute_signals(pos, df)
        assert r == {"ticker": "T", "status": "BAD_DATA", "reason": "Invalid entry_price (0.0)"}

    def test_entry_price_negative_also_returns_bad_data(self):
        pos, df = self._pos_and_df(entry_price=-10.0, n=10)
        r = compute_signals(pos, df)
        assert r["status"] == "BAD_DATA"

    def test_empty_dataframe_returns_no_data_not_a_crash(self):
        pos = Position(ticker="T", entry_date=date(2024, 1, 1), entry_price=100.0, shares=10)
        r = compute_signals(pos, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
        assert r["status"] == "NO_DATA"

    def test_lookahead_pre_entry_spike_does_not_inflate_hh_since_entry(self):
        """The core look-ahead-bias test (Assumption A1): a bar dated
        BEFORE entry with an enormous High must not count toward the
        position's own high-water mark. Isolated from ATR contamination by
        zeroing chand_trail_atr_k, so chandelier_stop == hh_since_entry
        exactly (see module docstring A2/A3)."""
        closes = [100, 100, 100, 100, 100, 100, 101, 102, 103]
        idx = pd.bdate_range("2024-01-02", periods=len(closes))
        highs = [c + 1.0 for c in closes]
        highs[2] = 9999.0  # 3 bars before entry (index 5) — must never leak in
        df = _ohlcv(idx, closes, highs, [c - 1.0 for c in closes], closes, [500_000] * len(closes))
        pos = Position(ticker="T", entry_date=idx[5].date(), entry_price=100.0, shares=10)
        ep = ExitParams(chand_trail_atr_k=0.0, chand_arm_pct=0.0)
        r = compute_signals(pos, df, exit_params=ep)
        assert r["chandelier_stop"] == 104.0  # max(post-entry highs) = 103+1
        assert r["chandelier_stop"] != 9999.0

    def test_planned_stop_overrides_atr_derived_initial_stop(self):
        pos, df = self._pos_and_df(stop_price=90.0, n=10)
        r = compute_signals(pos, df, planned_stop=90.0)
        assert r["initial_stop"] == 90.0  # not the ATR-derived 96.0

    def test_giveback_exact_boundary_float_precision(self):
        """FIXED (was xfail): the GIVEBACK rule fires at exactly 'peak -
        giveback_allow_pct', inclusive, for a peak of 115.0 and current price
        112.0 (pnl=12.0, threshold=12.0). This used to silently miss the
        boundary — 112.0/100.0*100-100 == 12.00000000000001 vs
        115.0/100.0*100-100 - 3.0 == 11.999999999999991, two independently-
        rounded float ratios landing on opposite sides of the mathematical
        tie — until compute_signals added a 1e-9 epsilon to the comparison."""
        pos, df = self._pos_and_df(n=24)  # last close = 112.0, exactly at the boundary
        r = compute_signals(pos, df)
        assert r["status"] == "SELL"
        assert "GIVEBACK" in r["reason"]

    @given(
        climb_days=st.integers(min_value=0, max_value=10),
        decline_days=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_stop_price_never_below_initial_stop(self, climb_days, decline_days):
        """Property: stop_price = max(initial_stop, chandelier_stop) — by
        construction it can never be LOWER than initial_stop, for any
        climb/decline shape."""
        closes = [100] * 6
        c = 100
        for _ in range(climb_days):
            c += 1
            closes.append(c)
        for _ in range(decline_days):
            c -= 1
            closes.append(c)
        df, idx = _flat_range_series(closes)
        pos = Position(ticker="T", entry_date=idx[5].date(), entry_price=100.0, shares=10)
        r = compute_signals(pos, df)
        assert r["stop_price"] >= r["initial_stop"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. auto_pipeline.compute_levels
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeLevels:
    """Hypothesis under test: the stop-cap math silently produces a stop
    ABOVE entry (a negative-risk trade) or a target below entry."""

    def test_ground_truth_matches_hand_computation(self):
        idx = pd.bdate_range("2024-01-01", periods=30)
        close = pd.Series([100.0] * 25 + [101, 102, 103, 104, 105], index=idx)
        high = close + 1.0
        low = close - 1.0
        levels = compute_levels(close, high, low, entry=105.0, atr_period=14, atr_mult=1.5, max_stop_pct=7.0)
        # Hand computation (see audit): ATR=2.0, atr_stop=102.0,
        # swing_low(20)=99.0 -> swing_stop=98.01, raw_stop=98.01,
        # min_stop=97.65, stop=max(98.01,97.65)=98.01
        assert levels["stop"] == 98.01
        assert levels["risk_pct"] == 6.66
        assert levels["target_2r"] == 118.98
        assert levels["target_3r"] == 125.97
        assert levels["atr"] == 2.0

    def test_property_stop_is_always_below_entry(self):
        """Property: for any entry/ATR/swing-low combination, the computed
        stop must be strictly below entry — a stop >= entry silently produces
        a negative or undefined per-share risk downstream in sizing."""
        idx = pd.bdate_range("2024-01-01", periods=30)
        close = pd.Series(np.linspace(90, 110, 30), index=idx)
        high = close + 2.0
        low = close - 2.0
        levels = compute_levels(close, high, low, entry=110.0, atr_period=14, atr_mult=1.5, max_stop_pct=7.0)
        assert levels["stop"] < 110.0

    def test_property_max_stop_pct_cap_is_respected(self):
        """Property: stop distance from entry can never exceed max_stop_pct,
        no matter how wide the ATR/swing-low would otherwise push it."""
        idx = pd.bdate_range("2024-01-01", periods=30)
        # A huge range so the uncapped stop would be far wider than 7%
        close = pd.Series([100.0] * 29 + [200.0], index=idx)
        high = close + 50.0
        low = close - 50.0
        levels = compute_levels(close, high, low, entry=200.0, atr_period=14, atr_mult=3.0, max_stop_pct=7.0)
        risk_pct_uncapped_would_be = (200.0 - levels["stop"]) / 200.0 * 100
        assert risk_pct_uncapped_would_be <= 7.0 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 6. auto_pipeline.compute_position_size
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePositionSize:
    """Hypothesis under test: account<=0 raises ZeroDivisionError (fixed in
    this session — regression-locked here) or sizing overspends the risk
    budget."""

    def test_ground_truth(self):
        r = compute_position_size(account=100_000, risk_pct=1.0, entry=50.0, stop=47.0)
        assert r == {"shares": 333, "position_$": 16650.0, "risk_$": 1000.0, "acct_pct": 16.65}

    def test_zero_account_returns_zeros_not_zerodivisionerror(self):
        r = compute_position_size(account=0.0, risk_pct=1.0, entry=50.0, stop=47.0)
        assert r == {"shares": 0, "position_$": 0.0, "risk_$": 0.0, "acct_pct": 0.0}

    def test_negative_account_returns_zeros(self):
        r = compute_position_size(account=-1000.0, risk_pct=1.0, entry=50.0, stop=47.0)
        assert r == {"shares": 0, "position_$": 0.0, "risk_$": 0.0, "acct_pct": 0.0}

    def test_stop_above_entry_uses_the_001_floor_not_negative_risk(self):
        """Boundary: entry < stop (an inverted, invalid setup). per_share_risk
        floors at 0.01 rather than going negative, so shares stays a huge but
        finite positive number, not negative (a negative share count would
        silently mean 'short' in a long-only paper system)."""
        r = compute_position_size(account=10_000, risk_pct=1.0, entry=50.0, stop=55.0)
        assert r["shares"] > 0

    @given(
        account=st.floats(min_value=0.01, max_value=10_000_000, allow_nan=False),
        risk_pct=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
        entry=st.floats(min_value=0.01, max_value=10_000, allow_nan=False),
        stop_delta=st.floats(min_value=0.01, max_value=1000, allow_nan=False),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_never_overspends_the_dollar_risk_budget(self, account, risk_pct, entry, stop_delta):
        """Property: shares * per_share_risk must never exceed dollar_risk —
        the whole point of risk-based sizing is that a stop-out can't lose
        more than the budgeted amount."""
        stop = entry - stop_delta
        r = compute_position_size(account, risk_pct, entry, stop)
        dollar_risk = account * (risk_pct / 100)
        per_share_risk = max(entry - stop, 0.01)
        assert r["shares"] * per_share_risk <= dollar_risk + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 7 & 8. portfolio.PortfolioState.buy / .sell
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioStateBuy:
    """Hypothesis under test: a float-precision cash check silently allows
    (or silently blocks) a boundary-exact buy, or a double-buy corrupts state
    instead of raising."""

    def test_ground_truth_cash_deducted_exactly(self):
        p = PortfolioState(initial_cash=1000.0)
        p.buy("T", date(2024, 1, 1), price=10.0, shares=50)
        assert p.cash == 500.0
        assert p.open_positions["T"].cost_basis == 500.0

    def test_boundary_cost_exactly_equal_to_cash_succeeds(self):
        p = PortfolioState(initial_cash=1000.0)
        p.buy("T", date(2024, 1, 1), price=10.0, shares=100)  # cost == cash exactly
        assert p.cash == 0.0

    def test_boundary_cost_over_tolerance_raises(self):
        p = PortfolioState(initial_cash=1000.0)
        with pytest.raises(ValueError, match="Insufficient cash"):
            p.buy("T", date(2024, 1, 1), price=10.000011, shares=100)  # cost=1000.0011

    def test_double_buy_same_ticker_raises_not_averages_down(self):
        p = PortfolioState(initial_cash=10_000.0)
        p.buy("T", date(2024, 1, 1), 10.0, 10)
        with pytest.raises(ValueError, match="already open"):
            p.buy("T", date(2024, 1, 2), 12.0, 5)

    def test_zero_shares_raises(self):
        p = PortfolioState(initial_cash=10_000.0)
        with pytest.raises(ValueError, match="shares must be > 0"):
            p.buy("T", date(2024, 1, 1), 10.0, 0)

    def test_negative_price_raises(self):
        p = PortfolioState(initial_cash=10_000.0)
        with pytest.raises(ValueError, match="price must be > 0"):
            p.buy("T", date(2024, 1, 1), -10.0, 5)

    @given(
        cash=st.floats(min_value=1, max_value=1_000_000, allow_nan=False),
        price=st.floats(min_value=0.01, max_value=10_000, allow_nan=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_never_goes_negative_from_a_valid_buy(self, cash, price):
        """Property: any buy that doesn't raise must leave cash >= 0."""
        p = PortfolioState(initial_cash=cash)
        shares = max(1, int((cash * 0.5) / price))
        try:
            p.buy("T", date(2024, 1, 1), price, shares)
        except ValueError:
            return  # rejected buys are out of scope for this invariant
        assert p.cash >= -1e-6


class TestPortfolioStateSell:
    """Hypothesis under test: pnl/pnl_pct arithmetic silently uses the wrong
    sign, or selling a nonexistent position silently no-ops instead of
    raising."""

    def test_ground_truth_pnl_and_cash(self):
        p = PortfolioState(initial_cash=10_000.0)
        p.buy("T", date(2024, 1, 1), 10.0, 100)  # cash -> 9000
        trade = p.sell("T", date(2024, 1, 5), 12.0)
        assert p.cash == 9000.0 + 1200.0
        assert trade.pnl == 200.0
        assert trade.pnl_pct == pytest.approx(20.0)
        assert p.realized_pnl == 200.0
        assert "T" not in p.open_positions

    def test_sell_nonexistent_ticker_raises_keyerror_not_silent_noop(self):
        p = PortfolioState(initial_cash=10_000.0)
        with pytest.raises(KeyError):
            p.sell("NOPE", date(2024, 1, 1), 10.0)

    def test_sell_at_zero_price_raises(self):
        p = PortfolioState(initial_cash=10_000.0)
        p.buy("T", date(2024, 1, 1), 10.0, 10)
        with pytest.raises(ValueError, match="price must be > 0"):
            p.sell("T", date(2024, 1, 2), 0.0)

    def test_sell_at_a_loss_produces_negative_pnl(self):
        p = PortfolioState(initial_cash=10_000.0)
        p.buy("T", date(2024, 1, 1), 10.0, 100)
        trade = p.sell("T", date(2024, 1, 5), 8.0)
        assert trade.pnl == -200.0
        assert trade.pnl_pct == pytest.approx(-20.0)

    def test_open_position_unrealized_pnl_pct_guards_entry_price_zero(self):
        """Boundary: a zero entry_price (corrupt state) must not raise
        ZeroDivisionError from the property helper."""
        pos = OpenPosition(ticker="T", entry_date=date(2024, 1, 1), entry_price=0.0, shares=10)
        assert pos.unrealized_pnl_pct(50.0) == 0.0

    @given(
        entry_price=st.floats(min_value=0.01, max_value=10_000, allow_nan=False),
        sell_price=st.floats(min_value=0.01, max_value=10_000, allow_nan=False),
        shares=st.integers(min_value=1, max_value=10_000),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_pnl_sign_matches_price_direction(self, entry_price, sell_price, shares):
        """Property: pnl > 0 iff sell_price > entry_price; pnl == 0 iff
        equal — the sign of realised P&L must always track the direction of
        the price move, for any entry/exit/size combination."""
        p = PortfolioState(initial_cash=entry_price * shares)
        p.buy("T", date(2024, 1, 1), entry_price, shares)
        trade = p.sell("T", date(2024, 1, 2), sell_price)
        if sell_price > entry_price:
            assert trade.pnl > 0
        elif sell_price < entry_price:
            assert trade.pnl < 0
        else:
            assert trade.pnl == 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. backtest_runner._mark_to_market / ._day_close_price
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkToMarketAndDayClosePrice:
    """Regression lock for this session's fix: a ticker whose data has gone
    stale (delisted/halted/vendor dropped coverage) must NOT be silently
    marked at its frozen last-known price forever."""

    def _zombie_provider(self):
        dates = pd.bdate_range("2024-01-01", periods=8)
        df = _ohlcv(dates, [50] * 8, [51] * 8, [49] * 8, [50] * 8, [100_000] * 8)
        return HistoricalSliceProvider({"ZOMBIE": df}), dates

    def test_ground_truth_fresh_data_prices_normally(self):
        provider, dates = self._zombie_provider()
        assert _day_close_price("ZOMBIE", provider, dates[-1]) == 50.0

    def test_stale_data_day_close_price_returns_none_not_frozen_price(self):
        """Hypothesis under attack: a ticker whose feed went dark 65 calendar
        days ago is silently priced at its last real close."""
        provider, dates = self._zombie_provider()
        as_of = dates[-1] + pd.Timedelta(days=65)
        assert _day_close_price("ZOMBIE", provider, as_of) is None

    def test_stale_data_mark_to_market_falls_back_to_cost_basis(self):
        provider, dates = self._zombie_provider()
        pos = OpenPosition(ticker="ZOMBIE", entry_date=dates[0].date(), entry_price=50.0, shares=100)
        as_of = dates[-1] + pd.Timedelta(days=65)
        value = _mark_to_market({"ZOMBIE": pos}, provider, as_of)
        assert value == pos.cost_basis == 5000.0

    def test_fresh_data_within_the_gap_tolerance_still_prices_live(self):
        """Boundary: a gap of exactly the tolerance (10 days) must still be
        treated as fresh, not stale — pins the other side of the boundary."""
        provider, dates = self._zombie_provider()
        as_of = dates[-1] + pd.Timedelta(days=10)
        assert _day_close_price("ZOMBIE", provider, as_of) == 50.0

    def test_just_past_the_gap_tolerance_is_treated_as_stale(self):
        provider, dates = self._zombie_provider()
        as_of = dates[-1] + pd.Timedelta(days=11)
        assert _day_close_price("ZOMBIE", provider, as_of) is None
