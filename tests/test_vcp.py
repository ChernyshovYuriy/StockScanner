"""VCP study Phase 2 -- gate, base, zigzag and score. Synthetic data only."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from research.vcp import vcp
from research.vcp.vcp import (
    DEFAULT, GATE_CHECKS, Bars, Contraction, VcpConfig, atr_scale, base_score, clean_bars,
    dryup_ratio, dryup_score, find_base_start, find_contractions, find_swings, gate_fail_reason, gate_table,
    is_valid_vcp, keeps_shrinking, merge_swings, missing_fraction, prior_uptrend, progression_score,
    proximity_score, rs_percentiles, score_at, score_ticker, tightness_score,
)

RS = 90.0
EVERY_SWING = replace(DEFAULT, merge_swings=False)


# ── builders ─────────────────────────────────────────────────────────────────

def frame(closes, volume=1_000_000.0, wiggle=0.002, start="2020-01-01"):
    c = np.asarray(closes, float)
    idx = pd.bdate_range(start, periods=len(c))
    v = np.broadcast_to(np.asarray(volume, float), c.shape)
    return pd.DataFrame({"Open": c, "High": c * (1 + wiggle), "Low": c * (1 - wiggle),
                         "Close": c, "Volume": v.copy()}, index=idx)


def path(*legs, start):
    """legs: (target, bars) straight lines, continuing from `start`."""
    out = [float(start)]
    for target, n in legs:
        out += list(np.linspace(out[-1], target, n + 1)[1:])
    return out


def uptrend(n=300, lo=30.0, hi=100.0):
    return list(np.geomspace(lo, hi, n))


# 100 -> 80 -> 97 -> 87.3 -> 96 -> 92.2 -> 94.5 : depths about 20%, 10%, 4%
CLEAN_LEGS = [(80, 20), (97, 20), (87.3, 12), (96, 12), (92.2, 6), (94.5, 4)]
# 100 -> 96 -> 99 -> 89 -> 98 -> 78.4 -> 94.5 : depths about 4%, 10%, 20%
LOOSE_LEGS = [(96, 6), (99, 8), (89, 12), (98, 12), (78.4, 20), (94.5, 16)]


def clean_vcp(dry=True):
    c = uptrend() + path(*CLEAN_LEGS, start=100)[1:]
    v = np.full(len(c), 1_000_000.0)
    if dry:
        v[-10:] = 350_000.0
    return frame(c, v)


def loose_pattern():
    c = uptrend() + path(*LOOSE_LEGS, start=100)[1:]
    v = np.full(len(c), 1_000_000.0)
    v[-40:] = 1_800_000.0
    return frame(c, v)


def bars_of(high, low, close=None, volume=None):
    n = len(high)
    close = np.asarray(low if close is None else close, float)
    volume = np.full(n, 1e6) if volume is None else np.asarray(volume, float)
    return Bars(pd.bdate_range("2020-01-01", periods=n).to_numpy(), close, np.asarray(high, float),
                np.asarray(low, float), close, volume, np.arange(n))


def random_walk(seed, n=420, vol=0.02, drift=0.0012, start=20.0):
    rng = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    spread = np.abs(rng.normal(0, vol / 2, n)) + 0.001
    o = c * (1 + rng.normal(0, vol / 3, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + spread),
                         "Low": np.minimum(o, c) * (1 - spread), "Close": c,
                         "Volume": rng.integers(200_000, 3_000_000, n).astype(float)},
                        index=pd.bdate_range("2020-01-01", periods=n))


# ── the four cases the brief asks for ────────────────────────────────────────

def test_clean_three_contraction_vcp_scores_high():
    r = score_ticker(clean_vcp(), RS)
    assert r["gate_passed"] and r["valid_vcp"] and r["n_contractions"] == 3
    assert r["contraction_depths"] == pytest.approx([20.3, 10.4, 4.3], abs=0.2)
    assert r["vcp_score"] >= 85


def test_widening_pattern_scores_low():
    r = score_ticker(loose_pattern(), RS)
    assert r["gate_passed"], r["gate_fail_reason"]
    assert not r["valid_vcp"] and r["n_contractions"] == 1      # each wider leg swallows the last
    assert r["vcp_score"] == 0.0
    r = score_ticker(loose_pattern(), RS, cfg=EVERY_SWING)
    assert not r["valid_vcp"] and r["n_contractions"] == 3
    assert r["c_progression"] == 0.0
    assert r["vcp_score"] <= DEFAULT.invalid_cap
    assert score_ticker(clean_vcp(), RS)["vcp_score"] - r["vcp_score"] > 40


def test_extended_stock_gets_zero_proximity():
    c = uptrend() + path(*CLEAN_LEGS, (103, 2), start=100)[1:]
    r = score_ticker(frame(c), RS)
    assert r["gate_passed"] and r["n_contractions"] == 3
    assert r["pct_from_pivot"] > 5
    assert r["c_proximity"] == 0.0


def test_below_200_day_ma_fails_gate():
    c = uptrend() + path((45, 30), start=100)[1:]
    r = score_ticker(frame(c), RS)
    assert not r["gate_passed"] and r["vcp_score"] == 0.0
    assert r["gate_fail_reason"] in ("price_above_ma150", "price_above_ma200")


# ── gate, one check at a time ────────────────────────────────────────────────

def _table(**over):
    row = {k: True for k in GATE_CHECKS} | {"ready": True, "rs_raw": 0.5, "atr_pct": 0.02} | over
    return pd.DataFrame([row] * 300)


@pytest.mark.parametrize("name", GATE_CHECKS)
def test_each_check_alone_fails_the_gate(name):
    b = bars_of(np.ones(300), np.ones(300))
    assert gate_fail_reason(b, _table(), 299, RS) is None
    assert gate_fail_reason(b, _table(**{name: False}), 299, RS) == name


@pytest.mark.parametrize("rs,reason", [(70.0, None), (69.99, "rs_percentile"),
                                       (np.nan, "rs_unavailable"), (None, "rs_unavailable")])
def test_rs_gate(rs, reason):
    assert gate_fail_reason(bars_of(np.ones(300), np.ones(300)), _table(), 299, rs) == reason


def test_not_ready_and_thin_data_come_first():
    b = bars_of(np.ones(300), np.ones(300))
    assert gate_fail_reason(b, _table(ready=False), 299, RS) == "insufficient_history"
    gappy = replace(b, session=np.where(np.arange(300) > 280, np.arange(300) + 29, np.arange(300)))
    assert gate_fail_reason(gappy, _table(), 299, RS) == "thin_data"         # 29/281 = 10.3%
    ok = replace(b, session=np.where(np.arange(300) > 280, np.arange(300) + 27, np.arange(300)))
    assert gate_fail_reason(ok, _table(), 299, RS) is None                   # 27/279 = 9.7%


@pytest.mark.parametrize("seed", range(40))
def test_gate_table_matches_a_plain_recomputation(seed):
    df = random_walk(seed, drift=0.0003, start=1.2)
    b = clean_bars(df)
    g = gate_table(b)
    c, h, l, v = b.close, b.high, b.low, b.volume
    for i in range(252, 420):
        ma = lambda n, j=i: c[j - n + 1:j + 1].mean()
        want = {
            "liquidity_price": c[i] >= 1.0,
            "liquidity_dollar_volume": (c * v)[i - 19:i + 1].mean() >= 1_000_000,
            "price_above_ma150": c[i] > ma(150), "price_above_ma200": c[i] > ma(200),
            "ma150_above_ma200": ma(150) > ma(200), "ma200_rising": ma(200) > ma(200, i - 21),
            "ma50_above_ma150": ma(50) > ma(150), "ma50_above_ma200": ma(50) > ma(200),
            "price_above_ma50": c[i] > ma(50),
            "above_52w_low": c[i] >= l[i - 251:i + 1].min() * 1.25,
            "near_52w_high": c[i] >= h[i - 251:i + 1].max() * 0.75,
        }
        assert {k: bool(g[k].iat[i]) for k in want} == want
        rs = 0.4 * (c[i] / c[i - 63] - 1) + sum(0.2 * (c[i] / c[i - d] - 1) for d in (126, 189, 252))
        assert g["rs_raw"].iat[i] == pytest.approx(rs)


def test_gate_checks_are_exercised_both_ways():
    """The recomputation above only means something if each check varies."""
    cols = {k: set() for k in GATE_CHECKS}
    for seed in range(40):
        g = gate_table(clean_bars(random_walk(seed, drift=0.0003, start=1.2)))
        for k in GATE_CHECKS:
            cols[k] |= set(g[k].iloc[252:].unique())
    assert all(v == {True, False} for v in cols.values()), cols


def test_ready_needs_a_full_year_back():
    g = gate_table(clean_bars(random_walk(1)))
    assert not g["ready"].iat[251] and g["ready"].iat[252]
    assert np.isnan(g["rs_raw"].iat[251]) and np.isfinite(g["rs_raw"].iat[252])


def test_one_bar_below_ma50_only():
    c = uptrend() + path(*CLEAN_LEGS, start=100)[1:]
    df = frame(c)
    df.iloc[-1, df.columns.get_indexer(["Open", "High", "Low", "Close"])] = [89.0, 89.2, 88.8, 89.0]
    assert score_ticker(df, RS)["gate_fail_reason"] == "price_above_ma50"


def test_rs_percentiles():
    r = rs_percentiles(pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": np.nan}))
    assert r[["a", "b", "c", "d"]].tolist() == [25.0, 50.0, 75.0, 100.0] and np.isnan(r["e"])


# ── bars: gaps and halts ─────────────────────────────────────────────────────

def test_untraded_bars_are_dropped_not_filled():
    df = random_walk(5, n=30)
    df.iloc[4, df.columns.get_loc("Volume")] = 0
    df.iloc[7, df.columns.get_loc("Close")] = np.nan
    df.iloc[9, df.columns.get_loc("Low")] = 0.0
    df.iloc[11, df.columns.get_loc("Volume")] = np.nan
    b = clean_bars(df)
    assert len(b) == 26
    assert b.session.tolist() == [k for k in range(30) if k not in (4, 7, 9, 11)]


def test_calendar_exposes_missing_rows():
    df = random_walk(5, n=30)
    b = clean_bars(df.drop(df.index[10:16]), calendar=df.index)
    assert len(b) == 24 and b.session[10] == 16
    assert missing_fraction(b, 0, len(b) - 1) == pytest.approx(6 / 30)
    assert missing_fraction(b, 0, 9) == 0.0


def test_rows_off_the_calendar_are_dropped():
    df = random_walk(5, n=30)
    cal = df.index.delete([3])
    b = clean_bars(df, cal)
    assert len(b) == 29 and missing_fraction(b, 0, 28) == 0.0


def test_unsorted_and_duplicate_rows():
    df = random_walk(5, n=30)
    messy = pd.concat([df.iloc[::-1], df.iloc[[12]]])
    a, b = clean_bars(df), clean_bars(messy)
    assert np.array_equal(a.close, b.close) and np.array_equal(a.session, b.session)


def test_halt_cannot_create_a_tight_range():
    """Padding a halt with flat zero-volume bars must not change the score."""
    df = clean_vcp()
    pad_idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=5)
    last = df["Close"].iloc[-1]
    pad = pd.DataFrame({"Open": last, "High": last, "Low": last, "Close": last, "Volume": 0.0}, index=pad_idx)
    a = score_ticker(df, RS)
    b = score_ticker(pd.concat([df, pad]), RS)
    for k in ("vcp_score", "n_contractions", "final_depth_pct", "vol_dryup_ratio", "base_length_days"):
        assert a[k] == pytest.approx(b[k])


def test_stale_and_empty():
    df = clean_vcp()
    cal = pd.bdate_range(df.index[0], periods=len(df) + 6)
    assert score_ticker(df, RS, calendar=cal)["gate_fail_reason"] == "stale_data"
    assert score_ticker(df, RS, calendar=cal[:-1])["gate_passed"]
    dead = df.assign(Volume=0.0)
    assert score_ticker(dead, RS)["gate_fail_reason"] == "no_data"
    assert score_ticker(df.iloc[:100], RS)["gate_fail_reason"] == "insufficient_history"


def test_thin_base_scores_zero():
    df = clean_vcp()
    thin = df.drop(df.index[-12:-2])                    # 10 of 74 base sessions untraded
    r = score_ticker(thin, RS, calendar=df.index)
    assert r["gate_passed"] and r["no_score_reason"] == "thin_base" and r["vcp_score"] == 0


# ── base ─────────────────────────────────────────────────────────────────────

def test_base_start_is_highest_high_outside_the_last_three_weeks():
    high = np.full(400, 10.0)
    high[300], high[390] = 20.0, 30.0                   # 390 is 9 bars ago: too recent
    b = bars_of(high, high - 1)
    assert find_base_start(b, 399) == 300


def test_base_start_boundaries():
    high = np.full(400, 10.0)
    high[399 - 15] = 20.0
    assert find_base_start(bars_of(high, high), 399) == 384
    high = np.full(400, 10.0)
    high[399 - 325], high[399 - 326] = 20.0, 30.0
    assert find_base_start(bars_of(high, high), 399) == 74
    assert find_base_start(bars_of(high, high), 14) is None
    assert find_base_start(bars_of(high, high), 15) == 0


def test_base_start_takes_the_latest_of_equal_highs():
    high = np.full(100, 10.0)
    high[[40, 60]] = 20.0
    assert find_base_start(bars_of(high, high), 99) == 60


# ── zigzag ───────────────────────────────────────────────────────────────────

def contractions(prices, thr=0.05, start=0):
    p = np.asarray(prices, float)
    return [(c.high_idx, c.low_idx) for c in find_swings(p, p, start, len(p) - 1, thr)]


def test_zigzag_textbook():
    assert contractions([100, 90, 80, 90, 97, 92, 87, 93, 96, 91]) == [(0, 2), (4, 6), (8, 9)]


def test_zigzag_ignores_moves_below_threshold():
    assert contractions([100, 97, 100, 97, 100]) == []
    assert contractions([100, 80, 83, 79, 82, 78]) == [(0, 5)]         # bounces under 5%: one leg


def test_zigzag_threshold_is_inclusive():
    assert contractions([100, 95, 98]) == [(0, 1)]
    assert contractions([100, 95.01, 98]) == []
    assert contractions([100, 80, 84, 79]) == [(0, 1), (2, 3)]         # exactly +5%, then -6%
    assert contractions([100, 80, 83.99, 79]) == [(0, 3)]


def test_zigzag_drop_from_a_later_high_is_inclusive_too():
    assert contractions([100, 80, 90, 85.5]) == [(0, 1), (2, 3)]        # exactly -5% from 90
    assert contractions([100, 80, 90, 85.51]) == [(0, 1)]


def test_zigzag_unfinished_low_counts_but_shallow_dip_does_not():
    assert contractions([100, 80, 95, 90]) == [(0, 1), (2, 3)]
    assert contractions([100, 80, 95, 91]) == [(0, 1)]


def test_zigzag_new_high_before_any_pullback_moves_the_start():
    assert contractions([100, 98, 110, 100]) == [(2, 3)]


def test_zigzag_same_bar_low_and_rally_is_not_a_reversal():
    high = np.array([100, 90, 99.0, 88])
    low = np.array([99, 85, 80.0, 87])
    legs = find_swings(high, low, 0, 3, 0.05)
    # bar 2 makes the low; its own 99 high can't confirm it, bar 3's 88 (+10%) does
    assert [(c.high_idx, c.low_idx) for c in legs] == [(0, 2)]
    assert legs[0].depth == pytest.approx(0.20)


def test_zigzag_outside_bar_at_a_high_does_not_flip():
    high = np.array([100, 85, 95, 96.0, 90])
    low = np.array([99, 80, 90, 85.0, 89])
    # bar 3 sets the swing high (96); its 85 low can't also end it, bar 4's 89 low (-7%) does
    legs = find_swings(high, low, 0, 4, 0.05)
    assert [(c.high_idx, c.low_idx) for c in legs] == [(0, 1), (3, 4)]


def test_zigzag_bar_above_the_swing_high_ends_the_pullback_even_if_it_also_makes_a_low():
    high = np.array([100, 90, 101.0, 100, 93])
    low = np.array([99, 85, 80.0, 94, 90])
    legs = find_swings(high, low, 0, 4, 0.05)
    assert [(c.high_idx, c.low_idx, c.high, c.low) for c in legs] == [(0, 1, 100, 85), (2, 4, 101, 90)]


def test_zigzag_starts_at_start():
    assert contractions([200, 50, 100, 80, 95, 85], start=2) == [(2, 3), (4, 5)]


@settings(max_examples=300, deadline=None)
@given(st.integers(0, 10**6), st.sampled_from([0.03, 0.06, 0.1]))
def test_zigzag_properties(seed, thr):
    df = random_walk(seed, n=150, vol=0.03)
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    legs = find_swings(high, low, 0, 149, thr)
    prev_low = -1
    for n, c in enumerate(legs):
        assert prev_low <= c.high_idx < c.low_idx or (n and prev_low < c.high_idx < c.low_idx)
        assert c.depth >= thr - 1e-12
        assert c.high == high[c.high_idx] and c.low == low[c.low_idx]
        assert c.low == low[c.high_idx + 1:c.low_idx + 1].min()            # the lowest of its own leg
        if n:
            rally = high[legs[n - 1].low_idx + 1:c.low_idx]
            assert c.high == rally.max()                                    # the highest since the last low
            assert c.high >= legs[n - 1].low * (1 + thr) - 1e-9             # a real rally in between
        prev_low = c.low_idx


@settings(max_examples=150, deadline=None)
@given(st.integers(0, 10**6), st.integers(40, 148))
def test_zigzag_past_swings_never_change(seed, cut):
    """Everything but the newest, unfinished leg is final once seen."""
    df = random_walk(seed, n=150, vol=0.03)
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    early = find_swings(high, low, 0, cut, 0.06)
    late = find_swings(high, low, 0, 149, 0.06)
    assert early[:-1] == late[:max(len(early) - 1, 0)]


# ── merging wiggles ──────────────────────────────────────────────────────────

def merged(*legs):
    """legs: (high, low) swings in time order -> merged (high, low) list."""
    sw = [Contraction(2 * k, 2 * k + 1, float(h), float(l)) for k, (h, l) in enumerate(legs)]
    return [(c.high, c.low) for c in merge_swings(sw)]


def test_merge_leaves_a_textbook_vcp_alone():
    assert merged((100, 80), (97, 87), (96, 92)) == [(100, 80), (97, 87), (96, 92)]


def test_merge_folds_bounces_inside_a_decline():
    assert merged((100, 92), (95, 86), (90, 80), (97, 87)) == [(100, 80), (97, 87)]


def test_merge_folds_dips_inside_a_recovery():
    assert merged((100, 80), (86, 83), (91, 87), (97, 88), (96, 92)) == [(100, 80), (97, 88), (96, 92)]


def test_merge_widening_collapses_to_one():
    assert merged((100, 96), (99, 89), (98, 78)) == [(100, 78)]


def test_merge_first_contraction_starts_at_the_base_high_even_after_a_breakout():
    assert merged((100, 80), (97, 87), (103, 99)) == [(100, 80), (103, 99)]


def test_merge_ties_take_the_later_high_and_the_later_low():
    sw = [Contraction(0, 1, 100., 80.), Contraction(2, 3, 97., 85.), Contraction(4, 5, 97., 90.)]
    assert [(c.high_idx, c.low_idx) for c in merge_swings(sw)] == [(0, 1), (4, 5)]
    sw = [Contraction(0, 1, 100., 80.), Contraction(2, 3, 95., 80.), Contraction(4, 5, 93., 90.)]
    assert [(c.high_idx, c.low_idx) for c in merge_swings(sw)] == [(0, 3), (4, 5)]


def test_merge_of_nothing():
    assert merge_swings([]) == []


@settings(max_examples=300, deadline=None)
@given(st.integers(0, 10**6), st.sampled_from([0.03, 0.06]))
def test_merge_properties(seed, thr):
    df = random_walk(seed, n=200, vol=0.03)
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    start = int(np.argmax(high[:150]))
    swings = find_swings(high, low, start, 199, thr)
    legs = merge_swings(swings)
    assert bool(legs) == bool(swings)
    if not legs:
        return
    assert legs[0].high_idx == swings[0].high_idx
    assert legs[0].low == min(c.low for c in swings)
    assert {c.high_idx for c in legs} <= {c.high_idx for c in swings}
    assert {c.low_idx for c in legs} <= {c.low_idx for c in swings}
    assert legs[-1].low_idx == swings[-1].low_idx or legs[-1].low <= swings[-1].low
    for a, b in zip(legs, legs[1:]):
        assert a.high_idx < a.low_idx < b.high_idx < b.low_idx
        assert a.low < b.low                                        # lows step up
        assert all(c.high <= b.high for c in swings if c.high_idx > a.low_idx)
    for c in legs:
        assert c.depth >= thr - 1e-12


def test_merge_flag_selects_the_method():
    p = np.array([100, 92, 97, 80, 97, 87, 93.0])
    a = find_contractions(p, p, 0, 6, 0.05)
    b = find_contractions(p, p, 0, 6, 0.05, EVERY_SWING)
    assert [(c.high_idx, c.low_idx) for c in a] == [(0, 3), (4, 5)]
    assert [(c.high_idx, c.low_idx) for c in b] == [(0, 1), (2, 3), (4, 5)]


def test_long_choppy_base_is_found():
    """A 30-week base with a 4% wiggle riding on every leg is still 25 / 12 / 6."""
    base = np.array(path((75, 50), (97, 40), (85.4, 25), (96, 20), (90.2, 12), (94, 5), start=100)[1:])
    wiggle = 1 + 0.022 * np.where(np.arange(len(base)) // 2 % 2, 1, -1)
    wiggle[[49, 89, 114, 134, 146]] = 1.0
    b = clean_bars(frame(uptrend() + list(base * wiggle), wiggle=0.001))
    r = vcp._pattern(b, len(b) - 1, 0.02, DEFAULT)
    swings = vcp._pattern(b, len(b) - 1, 0.02, EVERY_SWING)
    assert swings["n_contractions"] > 10 and not swings["valid_vcp"]
    assert r["valid_vcp"] and 3 <= r["n_contractions"] <= 4, r
    assert r["base_length_days"] >= 140 and r["vcp_score"] > 60
    assert r["contraction_depths"][0] == pytest.approx(26.7, abs=1.0)


# ── components, by hand ──────────────────────────────────────────────────────

def test_progression_by_hand():
    assert progression_score([0.25, 0.12, 0.06, 0.03]) == 1.0                # average shrink 50.7%
    assert progression_score([0.20, 0.16]) == pytest.approx(0.5 + 0.5 * 0.2 / 0.4)
    assert progression_score([0.05, 0.10, 0.20]) == 0.0
    # steps +50%, -20%: half consistent, mean shrink 15%
    assert progression_score([0.20, 0.10, 0.12]) == pytest.approx(0.25 + 0.5 * 0.15 / 0.4)
    assert progression_score([0.2]) == progression_score([]) == 0.0
    assert progression_score([0.1, 0.1]) == 0.0
    # 10% smaller is a smaller step than 0.85x: no consistency, half of 10%/40%
    assert progression_score([0.2, 0.18]) == pytest.approx(0.5 * 0.1 / 0.4)


@pytest.mark.parametrize("d,want", [(0.03, 1), (0.05, 1), (0.075, 0.8), (0.10, 0.6), (0.15, 0.3), (0.2, 0), (0.4, 0)])
def test_tightness_by_hand(d, want):
    assert tightness_score(d, 1.0) == pytest.approx(want)
    assert tightness_score(d * 2, 2.0) == pytest.approx(want)


@pytest.mark.parametrize("r,want", [(0.3, 1), (0.5, 1), (0.75, 0.5), (1.0, 0), (1.6, 0)])
def test_dryup_by_hand(r, want):
    assert dryup_score(r) == pytest.approx(want)


@pytest.mark.parametrize("p,want", [(-0.3, 0), (-0.15, 0), (-0.09, 0.5), (-0.03, 1), (0.0, 1),
                                    (0.025, 0.25), (0.0499999, 0), (0.0501, 0), (0.2, 0)])
def test_proximity_by_hand(p, want):
    assert proximity_score(p) == pytest.approx(want, abs=1e-5)


def test_proximity_below_pivot_beats_just_above():
    assert proximity_score(-0.01) > proximity_score(0.001) > proximity_score(0.03)


def test_base_score_by_hand():
    assert base_score(0.30, 0.50, 1.0) == 1.0
    assert base_score(0.425, 0.50, 1.0) == pytest.approx(0.75)
    assert base_score(0.60, 0.15, 1.0) == pytest.approx(0.25)
    assert base_score(0.60, 0.0, 2.0) == pytest.approx(0.5)                  # 60% / 2 = 30%


def test_dryup_ratio_by_hand():
    v = np.full(100, 100.0)
    v[91:95] = 40.0
    b = bars_of(np.ones(100), np.ones(100), volume=v)
    avg = (46 * 100 + 4 * 40) / 50
    # high 90, low 94: bars 91-94 plus one more to make five
    assert dryup_ratio(b, Contraction(90, 94, 1, 1), 99) == pytest.approx((4 * 40 + 100) / 5 / avg)
    # a long leg is read to its low only
    assert dryup_ratio(b, Contraction(84, 94, 1, 1), 99) == pytest.approx((6 * 100 + 4 * 40) / 10 / avg)
    # never past the bar being scored
    avg93 = (47 * 100 + 3 * 40) / 50
    assert dryup_ratio(b, Contraction(90, 93, 1, 1), 93) == pytest.approx(40 / avg93)


def test_dryup_ignores_breakout_volume_after_the_low():
    v = np.full(100, 100.0)
    v[97:] = 10_000.0
    b = bars_of(np.ones(100), np.ones(100), volume=v)
    assert dryup_ratio(b, Contraction(85, 94, 1, 1), 99) < 0.2


def test_prior_uptrend_window():
    low = np.full(300, 50.0)
    low[60] = 10.0                                      # 140 bars before the base high: outside
    b = bars_of(np.full(300, 80.0), low)
    assert prior_uptrend(b, 200) == pytest.approx(0.6)
    low[74] = 20.0                                      # exactly 126 bars before: inside
    assert prior_uptrend(bars_of(np.full(300, 80.0), low), 200) == pytest.approx(3.0)


@pytest.mark.parametrize("atr,want", [(0.01, 1.0), (0.025, 1.0), (0.05, 2.0), (0.2, 2.5), (np.nan, 1.0)])
def test_atr_scale(atr, want):
    assert atr_scale(atr) == pytest.approx(want)
    assert atr_scale(atr, replace(DEFAULT, scale_by_atr=False)) == 1.0


@pytest.mark.parametrize("depths,ok", [([.2, .1], True), ([.2], False), ([.2, .2], False), ([.2, .1, .12], False),
                                       ([.6, .5, .4, .3, .2, .1], True), ([.7, .59, .5, .4, .3, .2, .1], False),
                                       ([.2, .17], True), ([.2, .1701], False)])
def test_valid_vcp(depths, ok):
    assert is_valid_vcp(depths) is ok


def test_valid_vcp_uses_the_given_ratio():
    assert not is_valid_vcp([.2, .18]) and is_valid_vcp([.2, .18], replace(DEFAULT, max_shrink_ratio=0.95))


# ── the whole score ──────────────────────────────────────────────────────────

def test_weights_total_100():
    c = DEFAULT
    assert c.w_progression + c.w_tightness + c.w_dryup + c.w_proximity + c.w_base == 100.0


def test_rs_weights_total_1_and_liquidity_matches_swing_tickers():
    from swing_tickers import Thresholds
    assert sum(w for _, w in DEFAULT.rs_periods) == pytest.approx(1.0)
    assert (DEFAULT.min_price, DEFAULT.min_dollar_vol) == (Thresholds().min_price, Thresholds().min_avg_dollar_vol_20)


def test_score_is_the_sum_of_components_when_valid():
    r = score_ticker(clean_vcp(), RS)
    parts = [r[k] for k in ("c_progression", "c_tightness", "c_dryup", "c_proximity", "c_base")]
    assert r["vcp_score"] == pytest.approx(sum(parts)) and all(p > 0 for p in parts)


def test_invalid_pattern_is_capped():
    r = score_ticker(loose_pattern(), RS, cfg=replace(EVERY_SWING, invalid_cap=5.0))
    assert not r["valid_vcp"] and r["vcp_score"] == 5.0


def test_one_contraction_scores_zero():
    c = uptrend() + path((85, 12), (99, 12), start=100)[1:]
    r = score_ticker(frame(c), RS)
    assert r["gate_passed"] and r["n_contractions"] == 1
    assert r["vcp_score"] == 0.0 and r["no_score_reason"] == "too_few_contractions"


def test_straight_uptrend_has_no_vcp():
    r = score_ticker(frame(uptrend(400)), RS)
    assert r["gate_passed"] and r["n_contractions"] == 0 and r["vcp_score"] == 0.0


def test_volume_dryup_raises_the_score():
    assert score_ticker(clean_vcp(True), RS)["vcp_score"] > score_ticker(clean_vcp(False), RS)["vcp_score"] + 10


def test_pivot_is_high_of_final_contraction():
    r = score_ticker(clean_vcp(), RS)
    assert r["pivot_price"] == pytest.approx(96 * 1.002)
    assert r["pct_from_pivot"] == pytest.approx((94.5 / (96 * 1.002) - 1) * 100)
    assert r["base_length_days"] == 74
    assert pd.Timestamp(r["base_start"]) == clean_vcp().index[299]


def test_output_fields_always_present():
    want = {"vcp_score", "n_contractions", "contraction_depths", "final_depth_pct", "vol_dryup_ratio",
            "pivot_price", "pct_from_pivot", "base_length_days", "gate_passed", "gate_fail_reason"}
    frames = [clean_vcp(), loose_pattern(), frame(uptrend(50)), clean_vcp().assign(Volume=0.0)]
    keys = [set(score_ticker(f, RS)) for f in frames]
    assert all(want <= k for k in keys) and all(k == keys[0] for k in keys)


def test_zigzag_threshold_widens_with_atr():
    """Two 4% dips: swings for a quiet stock, noise for one with a 6% daily range."""
    c = uptrend() + path((85, 10), (97, 8), (93.1, 4), (96.5, 4), start=100)[1:]
    b = clean_bars(frame(c))
    i = len(b) - 1
    assert vcp._pattern(b, i, 0.02, DEFAULT)["n_contractions"] == 2
    assert vcp._pattern(b, i, 0.06, DEFAULT)["n_contractions"] == 1
    assert vcp._pattern(b, i, 0.06, replace(DEFAULT, scale_by_atr=False))["n_contractions"] == 2


# ── properties ───────────────────────────────────────────────────────────────

def _same(a, b):
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float):
            assert (np.isnan(x) and np.isnan(y)) or x == pytest.approx(y, rel=1e-6), k
        elif isinstance(x, list):
            assert x == pytest.approx(y, rel=1e-6), k
        else:
            assert x == y, k


NO_GATE = replace(DEFAULT, min_price=0, min_dollar_vol=0, min_above_low=-9, max_below_high=9)


def _ungated(df, cfg=NO_GATE):
    """The pattern part on its own, so random walks exercise it every time."""
    b = clean_bars(df)
    return vcp._pattern(b, len(b) - 1, float(gate_table(b, cfg)["atr_pct"].iat[-1]), cfg)


@settings(max_examples=120, deadline=None)
@given(st.integers(0, 10**6), st.integers(300, 415))
def test_future_bars_never_change_a_past_score(seed, i):
    df = random_walk(seed)
    full = clean_bars(df)
    cut = clean_bars(df.iloc[:i + 1])
    ga, gb = gate_table(full), gate_table(cut)
    pd.testing.assert_frame_equal(ga.iloc[:i + 1], gb)
    _same(score_at(full, ga, i, RS), score_at(cut, gb, i, RS))
    _same(vcp._pattern(full, i, float(ga["atr_pct"].iat[i]), DEFAULT),
          vcp._pattern(cut, i, float(gb["atr_pct"].iat[i]), DEFAULT))


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10**6))
def test_absurd_future_does_not_leak(seed):
    df = random_walk(seed)
    wild = df.copy()
    wild.iloc[351:, :4] *= 50.0
    wild.iloc[351:, 4] *= 1000
    a, b = clean_bars(df), clean_bars(wild)
    _same(score_at(a, gate_table(a), 350, RS), score_at(b, gate_table(b), 350, RS))
    _same(vcp._pattern(a, 350, 0.03, DEFAULT), vcp._pattern(b, 350, 0.03, DEFAULT))


@settings(max_examples=150, deadline=None)
@given(st.integers(0, 10**6), st.sampled_from([0.01, 0.02, 0.04]))
def test_score_bounds(seed, vol):
    r = _ungated(random_walk(seed, vol=vol))
    assert 0 <= r["vcp_score"] <= 100
    for k, w in [("c_progression", 30), ("c_tightness", 20), ("c_dryup", 20), ("c_proximity", 20), ("c_base", 10)]:
        assert 0 <= r[k] <= w + 1e-9
    assert r["valid_vcp"] or r["vcp_score"] <= DEFAULT.invalid_cap
    assert r["base_length_days"] == 0 or 15 <= r["base_length_days"] <= 325


@settings(max_examples=80, deadline=None)
@given(st.integers(0, 10**6), st.sampled_from([0.25, 4.0, 1024.0]))
def test_price_scale_invariance(seed, k):
    df = random_walk(seed)
    big = df.copy()
    big.iloc[:, :4] *= k
    a, b = _ungated(df), _ungated(big)
    b["pivot_price"] /= k
    _same(a, b)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10**6), st.sampled_from([0.5, 8.0]))
def test_volume_scale_invariance(seed, k):
    df = random_walk(seed)
    _same(_ungated(df), _ungated(df.assign(Volume=df["Volume"] * k)))


def test_random_walks_reach_both_valid_and_invalid_patterns():
    rs = [_ungated(random_walk(s)) for s in range(300)]
    assert sum(r["valid_vcp"] for r in rs) >= 3
    assert sum(r["vcp_score"] > 0 and not r["valid_vcp"] for r in rs) >= 20


def test_input_frame_is_not_modified():
    df = clean_vcp()
    before = df.copy(deep=True)
    score_ticker(df, RS)
    pd.testing.assert_frame_equal(df, before)


def test_keeps_shrinking():
    assert keeps_shrinking([3, 2, 1]) and not keeps_shrinking([3, 3, 1]) and keeps_shrinking([1])
    assert not keeps_shrinking([3, 2, 1.8]) and keeps_shrinking([3, 2, 1.8], replace(DEFAULT, max_shrink_ratio=0.95))
