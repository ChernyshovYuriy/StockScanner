"""
Barometer Phase 0 -- the stock-day panel (research/barometer/dataset.py).

The panel is only worth modelling if (a) no feature sees the future, (b) no
label sees the past beyond its entry close, and (c) the numbers are the
numbers the rest of the repo means: the stop is auto_pipeline's own stop,
the ATR is auto_pipeline's own ATR. Synthetic data only -- no network, no DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_pipeline import PipelineConfig, _atr, compute_levels
from research.barometer import dataset as ds
from research.barometer.dataset import (
    FEATURE_COLS, LABEL_COLS, MARKET_COLS, align_to_calendar, build_panel,
    build_ticker_rows, compute_features, compute_labels, compute_market_features,
    compute_stop, days_to_earnings,
)

N = 420
CAL = pd.bdate_range("2022-01-03", periods=N)


def _ohlcv(seed: int, start: float = 20.0, vol: float = 0.02, volume: float = 200_000,
           index: pd.DatetimeIndex = CAL) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0003, vol, len(index))))
    open_ = close * np.exp(rng.normal(0, vol / 3, len(index)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, len(index))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, len(index))))
    v = volume * np.exp(rng.normal(0, 0.3, len(index)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": v},
                        index=index)


@pytest.fixture
def bench():
    return _ohlcv(0, start=30.0, vol=0.008, volume=5_000_000)


@pytest.fixture
def data(bench):
    return {"XIU.TO": bench, "AAA.TO": _ohlcv(1), "BBB.TO": _ohlcv(2, start=8.0, vol=0.03),
            "CCC.V": _ohlcv(3, start=5.0, vol=0.04, volume=400_000)}


def _scramble_after(df: pd.DataFrame, t: pd.Timestamp, seed: int = 99) -> pd.DataFrame:
    """Same history through t, wildly different bars after it."""
    out = df.copy()
    after = out.index > t
    rng = np.random.default_rng(seed)
    out.loc[after, ["Open", "High", "Low", "Close"]] *= rng.uniform(0.3, 3.0, (after.sum(), 1))
    out.loc[after, "High"] = out.loc[after, ["Open", "High", "Low", "Close"]].max(axis=1) * 1.01
    out.loc[after, "Low"] = out.loc[after, ["Open", "High", "Low", "Close"]].min(axis=1) * 0.99
    out.loc[after, "Volume"] *= rng.uniform(0.01, 50, after.sum())
    return out


# ── (a) no feature sees the future ───────────────────────────────────────────

@pytest.mark.parametrize("cut_pos", [260, 330, 400])
def test_features_unchanged_when_future_bars_change(data, cut_pos):
    t = CAL[cut_pos]
    full = build_panel(data)
    scrambled = build_panel({k: _scramble_after(v, t) for k, v in data.items()})
    a = full[full.index.get_level_values("date") <= t]
    b = scrambled[scrambled.index.get_level_values("date") <= t]
    assert len(a) > 0
    assert a.index.equals(b.index)  # liquidity/completeness filters are past-only too
    pd.testing.assert_frame_equal(a[FEATURE_COLS + MARKET_COLS], b[FEATURE_COLS + MARKET_COLS])


def test_features_equal_on_truncated_history(data):
    t = CAL[350]
    full = build_panel(data)
    trunc = build_panel({k: v.loc[:t] for k, v in data.items()})
    a = full[full.index.get_level_values("date") <= t]
    assert a.index.equals(trunc.index)
    pd.testing.assert_frame_equal(a[FEATURE_COLS + MARKET_COLS], trunc[FEATURE_COLS + MARKET_COLS])


# ── (b) labels use t+1..t+H only ─────────────────────────────────────────────

def test_labels_ignore_bars_after_horizon(data):
    t_pos, h = 300, ds.HORIZON
    t = CAL[t_pos]
    full = build_panel(data)
    scrambled = build_panel({k: _scramble_after(v, CAL[t_pos + h]) for k, v in data.items()})
    a = full.xs(t, level="date")[LABEL_COLS]
    b = scrambled.xs(t, level="date")[LABEL_COLS]
    pd.testing.assert_frame_equal(a, b)


def test_labels_do_change_when_window_changes(data):
    """Sanity: the label really reads the future window (not a constant)."""
    t = CAL[300]
    full = build_panel(data)
    scrambled = build_panel({k: _scramble_after(v, t) for k, v in data.items()})
    assert not np.allclose(full.xs(t, level="date")["fwd_ret"], scrambled.xs(t, level="date")["fwd_ret"])


def test_last_horizon_rows_are_unlabelled(data):
    panel = build_panel(data)
    last = panel[panel.index.get_level_values("date") > CAL[-ds.HORIZON - 1]]
    assert len(last) > 0 and last[LABEL_COLS].isna().all().all()
    prior = panel[panel.index.get_level_values("date") == CAL[-ds.HORIZON - 1]]
    assert prior["y_stop_hit"].notna().all()


# ── (c) the numbers mean what the repo means ─────────────────────────────────

@pytest.mark.parametrize("pos", [30, 200, 419])
def test_stop_matches_compute_levels(data, pos):
    df = data["BBB.TO"]
    cfg = PipelineConfig()
    stop = compute_stop(align_to_calendar(df, CAL))
    c, h, l = df["Close"].iloc[:pos + 1], df["High"].iloc[:pos + 1], df["Low"].iloc[:pos + 1]
    live = compute_levels(c, h, l, float(c.iloc[-1]), cfg.atr_period, cfg.atr_stop_mult, cfg.max_stop_pct)
    assert stop.iloc[pos] == pytest.approx(live["stop"], abs=0.005 + 1e-9)


def test_stop_cap_binds_on_wide_atr():
    df = _ohlcv(5, vol=0.08)
    stop = compute_stop(align_to_calendar(df, CAL))
    floor = df["Close"] * (1 - PipelineConfig().max_stop_pct / 100)
    assert (stop.dropna() >= floor[stop.notna()] - 1e-12).all()
    assert np.isclose(stop, floor).sum() > N / 2  # an 8%-vol name mostly sits on the 7% cap


def test_atr_is_the_pipeline_atr(data):
    df = data["AAA.TO"]
    f = compute_features(align_to_calendar(df, CAL), data["XIU.TO"]["Close"])
    expected = _atr(df["High"], df["Low"], df["Close"], 14) / df["Close"]
    pd.testing.assert_series_equal(f["atr_pct"], expected, check_names=False)


def test_golden_values_on_a_known_series():
    close = pd.Series(np.arange(1, N + 1, dtype=float) + 10.0, index=CAL)  # 11, 12, 13, ...
    df = pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5,
                       "Close": close, "Volume": 1_000_000.0}, index=CAL)
    bench = close * 2
    f = compute_features(align_to_calendar(df, CAL), bench)
    t = 300
    c = close.iloc
    assert f["ret_5"].iloc[t] == pytest.approx(c[t] / c[t - 5] - 1)
    assert f["ret_63"].iloc[t] == pytest.approx(c[t] / c[t - 63] - 1)
    assert f["mom_12_1"].iloc[t] == pytest.approx(c[t - 21] / c[t - 252] - 1)
    assert f["dist_52w_high"].iloc[t] == pytest.approx(c[t] / (c[t] + 0.5) - 1)  # today's high is the max
    assert f["dist_52w_low"].iloc[t] == pytest.approx(c[t] / (c[t - 251] - 0.5) - 1)
    assert f["vol_ratio_20"].iloc[t] == pytest.approx(1.0)
    assert f["gap_abs_20"].iloc[t] == pytest.approx(np.mean([1 / c[i - 1] for i in range(t - 19, t + 1)]))
    assert f["beta_60"].iloc[t] == pytest.approx(
        pd.Series(np.log(close).diff()).iloc[t - 59:t + 1].cov(np.log(bench).diff().iloc[t - 59:t + 1])
        / np.log(bench).diff().iloc[t - 59:t + 1].var())
    lab = compute_labels(align_to_calendar(df, CAL), bench, compute_stop(align_to_calendar(df, CAL)), 20)
    assert lab["fwd_ret"].iloc[t] == pytest.approx(c[t + 20] / c[t] - 1)
    assert lab["y_stop_hit"].iloc[t] == 0.0          # a straight uptrend never hits its stop
    assert lab["y_beat_mkt"].iloc[t] == 0.0          # equal returns: bench = 2x the same path


def test_stop_hit_and_threshold_labels():
    close = pd.Series(100.0, index=CAL)
    close.iloc[311] = 80.0                             # one close far below any stop, inside t=300's window
    close.iloc[320] = 111.0                            # t+20 close for t=300: +11%
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                       "Volume": 1e6}, index=CAL)
    px = align_to_calendar(df, CAL)
    lab = compute_labels(px, pd.Series(50.0, index=CAL), compute_stop(px), 20)
    assert lab["y_stop_hit"].iloc[300] == 1.0
    assert lab["y_up10"].iloc[300] == 1.0 and lab["y_dn10"].iloc[300] == 0.0
    assert lab["y_beat_mkt"].iloc[300] == 1.0
    assert lab["y_stop_hit"].iloc[260] == 0.0          # window 261..280 is flat
    assert lab["y_stop_hit"].iloc[311] == 0.0          # the dip itself is the entry, not in its own window


def test_beta_recovers_a_known_multiple(bench):
    stock = bench.copy()
    r = np.log(bench["Close"]).diff().fillna(0)
    stock["Close"] = 10 * np.exp((2 * r).cumsum())
    stock[["Open", "High", "Low"]] = np.c_[stock["Close"], stock["Close"] * 1.01, stock["Close"] * 0.99]
    f = compute_features(align_to_calendar(stock, CAL), bench["Close"])
    assert f["beta_60"].iloc[100:].dropna().round(6).eq(2.0).all()


# ── liquidity, calendar, market, earnings ───────────────────────────────────

def test_liquidity_filters(bench):
    cheap = _ohlcv(7, start=1.0, vol=0.01)                 # below $2 throughout
    thin = _ohlcv(8, start=20.0, vol=0.01, volume=1_000)   # ~$20k/day
    panel = build_panel({"XIU.TO": bench, "CHEAP.V": cheap, "THIN.TO": thin, "OK.TO": _ohlcv(9)})
    tickers = set(panel.index.get_level_values("ticker"))
    assert tickers == {"OK.TO"}
    assert "XIU.TO" not in tickers


def test_liquidity_uses_past_volume_only(bench):
    """Volume jumps on session 300: no row may appear before it (that session's
    own volume is known at its close, so 300 itself may qualify)."""
    df = _ohlcv(10, volume=1_000)
    df.loc[CAL[300]:, "Volume"] = 10_000_000
    rows = build_ticker_rows("X.TO", df, bench["Close"])
    first = rows.index.min()
    assert first >= CAL[300] and CAL[299] not in rows.index
    assert (df["Close"] * df["Volume"]).rolling(20).mean().loc[first] >= ds.MIN_DOLLAR_VOL_20


def test_calendar_alignment_bridges_short_gaps_only():
    df = _ohlcv(11)
    short = df.drop(CAL[100:103])          # 3 missing sessions: bridged
    long_ = df.drop(CAL[200:208])          # 8 missing sessions: not bridged past 5
    a = align_to_calendar(short, CAL)
    assert a["Close"].iloc[100:103].eq(df["Close"].iloc[99]).all()
    assert a["Volume"].iloc[100:103].eq(0).all() and not a["traded"].iloc[100:103].any()
    b = align_to_calendar(long_, CAL)
    assert b["Close"].iloc[200:205].notna().all() and b["Close"].iloc[205:208].isna().all()


def test_halt_drops_rows_and_crossing_labels(bench):
    """10-session halt at 300..309: the first MAX_FFILL_SESSIONS (5) are bridged
    as flat bars, 305..309 stay NaN. A label whose window t+1..t+20 reaches
    the unbridged part (t >= 285) must be NaN; one that ends inside the
    bridge (t <= 284) is still labelled."""
    df = _ohlcv(12).drop(CAL[300:310])
    rows = build_ticker_rows("H.TO", df, bench["Close"])
    dates = rows.index
    assert not any(CAL[300] <= d < CAL[310] for d in dates)       # not traded -> no rows
    assert rows.loc[(dates >= CAL[285]) & (dates < CAL[300]), "y_stop_hit"].isna().all()
    assert rows.loc[(dates >= CAL[265]) & (dates < CAL[285]), "y_stop_hit"].notna().all()


def test_market_features_identical_across_tickers(data):
    panel = build_panel(data)
    spread = panel.groupby(level="date")[MARKET_COLS].nunique(dropna=False).max()
    assert (spread <= 1).all()
    m = compute_market_features(data["XIU.TO"]["Close"])
    assert m["mkt_above_200"].iloc[:199].isna().all() and m["mkt_above_200"].iloc[199:].notna().all()


def test_days_to_earnings_counts_sessions_to_next_report():
    cal = pd.bdate_range("2024-01-01", periods=60)
    reps = [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-13"),  # 13th is a Saturday
            pd.Timestamp("2024-03-15")]
    s = days_to_earnings(cal, reps, clip=30)
    assert s[pd.Timestamp("2024-01-08")] == 2                    # Mon -> Wed
    assert s[pd.Timestamp("2024-01-10")] == 3                    # report day itself -> next one (Mon 15th)
    assert s[pd.Timestamp("2024-01-12")] == 1                    # weekend report lands on Monday
    assert s[pd.Timestamp("2024-01-15")] == 30                   # March 15 is >30 sessions out: clipped
    assert np.isnan(s[cal[-1]]) or s[cal[-1]] <= 30
    assert days_to_earnings(cal, []).isna().all()


def test_panel_shape_and_columns(data):
    panel = build_panel(data)
    assert list(panel.index.names) == ["date", "ticker"]
    assert panel.index.is_unique and panel.index.is_monotonic_increasing
    for col in FEATURE_COLS + MARKET_COLS + LABEL_COLS + ["days_to_earnings", "close"]:
        assert col in panel.columns
    assert panel[FEATURE_COLS].notna().all().all()
    assert set(panel["y_stop_hit"].dropna().unique()) <= {0.0, 1.0}


@pytest.mark.parametrize("dip_offset, expected", [(20, 1.0), (21, 0.0), (1, 1.0)])
def test_label_window_edges_are_exactly_t_plus_1_to_t_plus_h(dip_offset, expected):
    """A dip on t+H is inside t's window, on t+H+1 it isn't; fwd_rvol is the
    std of exactly the H log returns ending t+1..t+H (not t's own return)."""
    t, h = 300, 20
    rng = np.random.default_rng(13)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, N))), index=CAL)
    close.iloc[t] *= 1.5                          # big return INTO t: must not leak into fwd_rvol
    close.iloc[t + 1:] *= 1.5
    close.iloc[t + dip_offset] *= 0.7
    df = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999,
                       "Close": close, "Volume": 1e6}, index=CAL)
    px = align_to_calendar(df, CAL)
    lab = compute_labels(px, close, compute_stop(px), h)
    assert lab["y_stop_hit"].iloc[t] == expected
    lr = np.log(close).diff()
    assert lab["fwd_rvol"].iloc[t] == pytest.approx(lr.iloc[t + 1:t + h + 1].std())
