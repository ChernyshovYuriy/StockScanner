"""VCP study Phase 5 -- the backtest. Synthetic markets only."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.vcp import backtest_vcp as bt, vcp
from research.vcp.backtest_vcp import (
    BLOCK_WEEKS, HORIZONS, bar_at, block_bootstrap, breakout, build_market, build_panel,
    entry_bar, fix_split_errors, forward, group_masks, high_score_events, outcomes, same_date_mean, score_panel,
    scoring_sessions, sensitivity_configs, separate_events, stage2_flags, summary_table, verdict,
    weekly_sessions, with_excess,
)
from research.vcp.prices import BENCHMARK
from research.vcp.vcp import DEFAULT, Bars


def bars(close, open_=None, high=None, low=None, volume=None, session=None):
    c = np.asarray(close, float)
    pick = lambda x: c if x is None else np.asarray(x, float)
    s = np.arange(len(c)) if session is None else np.asarray(session)
    return Bars(pd.bdate_range("2020-01-01", periods=len(c)).to_numpy(), pick(open_), pick(high), pick(low), c,
                np.full(len(c), 100.0) if volume is None else np.asarray(volume, float), s)


# ── entries and exits, by hand ───────────────────────────────────────────────

def test_entry_is_the_next_trade():
    assert entry_bar(bars(range(1, 11)), 4) == 5


def test_entry_waits_at_most_five_sessions():
    b = bars(range(1, 6), session=[0, 1, 2, 7, 8])
    assert entry_bar(b, 2) == 3                         # session 7, five later
    assert entry_bar(bars(range(1, 6), session=[0, 1, 2, 8, 9]), 2) == -1
    assert entry_bar(b, 8) == -1


def test_forward_open_to_close():
    c = np.arange(100.0, 140.0)
    b = bars(c, open_=c - 1, low=c - 2)
    r, worst, ended = forward(b, 10, 20, 40)
    assert r == pytest.approx(130 / 109 - 1) and worst == pytest.approx(108 / 109 - 1) and not ended


def test_forward_worst_low_stops_at_the_exit():
    c = np.full(40, 100.0)
    low = c.copy(); low[9] = 1; low[10] = 95; low[30] = 90; low[31] = 1
    assert forward(bars(c, low=low), 10, 20, 40)[1] == pytest.approx(-0.10)


def test_forward_needs_the_exit_session_to_exist():
    b = bars(np.full(40, 100.0))
    assert np.isfinite(forward(b, 10, 29, 40)[0])
    assert np.isnan(forward(b, 10, 30, 40)[0])


def test_forward_uses_last_trade_when_halted_at_exit():
    b = bars([100, 110, 120, 130], session=[0, 1, 2, 9])
    r, _, ended = forward(b, 0, 5, 50)
    assert r == pytest.approx(0.2) and not ended        # still trades later


def test_forward_long_halt_is_not_dead():
    assert not forward(bars([100, 110, 120, 130], session=[0, 1, 2, 20]), 0, 10, 50)[2]


def test_forward_closes_a_dead_name_at_its_last_trade():
    b = bars([100, 90, 50], session=[0, 1, 2])
    r, worst, ended = forward(b, 0, 60, 500)
    assert r == pytest.approx(-0.5) and worst == pytest.approx(-0.5) and ended
    assert not forward(b, 0, 7, 500)[2]                 # five sessions of silence isn't dead yet


def test_bar_at():
    b = bars([1, 2, 3], session=[0, 1, 5])
    assert [bar_at(b, s, 2) for s in (0, 1, 3, 4, 5, 7, 8)] == [0, 1, 1, -1, 2, 2, -1]
    assert bar_at(bars([1], session=[4]), 2, 5) == -1


# ── breakouts ────────────────────────────────────────────────────────────────

def _bo(close, volume=None, low=None, i=50, pivot=10.0, n=200, e=51):
    return breakout(bars(close, volume=volume, low=low), i, e, i, pivot, n)


def test_breakout_needs_price_and_volume():
    c = np.full(100, 9.0); c[60] = 10.5
    v = np.full(100, 100.0)
    assert _bo(c, v)[0] == 0.0
    v[60] = 140.0
    assert _bo(c, v)[0] == 1.0
    v[60] = 139.0
    assert _bo(c, v)[0] == 0.0


def test_breakout_window_is_30_sessions():
    v = np.full(100, 100.0)
    for k, want in ((80, 1.0), (81, 0.0)):
        c = np.full(100, 9.0); c[k] = 11
        w = v.copy(); w[k] = 500
        assert _bo(c, w)[0] == want


def test_breakout_drawdown_stops_at_the_breakout():
    c = np.full(100, 9.0); c[60] = 11
    v = np.full(100, 100.0); v[60] = 500
    low = np.full(100, 9.0); low[50] = 1; low[55] = 8.1; low[61] = 2
    assert _bo(c, v, low)[1] == pytest.approx(-0.10)
    assert _bo(np.full(100, 9.0), v, low)[1] == pytest.approx(2 / 9 - 1)


def test_breakout_unknowable_cases_are_nan():
    c = np.full(100, 9.0)
    assert np.isnan(_bo(c, n=80)[0]) and np.isfinite(_bo(c, n=81)[0])
    assert np.isnan(_bo(c, pivot=np.nan)[0]) and np.isnan(_bo(c, e=-1)[0])
    c[50] = 10.01
    assert np.isnan(_bo(c)[0])                          # already above


def test_breakout_volume_average_excludes_the_day_itself():
    c = np.full(100, 9.0); c[60] = 11
    v = np.full(100, 100.0); v[60] = 141.0              # would fail if its own 141 were averaged in
    assert _bo(c, v)[0] == 1.0
    v[60] = 140.5
    assert 140.5 < 1.4 * np.r_[v[11:60], 140.5].mean() and _bo(c, v)[0] == 1.0


# ── unadjusted splits ────────────────────────────────────────────────────────

def _split(jump=10.0, vol_after=2_000.0, vol=100_000.0, price=2.0, k=30, n=60):
    c = np.full(n, price); c[k:] *= jump
    v = np.full(n, vol); v[k:] = vol_after
    return bars(c, open_=c * 0.99, high=c * 1.01, low=c * 0.98, volume=v)


def test_unadjusted_reverse_split_is_rescaled():
    raw = _split()
    b = fix_split_errors(raw)
    for f in ("open", "high", "low", "close"):
        np.testing.assert_allclose(getattr(b, f)[:30], getattr(raw, f)[:30] * 10)
        np.testing.assert_array_equal(getattr(b, f)[30:], getattr(raw, f)[30:])
    np.testing.assert_allclose(b.volume[:30], 10_000)
    np.testing.assert_allclose(b.close[:30] * b.volume[:30], raw.close[:30] * raw.volume[:30])
    np.testing.assert_array_equal(b.volume[30:], raw.volume[30:])


def test_two_split_errors_compound():
    c = np.r_[np.full(30, 2.0), np.full(30, 8.0), np.full(30, 24.0)]
    v = np.r_[np.full(30, 1e6), np.full(30, 2e5), np.full(30, 5e4)]
    np.testing.assert_allclose(fix_split_errors(bars(c, volume=v)).close, 24.0)


@pytest.mark.parametrize("kw", [dict(vol_after=60_000.0),            # a real jump comes with volume
                                dict(jump=1.99), dict(vol=40_000.0, vol_after=100.0)])   # small; $80k a day
def test_real_jumps_are_left_alone(kw):
    raw = _split(**kw)
    assert fix_split_errors(raw) is raw


def test_split_thresholds_are_exact():
    assert fix_split_errors(_split(vol_after=50_000.0)).close[0] == 2.0
    assert fix_split_errors(_split(vol_after=49_999.0)).close[0] == 20.0
    assert fix_split_errors(_split(jump=2.0)).close[0] == 4.0
    assert fix_split_errors(_split(vol=50_000.0)).close[0] == 20.0
    assert fix_split_errors(_split(vol=49_999.0)).close[0] == 2.0


def test_split_usual_volume_is_the_20_days_before():
    raw = _split(vol_after=40_000.0)
    v = raw.volume.copy(); v[:10] = 1.0                  # older than 20 days: ignored
    assert fix_split_errors(replace(raw, volume=v)).close[0] == 20.0
    v = raw.volume.copy(); v[10:30] = 70_000.0           # the last 20 days trade like the day itself
    assert fix_split_errors(replace(raw, volume=v)).close[0] == 2.0


def test_crash_is_never_rescaled():
    assert fix_split_errors(_split(jump=0.1)).close[0] == 2.0


def test_market_build_applies_the_fix(small):
    data = dict(small[0])
    d = data["T0.TO"].copy()
    d.iloc[:400, :4] /= 10
    d.iloc[:400, 4] *= 10
    data["T0.TO"] = d
    d.iloc[400, 4] = VOL / 10
    np.testing.assert_allclose(build_market(data).bars["T0.TO"].close[400:], small[0]["T0.TO"]["Close"][400:])
    r = np.diff(np.log(build_market(data).bars["T0.TO"].close))
    assert np.abs(r).max() < 0.2


# ── dates ────────────────────────────────────────────────────────────────────

def test_weekly_sessions_are_last_of_each_week():
    cal = pd.bdate_range("2024-01-01", "2024-01-31").delete([4])        # Fri Jan 5 closed
    got = [cal[s].strftime("%m-%d") for s in weekly_sessions(cal, 0)]
    assert got == ["01-04", "01-12", "01-19", "01-26", "01-31"]
    assert [cal[s].strftime("%m-%d") for s in weekly_sessions(cal, 10)] == ["01-19", "01-26", "01-31"]
    assert weekly_sessions(cal, 13)[0] == 13 and cal[13] == pd.Timestamp("2024-01-19")


# ── grouping and statistics ──────────────────────────────────────────────────

def _panel(scores, gate=None, valid=None):
    n = len(scores)
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "session": 0, "vcp_score": np.array(scores, float),
                         "gate_passed": [True] * n if gate is None else gate,
                         "valid_vcp": [s > 40 for s in scores] if valid is None else valid})


def test_buckets_partition_every_row():
    p = _panel([0, 0, 0.4, 0.5, 40.4, 40.5, 70.4, 70.5, 85.4, 85.5, 100], gate=[False] + [True] * 10)
    m = group_masks(p)
    parts = ["0 (gate failed)", "0 (gate passed)", "1-40", "41-70", "71-85", "86-100"]
    assert (sum(m[k].astype(int) for k in parts) == 1).all()
    assert [int(m[k].sum()) for k in parts] == [1, 2, 2, 2, 2, 2]
    assert (m["71+"] == (m["71-85"] | m["86-100"])).all()
    assert m["all liquid"].all()


def test_gate_only_is_gate_passed_without_valid_vcp():
    p = _panel([0, 0, 30, 80], gate=[False, True, True, True], valid=[False, False, False, True])
    m = group_masks(p)
    assert m["gate-only"].tolist() == [False, True, True, False]
    assert m["all gate passers"].tolist() == [False, True, True, True]


def test_separate_events_cooldown():
    rows = pd.DataFrame({"ticker": ["A"] * 5 + ["B"], "session": [0, 30, 59, 60, 125, 10]})
    ev = separate_events(rows)
    assert list(zip(ev["ticker"], ev["session"])) == [("A", 0), ("A", 60), ("A", 125), ("B", 10)]


def test_separate_events_counts_from_the_last_kept():
    rows = pd.DataFrame({"ticker": "A", "session": [0, 40, 80, 120]}, index=[9, 7, 5, 3])
    assert sorted(separate_events(rows)["session"]) == [0, 80]


def test_same_date_mean():
    p = pd.DataFrame({"session": [1, 1, 1, 2, 3], "x": [1.0, 3.0, 99.0, 5.0, 7.0]})
    got = same_date_mean(p, pd.Series([True, True, False, True, False]), "x")
    assert got[:4].tolist() == [2, 2, 2, 5] and np.isnan(got[4])


def test_bootstrap_resamples_blocks_not_rows():
    """All rows of a block identical: the interval must be as wide as blocks allow,
    however many rows each has."""
    weeks = np.repeat(np.arange(8 * BLOCK_WEEKS), 50)
    vals = np.where((weeks // BLOCK_WEEKS) % 2, 1.0, -1.0)
    lo, hi = block_bootstrap(vals, weeks, {int(w): int(w) for w in weeks})
    assert lo < -0.4 and hi > 0.4


def test_bootstrap_covers_about_95_percent():
    rng = np.random.default_rng(0)
    hits = 0
    for k in range(200):
        weeks = np.repeat(np.arange(30 * BLOCK_WEEKS), 2)
        vals = rng.normal(0.1, 1, len(weeks)) + rng.normal(0, 0.5, 30)[weeks // BLOCK_WEEKS]
        lo, hi = block_bootstrap(vals, weeks, {int(w): int(w) for w in weeks}, n_boot=400, seed=k)
        hits += lo <= 0.1 <= hi
    assert 0.88 <= hits / 200 <= 0.98


def test_bootstrap_ignores_nan_and_needs_two_blocks():
    assert np.isnan(block_bootstrap(np.ones(5), np.zeros(5), {0: 0})[0])
    s = np.array([0, 0, 20, 20]); w = {0: 0, 20: 20}
    assert block_bootstrap(np.array([1, np.nan, 1, 1.0]), s, w) == (1.0, 1.0)


# ── verdict ──────────────────────────────────────────────────────────────────

def _verdict_panel(n_tickers=60, per=8, excess=0.05, noise=0.02, seed=0, year_flip=()):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tickers):
        for k in range(per):
            d = pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=26 * k + t % 26)
            x = -excess if d.year in year_flip else excess
            rows.append((f"H{t}", (26 * k + t % 26) * 5, d, 80.0, True, x + rng.normal(0, noise)))
            rows.append((f"G{t}", (26 * k + t % 26) * 5, d, 0.0, False, 0.0))
    p = pd.DataFrame(rows, columns=["ticker", "session", "date", "vcp_score", "valid_vcp", "ret_60"])
    p["gate_passed"] = True
    for h in HORIZONS:
        p[f"ret_{h}"], p[f"xiu_{h}"] = p["ret_60"], 0.0
    return with_excess(p)


WEEKS = {s: s // 5 for s in range(0, 3000, 5)}


def test_verdict_passes_a_real_edge():
    v = verdict(_verdict_panel(), WEEKS, [0.01] * 9 + [-0.01])
    assert v["passed"] and v["mean_excess"] == pytest.approx(0.05, abs=0.005) and v["events"] == 480


def test_verdict_without_sensitivity_never_passes():
    assert "passed" not in verdict(_verdict_panel(), WEEKS)


@pytest.mark.parametrize("kw,sens,failed", [
    (dict(excess=0.0), [0.1] * 10, "c1_positive"),
    (dict(excess=-0.05), [0.1] * 10, "c1_positive"),
    (dict(n_tickers=49, per=8), [0.1] * 10, "c2_sample"),
    (dict(n_tickers=60, per=1), [0.1] * 10, "c2_sample"),
    (dict(excess=0.05, year_flip=(2021, 2023)), [0.1] * 10, "c3_years"),
    (dict(), [0.1] * 7 + [-0.1] * 3, "c4_sensitivity"),
])
def test_verdict_fails_each_condition(kw, sens, failed):
    v = verdict(_verdict_panel(**kw), WEEKS, sens)
    assert not v[failed] and not v["passed"]


def test_verdict_uses_excess_not_raw_return():
    """Everything rises 10%, high scores included: no edge."""
    p = _verdict_panel(excess=0.0)
    for h in HORIZONS:
        p[f"ret_{h}"] += 0.10
    assert not verdict(with_excess(p), WEEKS, [0.1])["c1_positive"]


def test_high_score_events_are_separate():
    p = _verdict_panel(n_tickers=1, per=1)
    more = [p.assign(session=p["session"] + k) for k in (5, 59, 60)]
    ev = high_score_events(with_excess(pd.concat([p, *more], ignore_index=True)))
    assert (ev["session"] - ev["session"].min()).tolist() == [0, 60]


def test_events_without_a_same_date_baseline_are_left_out():
    p = _verdict_panel(n_tickers=3, per=2)
    p.loc[p["session"] == p["session"].min(), "ret_60"] = np.where(
        p.loc[p["session"] == p["session"].min(), "ticker"].str.startswith("G"), np.nan, 0.9)
    ev = high_score_events(with_excess(p))
    assert p["session"].min() not in set(ev["session"])


# ── sensitivity configs ──────────────────────────────────────────────────────

def test_sensitivity_configs():
    c = sensitivity_configs()
    assert len(c) == 2 * (len(bt.SENSITIVITY_GROUPS) + 5)
    assert all(v != DEFAULT for v in c.values())
    assert c["zigzag_pct -20%"].zigzag_pct == pytest.approx(0.024)
    assert c["min_base_days +20%"].min_base_days == 18 and isinstance(c["vol_avg_days +20%"].vol_avg_days, int)
    assert c["max_shrink_ratio +20%"].max_shrink_ratio == 1.0
    assert c["tightness +20%"].tight_zero == pytest.approx(0.24)
    for v in c.values():
        assert sum(getattr(v, w) for w in bt.WEIGHTS) == pytest.approx(100.0)
    changed = [k for k, v in c.items() if k.startswith("zigzag")]
    assert replace(c[changed[0]], zigzag_pct=DEFAULT.zigzag_pct) == DEFAULT


# ── whole pipeline on a synthetic market ─────────────────────────────────────

N_DAYS = 900
CAL = pd.bdate_range("2020-01-01", periods=N_DAYS)
VOL = 2_000_000.0


def _frame(c, rng, volume=None):
    c = np.asarray(c, float)
    spread = np.abs(rng.normal(0, 0.002, len(c))) + 0.001
    o = c * (1 + rng.normal(0, 0.0008, len(c)))         # never equal to the close
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + spread),
                         "Low": np.minimum(o, c) * (1 - spread), "Close": c,
                         "Volume": np.full(len(c), VOL) if volume is None else volume}, index=CAL[:len(c)])


def _vcp_stock(rng, pop):
    """Uptrend, 3-contraction base, then a move of `pop` over 60 days, repeated."""
    c = list(10 * np.exp(np.cumsum(rng.normal(0.004, 0.006, 290))))
    v = [VOL] * 290
    while len(c) < N_DAYS:
        top = c[-1]
        for target, n in ((0.80, 18), (0.98, 16), (0.885, 10), (0.965, 10), (0.93, 6), (0.955, 4)):
            c += list(np.linspace(c[-1], top * target, n + 1)[1:] * np.exp(rng.normal(0, 0.002, n)))
            v += [VOL * (0.35 if target in (0.93, 0.955) else 1.0)] * n
        c += list(c[-1] * np.exp(np.cumsum(np.full(60, np.log1p(pop) / 60) + rng.normal(0, 0.004, 60))))
        v += [VOL * 2] * 12 + [VOL] * 48
        c += list(c[-1] * np.exp(np.cumsum(rng.normal(0.003, 0.006, 60))))
        v += [VOL] * 60
    return _frame(c[:N_DAYS], rng, np.array(v[:N_DAYS]))


def synthetic_market(pop, n_vcp=25, n_trend=25, n_noise=25, seed=0):
    rng = np.random.default_rng(seed)
    data = {BENCHMARK: _frame(30 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, N_DAYS))), rng)}
    for k in range(n_vcp):
        data[f"V{k}.TO"] = _vcp_stock(rng, pop)
    for k in range(n_trend):                             # gate passers without a pattern
        data[f"T{k}.TO"] = _frame(10 * np.exp(np.cumsum(rng.normal(0.0022, 0.012, N_DAYS))), rng)
    for k in range(n_noise):
        data[f"N{k}.TO"] = _frame(10 * np.exp(np.cumsum(rng.normal(-0.0003, 0.02, N_DAYS))), rng)
    return data


@pytest.fixture(scope="module")
def small():
    """Six names: enough for exact checks."""
    data = synthetic_market(0.3, 3, 2, 1, seed=3)
    m = build_market(data)
    sessions, week_of = scoring_sessions(m)
    return data, m, sessions, week_of, build_panel(m, sessions)


def test_planted_edge_is_found():
    m = build_market(synthetic_market(0.45))
    sessions, week_of = scoring_sessions(m)
    p = build_panel(m, sessions)
    ev = high_score_events(p)
    assert (ev["ticker"].str.startswith("V")).mean() > 0.6
    v = verdict(p, week_of, [1.0])
    assert v["c1_positive"] and v["mean_excess"] > 0.08, v


def test_no_edge_is_not_found():
    """Independent random walks with one drift: whatever is scored, nothing follows."""
    rng = np.random.default_rng(11)
    hits, seen = 0, 0
    for rep in range(6):
        data = {BENCHMARK: _frame(30 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, N_DAYS))), rng)}
        for k in range(60):
            data[f"R{k}.TO"] = _frame(10 * np.exp(np.cumsum(rng.normal(0.0008, 0.016, N_DAYS))), rng)
        m = build_market(data)
        sessions, week_of = scoring_sessions(m)
        panel = score_panel(m, sessions)
        panel = with_excess(panel.merge(outcomes(m, panel[["ticker", "session"]]), on=["ticker", "session"]))
        v = verdict(panel, week_of, [1.0])
        seen += v["events"]
        hits += v["c1_positive"]
        assert abs(v["mean_excess"]) < 0.08
    assert seen > 100 and hits <= 1


def test_panel_rows_match_score_ticker(small):
    """The backtest's score is vcp.score_ticker() on the frame cut at that date."""
    data, m, sessions, _, p = small
    for _, r in p.iterrows():
        cut = data[r["ticker"]].loc[:r["date"]]
        s = vcp.score_ticker(cut, r["rs_percentile"], m.calendar[:r["session"] + 1])
        assert s["vcp_score"] == pytest.approx(r["vcp_score"]) and s["gate_passed"] == r["gate_passed"]
        assert (s["gate_fail_reason"] or "") == ("" if pd.isna(r["gate_fail_reason"]) else r["gate_fail_reason"])
        assert s["n_contractions"] == r["n_contractions"]


def test_rs_is_ranked_among_that_dates_liquid_names(small):
    _, m, _, _, p = small
    for s, g in p.groupby("session"):
        raw = {t: sum(w * (m.bars[t].close[s] / m.bars[t].close[s - d] - 1) for d, w in DEFAULT.rs_periods) for t in g["ticker"]}
        for t, got in zip(g["ticker"], g["rs_percentile"]):
            assert got == pytest.approx(100 * np.mean([v <= raw[t] for v in raw.values()]))


def test_truncated_data_gives_identical_past_rows(small):
    """No lookahead: rebuild on data cut at a date; every earlier row is the same."""
    data, m, sessions, _, p = small
    cut = CAL[620]
    m2 = build_market(data, cut=cut)
    s2, _ = scoring_sessions(m2)
    p2 = build_panel(m2, s2)
    cols = ["ticker", "session", "date", "vcp_score", "gate_passed", "gate_fail_reason", "rs_percentile",
            "pivot_price", "n_contractions", "stage2", "valid_vcp", "entry_bar"]
    last_full = 618                                      # the final session has no next open yet
    a = p[p["session"] < last_full][cols].reset_index(drop=True)
    b = p2[p2["session"] < last_full][cols].reset_index(drop=True)
    full_weeks = set(a["session"]) & set(b["session"])
    assert len(full_weeks) > 50
    pd.testing.assert_frame_equal(a[a["session"].isin(full_weeks)].reset_index(drop=True),
                                  b[b["session"].isin(full_weeks)].reset_index(drop=True))
    assert p2["date"].max() < cut
    # returns that would end past the cut are missing, the rest identical
    j = p.merge(p2, on=["ticker", "session"], suffixes=("", "_cut"))
    for h in HORIZONS:
        done = j["session"] + 1 + h < 620
        assert j.loc[~done, f"ret_{h}_cut"].isna().all()
        np.testing.assert_allclose(j.loc[done, f"ret_{h}"], j.loc[done, f"ret_{h}_cut"])


def test_scrambling_the_future_changes_no_score(small):
    data, m, sessions, _, p = small
    rng = np.random.default_rng(5)
    wild = {}
    for t, d in data.items():
        d = d.copy()
        k = 701
        f = np.exp(np.cumsum(rng.normal(0, 0.1, len(d) - k)))
        for c in ("Open", "High", "Low", "Close"):
            d.iloc[k:, d.columns.get_loc(c)] *= f
        wild[t] = d
    m2 = build_market(wild)
    q = score_panel(m2, [s for s in sessions if s <= 700])
    assert stage2_flags(m2, q).tolist() == p.loc[p["session"] <= 700, "stage2"].tolist()
    cols = ["ticker", "session", "vcp_score", "gate_fail_reason", "rs_percentile", "pivot_price"]
    pd.testing.assert_frame_equal(p[p["session"] <= 700][cols].reset_index(drop=True), q[cols])


def test_returns_by_hand(small):
    data, m, _, _, p = small
    xiu = data[BENCHMARK]
    for _, r in p.dropna(subset=["ret_120"]).iterrows():
        d, s = data[r["ticker"]], r["session"]
        for h in HORIZONS:
            assert r[f"ret_{h}"] == pytest.approx(d["Close"].iloc[s + 1 + h] / d["Open"].iloc[s + 1] - 1)
            assert r[f"xiu_{h}"] == pytest.approx(xiu["Close"].iloc[s + 1 + h] / xiu["Open"].iloc[s + 1] - 1)
            assert r[f"mae_{h}"] == pytest.approx(d["Low"].iloc[s + 1:s + 2 + h].min() / d["Open"].iloc[s + 1] - 1)
            assert r[f"ex_xiu_{h}"] == pytest.approx(r[f"ret_{h}"] - r[f"xiu_{h}"])


def test_excess_by_hand(small):
    p = small[4]
    for s, g in p.groupby("session"):
        base = g.loc[g["gate_passed"] & ~g["valid_vcp"], "ret_60"].mean()
        np.testing.assert_allclose(g["ex_gate_60"], g["ret_60"] - base)
        alls = g.loc[g["gate_passed"], "ret_60"].mean()
        np.testing.assert_allclose(g["ex_all_60"], g["ret_60"] - alls)
        st2 = g.loc[g["stage2"], "ret_60"].mean()
        np.testing.assert_allclose(g["ex_stage2_60"], g["ret_60"] - st2)


def test_stage2_matches_the_repo_function(small):
    from canadian_stock_screener import CONFIG, ScoreCalculator
    data, _, _, _, p = small
    calc = ScoreCalculator(CONFIG)
    for _, r in p.iterrows():
        want = calc.score_stage2(data[r["ticker"]]["Close"].loc[:r["date"]]) > 0
        assert r["stage2"] == want
    assert p["stage2"].nunique() == 2


def test_breakouts_only_for_gate_passers(small):
    p = small[4]
    assert p.loc[~p["gate_passed"], "broke_out"].isna().all()
    assert set(p["broke_out"].dropna()) == {0.0, 1.0}


def test_summary_table_by_hand(small):
    _, _, _, week_of, p = small
    t = summary_table(p, 60, week_of, separate=False).set_index("group")
    g = p[p["gate_passed"] & (np.floor(p["vcp_score"] + 0.5) >= 71) & p["ret_60"].notna()]
    row = t.loc["71+"]
    assert row["n"] == len(g) and row["tickers"] == g["ticker"].nunique()
    assert row["mean"] == pytest.approx(g["ret_60"].mean()) and row["median"] == pytest.approx(g["ret_60"].median())
    assert row["hit"] == pytest.approx((g["ret_60"] > 0).mean())
    assert row["mean_net"] == pytest.approx(g["ret_60"].mean() - 0.005)
    assert row["ex_gate"] == pytest.approx(g["ex_gate_60"].mean()) and row["ex_xiu"] == pytest.approx(g["ex_xiu_60"].mean())
    assert row["mae_median"] == pytest.approx(g["mae_60"].median()) and row["broke_out"] == pytest.approx(g["broke_out"].mean())
    e = summary_table(p, 60, week_of, separate=True).set_index("group")
    assert e.loc["71+", "n"] == len(separate_events(g)) < len(g)


def test_illiquid_and_short_history_names_never_appear(small):
    data = dict(small[0])
    rng = np.random.default_rng(1)
    data["THIN.TO"] = _frame(np.linspace(10, 30, N_DAYS), rng, np.full(N_DAYS, 1000.0))
    data["PENNY.TO"] = _frame(np.linspace(0.2, 0.9, N_DAYS), rng, np.full(N_DAYS, 1e8))
    data["NEW.TO"] = _frame(np.linspace(10, 30, N_DAYS), rng).iloc[-200:]
    m = build_market(data)
    p = score_panel(m, weekly_sessions(m.calendar, DEFAULT.year_days))
    assert set(p["ticker"]) == set(small[4]["ticker"])


def test_dead_name_leaves_the_panel_but_keeps_its_return(small):
    data = dict(small[0])
    t = "T0.TO"
    data[t] = data[t].iloc[:700]
    m = build_market(data)
    sessions, _ = scoring_sessions(m)
    p = score_panel(m, sessions)
    mine = p[p["ticker"] == t]
    assert mine["session"].max() <= 699 + DEFAULT.max_stale_sessions
    o = outcomes(m, mine[["ticker", "session"]]).set_index("session")
    s = max(x for x in mine["session"] if x < 690)
    c, op = data[t]["Close"], data[t]["Open"]
    assert o.loc[s, "ret_60"] == pytest.approx(c.iloc[-1] / op.iloc[s + 1] - 1) and o.loc[s, "ended_60"]


def test_market_input_is_not_modified():
    data = synthetic_market(0.3, 1, 1, 0)
    before = {t: d.copy(deep=True) for t, d in data.items()}
    m = build_market(data)
    build_panel(m, scoring_sessions(m)[0][::8])
    for t in data:
        pd.testing.assert_frame_equal(data[t], before[t])


# ── command line guards ──────────────────────────────────────────────────────

def test_holdout_refuses_without_a_dev_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "OUT_DIR", tmp_path)
    monkeypatch.setattr(bt.prices, "connect", lambda *a, **k: pytest.fail("opened the data"))
    assert bt.main(["--phase", "holdout"]) == 1


def test_dev_and_sensitivity_runs_never_see_holdout_data(tmp_path, monkeypatch):
    data = synthetic_market(0.3, 1, 1, 0)
    monkeypatch.setattr(bt, "OUT_DIR", tmp_path)
    monkeypatch.setattr(bt, "PANEL_DIR", tmp_path)
    monkeypatch.setattr(bt, "HOLDOUT", CAL[600])
    monkeypatch.setattr(bt.prices, "connect", lambda *a, **k: type("C", (), {"close": lambda self: None})())
    monkeypatch.setattr(bt.prices, "load_all", lambda conn: data)
    seen = []
    real = bt.score_panel
    monkeypatch.setattr(bt, "score_panel", lambda m, s, cfg=DEFAULT: (seen.append(m.calendar.max()), real(m, s, cfg))[1])
    monkeypatch.setattr(bt, "_report", lambda *a: None)
    assert bt.main(["--phase", "dev"]) == 0
    assert seen and max(seen) < CAL[600]
    (tmp_path / "dev_verdict.json").write_text("{}")
    seen.clear()
    assert bt.main(["--phase", "holdout"]) == 0
    assert max(seen) == CAL[-1]
    assert pd.read_pickle(tmp_path / "vcp_holdout_panel.pkl")["date"].min() >= CAL[600]
