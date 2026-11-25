"""
VCP (Volatility Contraction Pattern) score for one ticker's daily OHLCV, 0-100.

Pure functions, no I/O. score_ticker() is the entry point: it scores the LAST
bar of the frame it is given, so "as of a date" means "the frame cut at that
date". The backtest calls the same score_at() on the same arrays.

Steps: clean_bars() keeps real trades only -> trend-template gate
(gate_table) -> find_base -> find_contractions (causal zigzag,
wiggles merged) -> five component scores.

Every threshold and weight is in VcpConfig. Percentages are written as
fractions (0.25 = 25%) except in the returned dict, where *_pct fields are
percentages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from canadian_stock_screener import TechnicalIndicators
from swing_tickers import Thresholds

_LIQUIDITY = Thresholds()


@dataclass(frozen=True)
class VcpConfig:
    # ── Liquidity and data quality ──
    min_price: float = _LIQUIDITY.min_price
    min_dollar_vol: float = _LIQUIDITY.min_avg_dollar_vol_20
    dollar_vol_days: int = 20
    max_stale_sessions: int = 5        # last trade must be this recent
    max_missing_frac: float = 0.10     # untraded share of sessions, over 52 weeks and over the base

    # ── Trend template ──
    ma_fast: int = 50
    ma_mid: int = 150
    ma_slow: int = 200
    ma_slow_rising_days: int = 21
    year_days: int = 252
    min_above_low: float = 0.25        # at least this far above the 52-week low
    max_below_high: float = 0.25       # at most this far below the 52-week high
    min_rs_percentile: float = 70.0
    # (days, weight): the last quarter counts double, as in IBD's RS rating
    rs_periods: Tuple[Tuple[int, float], ...] = ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2))

    # ── Base and contractions ──
    min_base_days: int = 15            # 3 weeks
    max_base_days: int = 325           # 65 weeks
    zigzag_pct: float = 0.03           # reversal that makes a swing
    merge_swings: bool = True          # False: every swing is a contraction of its own
    max_shrink_ratio: float = 0.85     # each contraction at most this times the one before (as _detect_vcp)
    min_contractions: int = 2
    max_contractions: int = 6
    atr_days: int = 14
    scale_by_atr: bool = True          # widen the depth thresholds for volatile stocks
    atr_ref: float = 0.025             # ATR% at which nothing is widened
    max_atr_scale: float = 2.5

    # ── Component shapes ──
    full_shrink: float = 0.40          # average shrink per contraction worth full marks
    tight_full: float = 0.05           # final depth worth full marks
    tight_good: float = 0.10           # final depth worth tight_good_score
    tight_good_score: float = 0.6
    tight_zero: float = 0.20
    vol_avg_days: int = 50
    min_dryup_days: int = 5            # shortest stretch the final contraction's volume is read over
    dryup_full: float = 0.5
    dryup_zero: float = 1.0
    near_full: float = 0.03            # at most this far below the pivot: full marks
    near_zero: float = 0.15
    max_extended: float = 0.05         # beyond this above the pivot: 0
    extended_score: float = 0.5        # just above the pivot
    base_depth_full: float = 0.35
    base_depth_zero: float = 0.50
    uptrend_days: int = 126            # window before the base high
    uptrend_full: float = 0.30

    # ── Weights (total 100) ──
    w_progression: float = 30.0
    w_tightness: float = 20.0
    w_dryup: float = 20.0
    w_proximity: float = 20.0
    w_base: float = 10.0
    invalid_cap: float = 40.0          # highest score when the contractions don't keep shrinking


DEFAULT = VcpConfig()


@dataclass(frozen=True)
class Bars:
    """Real trading bars only, plus each bar's position in the session calendar."""
    dates: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    session: np.ndarray

    def __len__(self) -> int:
        return len(self.close)


@dataclass(frozen=True)
class Contraction:
    high_idx: int
    low_idx: int
    high: float
    low: float

    @property
    def depth(self) -> float:
        return 1.0 - self.low / self.high


# ── Bars ─────────────────────────────────────────────────────────────────────

def clean_bars(df: pd.DataFrame, calendar: Optional[pd.DatetimeIndex] = None) -> Bars:
    """Sort, drop duplicate days and every bar without a real trade (zero
    volume, missing or non-positive price). Nothing is filled. `calendar` is
    the exchange's sessions; without one, the frame's own rows stand in."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    sessions = df.index if calendar is None else calendar
    prices = df[["Open", "High", "Low", "Close"]].to_numpy(float)
    volume = df["Volume"].to_numpy(float)
    traded = (np.isfinite(prices) & (prices > 0)).all(axis=1) & (np.nan_to_num(volume) > 0)
    traded &= np.isin(df.index.to_numpy(), sessions.to_numpy())
    df, prices, volume = df[traded], prices[traded], volume[traded]
    return Bars(df.index.to_numpy(), prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3],
                volume, np.searchsorted(sessions.to_numpy(), df.index.to_numpy()))


def missing_fraction(bars: Bars, start: int, end: int) -> float:
    """Share of sessions between two bars (inclusive) with no trade."""
    span = bars.session[end] - bars.session[start] + 1
    return 1.0 - (end - start + 1) / span


# ── Trend-template gate ──────────────────────────────────────────────────────

# Checked in this order; the first failure is the reported reason.
GATE_CHECKS = (
    "liquidity_price", "liquidity_dollar_volume",
    "price_above_ma150", "price_above_ma200", "ma150_above_ma200", "ma200_rising",
    "ma50_above_ma150", "ma50_above_ma200", "price_above_ma50",
    "above_52w_low", "near_52w_high",
)


def gate_table(bars: Bars, cfg: VcpConfig = DEFAULT) -> pd.DataFrame:
    """One row per bar, one boolean per GATE_CHECKS entry, plus `ready` (a year
    of bars), `rs_raw` and `atr_pct`. Every column looks backwards only."""
    close, high, low = pd.Series(bars.close), pd.Series(bars.high), pd.Series(bars.low)
    sma = TechnicalIndicators.sma
    ma_f, ma_m, ma_s = sma(close, cfg.ma_fast), sma(close, cfg.ma_mid), sma(close, cfg.ma_slow)
    dollar = sma(close * pd.Series(bars.volume), cfg.dollar_vol_days)
    low_52w = low.rolling(cfg.year_days).min()
    high_52w = high.rolling(cfg.year_days).max()
    rs = sum(w * (close / close.shift(d) - 1.0) for d, w in cfg.rs_periods)
    return pd.DataFrame({
        "liquidity_price": close >= cfg.min_price,
        "liquidity_dollar_volume": dollar >= cfg.min_dollar_vol,
        "price_above_ma150": close > ma_m,
        "price_above_ma200": close > ma_s,
        "ma150_above_ma200": ma_m > ma_s,
        "ma200_rising": ma_s > ma_s.shift(cfg.ma_slow_rising_days),
        "ma50_above_ma150": ma_f > ma_m,
        "ma50_above_ma200": ma_f > ma_s,
        "price_above_ma50": close > ma_f,
        "above_52w_low": close >= low_52w * (1 + cfg.min_above_low),
        "near_52w_high": close >= high_52w * (1 - cfg.max_below_high),
        "ready": np.arange(len(close)) >= max(cfg.year_days, cfg.ma_slow + cfg.ma_slow_rising_days - 1),
        "rs_raw": rs,
        "atr_pct": TechnicalIndicators.atr(high, low, close, cfg.atr_days) / close,
    })


def rs_percentiles(raw: pd.Series) -> pd.Series:
    """Percentile (0-100) of each name's rs_raw among the names given: the share
    it is at or above. Names without an rs_raw stay NaN."""
    return raw.rank(pct=True, method="max") * 100.0


def is_liquid(gates: pd.DataFrame, i: int) -> bool:
    return bool(gates["liquidity_price"].iat[i] and gates["liquidity_dollar_volume"].iat[i])


def gate_fail_reason(bars: Bars, gates: pd.DataFrame, i: int, rs_percentile: Optional[float],
                     cfg: VcpConfig = DEFAULT) -> Optional[str]:
    """None if bar i passes everything."""
    if not gates["ready"].iat[i]:
        return "insufficient_history"
    if missing_fraction(bars, i - cfg.year_days + 1, i) > cfg.max_missing_frac:
        return "thin_data"
    for name in GATE_CHECKS:
        if not gates[name].iat[i]:
            return name
    if rs_percentile is None or not np.isfinite(rs_percentile):
        return "rs_unavailable"
    if rs_percentile < cfg.min_rs_percentile:
        return "rs_percentile"
    return None


# ── Base and contractions ────────────────────────────────────────────────────

def atr_scale(atr_pct: float, cfg: VcpConfig = DEFAULT) -> float:
    """How much to widen the depth thresholds: 1 up to atr_ref, then in
    proportion to ATR%, at most max_atr_scale."""
    if not cfg.scale_by_atr or not np.isfinite(atr_pct):
        return 1.0
    return float(min(max(atr_pct / cfg.atr_ref, 1.0), cfg.max_atr_scale))


def find_base_start(bars: Bars, i: int, cfg: VcpConfig = DEFAULT) -> Optional[int]:
    """Index of the base high: the highest high between max_base_days and
    min_base_days ago (the latest, if tied). Newer highs are left out so that
    a breakout doesn't erase the base it broke out of."""
    last = i - cfg.min_base_days
    if last < 0:
        return None
    first = max(0, i - cfg.max_base_days)
    window = bars.high[first:last + 1]
    return first + len(window) - 1 - int(np.argmax(window[::-1]))


def find_swings(high: Sequence[float], low: Sequence[float], start: int, end: int,
                      threshold: float) -> List[Contraction]:
    """Successive high-to-low pullbacks from bar `start` (a high) to `end`.

    A swing counts only once price has reversed by `threshold` from it, so the
    list never depends on bars after `end`. The newest pullback is included
    while still unfinished, with the low reached so far. Within one bar the
    order of high and low is unknown: a bar never both sets an extreme and
    reverses from it, and a bar that trades above the swing high counts as that
    new high (a pivot that has been exceeded is no pivot)."""
    out: List[Contraction] = []
    falling = True
    hi_idx, hi = start, high[start]
    lo_idx, lo = -1, np.inf
    for k in range(start + 1, end + 1):
        if not falling:
            if high[k] > hi:
                hi_idx, hi = k, high[k]
            elif low[k] <= hi * (1 - threshold):
                falling, lo_idx, lo = True, k, low[k]
            continue
        deep_enough = lo <= hi * (1 - threshold)
        if high[k] > hi or (low[k] >= lo and deep_enough and high[k] >= lo * (1 + threshold)):
            if deep_enough:
                out.append(Contraction(hi_idx, lo_idx, hi, lo))
                falling = False
            hi_idx, hi, lo_idx, lo = k, high[k], -1, np.inf
        elif low[k] < lo:
            lo_idx, lo = k, low[k]
    if falling and lo <= hi * (1 - threshold):
        out.append(Contraction(hi_idx, lo_idx, hi, lo))
    return out


def merge_swings(swings: Sequence[Contraction]) -> List[Contraction]:
    """Fold the wiggles inside a pullback or its recovery into one contraction:
    from the first swing's high to the lowest low after it, then from the
    highest high after that low to the lowest low after it, and so on."""
    out: List[Contraction] = []
    rest = list(swings)
    while rest:
        top = rest[0] if not out else max(rest, key=lambda c: (c.high, c.high_idx))
        rest = rest[rest.index(top):]
        bottom = min(rest, key=lambda c: (c.low, -c.low_idx))
        out.append(Contraction(top.high_idx, bottom.low_idx, top.high, bottom.low))
        rest = rest[rest.index(bottom) + 1:]
    return out


def find_contractions(high: Sequence[float], low: Sequence[float], start: int, end: int,
                      threshold: float, cfg: VcpConfig = DEFAULT) -> List[Contraction]:
    swings = find_swings(high, low, start, end, threshold)
    return merge_swings(swings) if cfg.merge_swings else swings


def _shrinks(earlier: float, later: float, cfg: VcpConfig) -> bool:
    return later <= earlier * cfg.max_shrink_ratio


def keeps_shrinking(depths: Sequence[float], cfg: VcpConfig = DEFAULT) -> bool:
    return all(_shrinks(a, b, cfg) for a, b in zip(depths, depths[1:]))


def is_valid_vcp(depths: Sequence[float], cfg: VcpConfig = DEFAULT) -> bool:
    return cfg.min_contractions <= len(depths) <= cfg.max_contractions and keeps_shrinking(depths, cfg)


# ── Component scores (each 0..1) ─────────────────────────────────────────────

def _ramp(x: float, zero: float, full: float) -> float:
    """0 at `zero`, 1 at `full`, straight line between, flat outside."""
    return float(min(max((x - zero) / (full - zero), 0.0), 1.0))


def progression_score(depths: Sequence[float], cfg: VcpConfig = DEFAULT) -> float:
    """Half: share of steps that shrink. Half: average shrink per step."""
    if len(depths) < 2:
        return 0.0
    steps = list(zip(depths, depths[1:]))
    shrink = [1.0 - b / a for a, b in steps]
    consistency = sum(_shrinks(a, b, cfg) for a, b in steps) / len(steps)
    return 0.5 * consistency + 0.5 * _ramp(float(np.mean(shrink)), 0.0, cfg.full_shrink)


def tightness_score(final_depth: float, scale: float, cfg: VcpConfig = DEFAULT) -> float:
    d = final_depth / scale
    if d <= cfg.tight_good:
        return cfg.tight_good_score + (1 - cfg.tight_good_score) * _ramp(d, cfg.tight_good, cfg.tight_full)
    return cfg.tight_good_score * _ramp(d, cfg.tight_zero, cfg.tight_good)


def dryup_ratio(bars: Bars, last: Contraction, i: int, cfg: VcpConfig = DEFAULT) -> float:
    """Average volume from the day after the final high to its low (at least
    min_dryup_days, never past bar i) over the vol_avg_days average at bar i."""
    start = last.high_idx + 1
    stop = min(i, max(last.low_idx, start + cfg.min_dryup_days - 1))
    average = bars.volume[max(0, i - cfg.vol_avg_days + 1):i + 1].mean()
    return float(bars.volume[start:stop + 1].mean() / average)


def dryup_score(ratio: float, cfg: VcpConfig = DEFAULT) -> float:
    return _ramp(ratio, cfg.dryup_zero, cfg.dryup_full)


def proximity_score(from_pivot: float, cfg: VcpConfig = DEFAULT) -> float:
    """from_pivot: price / pivot - 1 (negative below the pivot)."""
    if from_pivot <= 0:
        return _ramp(-from_pivot, cfg.near_zero, cfg.near_full)
    if from_pivot > cfg.max_extended:
        return 0.0
    return cfg.extended_score * _ramp(from_pivot, cfg.max_extended, 0.0)


def prior_uptrend(bars: Bars, base_start: int, cfg: VcpConfig = DEFAULT) -> float:
    """Gain into the base high from the lowest low of the uptrend_days before it."""
    lows = bars.low[max(0, base_start - cfg.uptrend_days):base_start + 1]
    return float(bars.high[base_start] / lows.min() - 1.0)


def base_score(base_depth: float, uptrend: float, scale: float, cfg: VcpConfig = DEFAULT) -> float:
    return (0.5 * _ramp(base_depth / scale, cfg.base_depth_zero, cfg.base_depth_full)
            + 0.5 * _ramp(uptrend, 0.0, cfg.uptrend_full))


# ── Scoring one bar ──────────────────────────────────────────────────────────

def _empty_result() -> Dict[str, Any]:
    return {
        "vcp_score": 0.0, "valid_vcp": False,
        "gate_passed": False, "gate_fail_reason": None, "no_score_reason": None,
        "n_contractions": 0, "contraction_depths": [],
        "base_depth_pct": np.nan, "base_length_days": 0, "base_start": None,
        "final_depth_pct": np.nan, "vol_dryup_ratio": np.nan,
        "pivot_price": np.nan, "pct_from_pivot": np.nan,
        "prior_uptrend_pct": np.nan, "atr_pct": np.nan, "atr_scale": np.nan,
        "c_progression": 0.0, "c_tightness": 0.0, "c_dryup": 0.0, "c_proximity": 0.0, "c_base": 0.0,
    }


def _pattern(bars: Bars, i: int, atr_pct: float, cfg: VcpConfig) -> Dict[str, Any]:
    """The base, its contractions and the score, ignoring the gate."""
    out = _empty_result()
    scale = atr_scale(atr_pct, cfg)
    out["atr_pct"], out["atr_scale"] = atr_pct * 100, scale
    start = find_base_start(bars, i, cfg)
    if start is None:
        out["no_score_reason"] = "no_base"
        return out
    out["base_start"] = bars.dates[start]
    out["base_length_days"] = i - start
    base_depth = 1.0 - bars.low[start:i + 1].min() / bars.high[start]
    out["base_depth_pct"] = base_depth * 100
    if missing_fraction(bars, start, i) > cfg.max_missing_frac:
        out["no_score_reason"] = "thin_base"
        return out

    legs = find_contractions(bars.high, bars.low, start, i, cfg.zigzag_pct * scale, cfg)
    depths = [c.depth for c in legs]
    out["n_contractions"], out["contraction_depths"] = len(legs), [d * 100 for d in depths]
    if not legs:
        out["no_score_reason"] = "too_few_contractions"
        return out
    last = legs[-1]
    from_pivot = bars.close[i] / last.high - 1.0
    ratio = dryup_ratio(bars, last, i, cfg)
    uptrend = prior_uptrend(bars, start, cfg)
    out.update(
        final_depth_pct=last.depth * 100, vol_dryup_ratio=ratio, pivot_price=float(last.high),
        pct_from_pivot=from_pivot * 100, prior_uptrend_pct=uptrend * 100)
    if len(legs) < cfg.min_contractions:
        out["no_score_reason"] = "too_few_contractions"
        return out

    out.update(
        valid_vcp=is_valid_vcp(depths, cfg),
        c_progression=cfg.w_progression * progression_score(depths, cfg),
        c_tightness=cfg.w_tightness * tightness_score(last.depth, scale, cfg),
        c_dryup=cfg.w_dryup * dryup_score(ratio, cfg),
        c_proximity=cfg.w_proximity * proximity_score(from_pivot, cfg),
        c_base=cfg.w_base * base_score(base_depth, uptrend, scale, cfg))
    total = sum(out[c] for c in ("c_progression", "c_tightness", "c_dryup", "c_proximity", "c_base"))
    out["vcp_score"] = float(total if out["valid_vcp"] else min(total, cfg.invalid_cap))
    return out


def score_at(bars: Bars, gates: pd.DataFrame, i: int, rs_percentile: Optional[float],
             cfg: VcpConfig = DEFAULT) -> Dict[str, Any]:
    """Score bar i. A gate failure scores 0 and reports the reason."""
    reason = gate_fail_reason(bars, gates, i, rs_percentile, cfg)
    if reason is not None:
        out = _empty_result()
        out["gate_fail_reason"] = reason
        return out
    out = _pattern(bars, i, float(gates["atr_pct"].iat[i]), cfg)
    out["gate_passed"] = True
    return out


def score_ticker(df: pd.DataFrame, rs_percentile: Optional[float],
                 calendar: Optional[pd.DatetimeIndex] = None,
                 cfg: VcpConfig = DEFAULT) -> Dict[str, Any]:
    """Score the last bar of one ticker's daily OHLCV. `rs_percentile` is its
    rank among that day's liquid names (rs_percentiles over rs_raw)."""
    bars = clean_bars(df, calendar)
    last_session = len(df.index if calendar is None else calendar) - 1
    if not len(bars) or last_session - bars.session[-1] > cfg.max_stale_sessions:
        out = _empty_result()
        out["gate_fail_reason"] = "stale_data" if len(bars) else "no_data"
        return out
    return score_at(bars, gate_table(bars, cfg), len(bars) - 1, rs_percentile, cfg)
