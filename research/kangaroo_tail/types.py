"""
Data types for the Kangaroo Tail detector. Plain (frozen) containers only --
all behavior lives in indicators.py / detector.py.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TailConfig:
    """Tunable thresholds for bullish Kangaroo Tail detection.

    Defaults below are Phase 2 backtest-validated (see
    KANGAROO_TAIL_VERIFICATION_FINDINGS.md, "sweet_spot" config): a 16-fold
    walk-forward breakout-entry R-multiple backtest (2022-07..2026-07, 134
    CAN tickers, rr=1.5) found win_rate=56.6%, avg_R=+0.246, profit_factor
    1.72, fold-paired t-test p=0.033 (n=143 trades) -- a real, if modest,
    edge for the ENTRY MECHANIC these defaults are meant to be traded with:
    wait for price to trade through the tail's own high (a pending
    breakout/buy-stop), stop at the tail's low, target a multiple of that
    risk. Same-day "buy at the tail's own close" entry showed no edge under
    any config tested and should not be used live (see kangaroo_buy.py).

    lookback            -- bars strictly before the candidate bar used to
                            establish "prior range" (its rolling low).
    atr_period          -- Wilder's ATR period (14 matches this repo's
                            existing convention, see position_monitor.py /
                            swing_tickers.py).
    break_atr_mult      -- the candidate bar's low must undercut the prior
                            N-bar low by at least this many ATRs -- the
                            "genuine structure break" requirement that
                            distinguishes a Kangaroo Tail from a plain
                            hammer. Set far below zero (a de facto no-op)
                            by default: the walk-forward found this rule
                            excluded real, working setups (e.g. the AAPL
                            2026-09-09 bar -- a shallow pullback near a
                            rising 50-day MA, not a breakdown below a
                            multi-week low) without measurably improving
                            the edge (see "no_break" vs "strict" in the
                            findings doc).
    close_back_pct      -- the candidate bar's close must land at least
                            this far up its own [low, high] range (e.g. 0.5
                            = upper half) -- the "closed back into the
                            range" requirement.
    wick_dominance_pct  -- the lower wick (min(open, close) - low) must be
                            at least this fraction of the bar's total
                            range.
    wick_atr_mult       -- the lower wick must also be at least this many
                            ATRs long in absolute terms -- catches a
                            dominant wick on an otherwise tiny (low-ATR)
                            bar, which wick_dominance_pct alone would not.
                            1.2 is the walk-forward "sweet spot": below it
                            (down to 0.7) sample size grows a lot but
                            avg_R/win_rate collapse to roughly a third of
                            this value; above it (1.5, the original
                            "strict" value) the sample shrinks to n~13-50
                            without a materially better avg_R.
    max_body_pct        -- the bar's body (|close - open|) must be at most
                            this fraction of its total range.
    volume_lookback     -- bars strictly before the candidate bar used for
                            the average-volume baseline.
    volume_mult         -- the candidate bar's volume must be at least this
                            many times the volume_lookback average. `None`
                            disables the volume filter entirely (Phase 2
                            sweeps with and without it).
    """
    lookback: int = 10
    atr_period: int = 14
    break_atr_mult: float = -999.0
    close_back_pct: float = 0.5
    wick_dominance_pct: float = 0.55
    wick_atr_mult: float = 1.2
    max_body_pct: float = 0.35
    volume_lookback: int = 20
    volume_mult: float | None = 1.3


@dataclass(frozen=True)
class TailSignal:
    """One detected bullish Kangaroo Tail, with the full evidence that
    fired it -- auditable, not just trusted (same convention as
    research/triple_screen's *Verdict types).
    """
    ticker: str
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr: float
    prior_low: float
    break_amount: float          # prior_low - low; how far structure broke
    close_position_pct: float    # (close - low) / (high - low)
    wick_pct: float              # lower wick / (high - low)
    body_pct: float              # |close - open| / (high - low)
    volume_ratio: float | None   # volume / volume_lookback average, or
                                  # None if the volume filter was disabled
