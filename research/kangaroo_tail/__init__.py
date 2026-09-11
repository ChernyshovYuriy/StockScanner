"""
Kangaroo Tail indicator -- a spike-and-reject daily candle used here as a
bullish reversal signal for swing trading (see the plan discussed in-session
2026-09-10; no separate design doc yet).

A bullish Kangaroo Tail is a false-breakdown bar: its low pierces below the
prior N-bar range by a meaningful multiple of ATR (a genuine structure
break, not noise), then it closes back up into the range with a long lower
wick and a small body -- price rejected the lower level within the same
session. This is deliberately stricter than a plain hammer/pin bar, which
requires no structural break.

Bearish (topping) tails are the mirror image but out of scope for now --
only bullish detection is implemented; see detector.py for the one place
that would need a `direction` parameter to extend to both.

Working order: Phase 1 (this commit) is the reference indicator only --
pure detection math plus a batch CLI for eyeballing detections against a
chart. No persistence, no email, no live scanning. Phase 2 (historical
verification: does this pattern have a real forward-looking edge, walk-
forward tested across the CAN_TICKERS_URL universe) and Phase 3 (live daily
tracker + email digest, gated on Phase 2 clearing) are separate, later
efforts -- see the plan for what each phase covers.

Phase 1 (this commit) -- see:
  types.py      TailConfig (tunable thresholds), TailSignal (a detection's
                 full evidence, so a result is auditable, not just trusted).
  indicators.py  Wilder's ATR -- the only shared math the detector needs.
  detector.py    detect_kangaroo_tail() (single-bar detection, no
                 lookahead), confirms_next_bar() (the optional next-bar
                 confirmation variant), scan_history() (walks a ticker's
                 full bar history collecting every detection, reused by
                 both this phase's batch CLI and Phase 2's verification
                 harness).
  batch.py       CLI: python -m research.kangaroo_tail.batch AAPL SLF.TO
"""
