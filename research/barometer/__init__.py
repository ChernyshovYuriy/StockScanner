"""
Barometer -- calibrated probabilities of what the next 20 trading days hold
for each liquid TSX stock (stop-hit risk, +/-10% moves, volatility, beating
XIU.TO). Not a direction caller and not a sleeve: no capital, no positions,
no schedule. See PLAN.md for the phases, targets and pre-registered gates.

Phase 0 (this package so far):
  dataset.py   build_panel() -- one row per liquid stock-day: past-only
               features + future-only labels, the stop computed exactly as
               auto_pipeline.compute_levels() does. CLI:
               python -m research.barometer.dataset                    # build from cache, save, summarise
               python -m research.barometer.dataset --check-lookahead  # real-data truncation proof

Phase 1:
  metrics.py    Brier/log-loss/skill, AUC, calibration, MSE/QLIKE, and a
                date-clustered Newey-West t-stat -- shared by every later phase.
  baselines.py  the base rate and the closed-form volatility rule, scored
                walk-forward (20-session embargo); results in PLAN.md.
               python -m research.barometer.baselines   # -> out/barometer_phase1.csv

Phase 2:
  models.py     logistic (primary) / gradient boosting (challenger) on
                per-day-ranked features + the rule's own logit; ridge for
                volatility; walk-forward out-of-sample predictions scored
                against both baselines. Results in PLAN.md.
               python -m research.barometer.models [--earnings]   # -> out/barometer_phase2*.csv

Phases 3-4:
  gate.py       the pre-registered decision gate (logistic / ridge), T4's
                top-minus-bottom spread, bull/bear breakdown, pooled
                calibration. Verdict in PLAN.md: only T3 (volatility) passes.
               python -m research.barometer.gate   # -> out/barometer_gate.csv
"""
