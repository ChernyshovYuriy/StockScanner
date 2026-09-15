"""
volume_spike_intraday_backtest.py
===================================
RETRACTED (2026-09): every checkpoint/hold finding this script ever
produced (curated-46, full-~900-universe, expanded-106, open-hour) is
INVALID. Root cause, confirmed empirically: yfinance's hourly ("60m")
bars materially UNDERCOUNT real TSX trading volume, and by a wildly
inconsistent amount -- CNQ.TO's hourly-bar sum for a single day came
back 6.5x-12x BELOW that same day's real (accurate, daily-bar-feed)
consolidated volume; most other names 1.1x-2.4x; the ratio erratic even
for the same ticker across consecutive days (likely dark-pool/off-
exchange prints that reach the end-of-day consolidated tape but not
yfinance's intraday feed). `_checkpoint_reads()`'s cum_vol -- used for
EVERY checkpoint's is_event flag, "close" included, since that
checkpoint just sums every bar of the day -- was built on this broken
number the whole time. It never measured the same thing
volume_spike_scanner.py's live signal measures (which correctly reads
the accurate daily-bar feed), so nothing this script concluded about
"is there an edge" or "which checkpoint/hold wins" can be trusted.

The corrected replacements:
  volume_spike_daily_backtest.py       -- daily-bars-only (no hourly
                                           data, so immune to the bug
                                           above): buy at a CONFIRMED
                                           spike day's CLOSE, hold N
                                           trading days forward. Finding:
                                           NEGATIVE, very high confidence
                                           (900 tickers, 4yr, n=121,764,
                                           t~-7 to -11, p~0, every fold).
  volume_spike_same_day_backtest.py    -- the finding that actually
                                           matters: buy INTRADAY on a
                                           day that turns out to be a
                                           confirmed spike (accurate
                                           daily-bar detection; hourly
                                           bars used ONLY for price
                                           within that day, which was
                                           never the broken part), hold
                                           to THAT SAME day's close.
                                           Finding: POSITIVE, 16/16
                                           folds, t=18.34, p~0 -- see
                                           that script's own docstring.

This file is kept only as a documented dead end -- read the two scripts
above for the actual, validated conclusion. Nothing below this notice
was rewritten; it's left exactly as it ran so the mistake (and how it
was caught) stays reproducible.

Original framing, preserved for context: does buying a "volume spike +
price up" event actually have an edge, and if so, WHEN during the
session should you act on it, and how long should you hold?

Same event-study methodology as this repo's other validated/rejected
signal research (research/kangaroo_tail/verify.py, walk_forward_ab.py):
for each (checkpoint, hold) combo, "candidate" = the mean forward return
of every actual spike+price-up event; "baseline" = the mean forward
return of EVERY ticker-day at that same checkpoint/hold (that ticker's
own normal drift at that time of day) -- a fold "wins" when
candidate > baseline, and the per-fold (candidate - baseline)
differences feed a paired one-sample t-test (scipy.stats.ttest_1samp).
This isolates whether the signal predicts an ABOVE-NORMAL move, not just
"stocks were drifting up that quarter."

Data / why this is capped at ~2 years, not the repo's usual 4:
yfinance only hands out free hourly ("60m") bars for the trailing ~730
days -- the longest intraday history available at all (5-min bars only
go back ~60 days, too short to draw any conclusion from -- see
dip_grid_backtest.py's own docstring). market_data.LiveDataProvider.
download_bars() is a per-ticker fetch (no batched intraday endpoint
exists in market_data.py), so this script also runs against a curated
~40-name liquid TSX universe rather than the full VOLUME_SPIKE_TICKERS_URL
list, to keep the fetch to a few minutes.

"Checkpoint" = how much of the trading day has actually happened when you
act, using the SAME spike definition the live scanner uses (today's
cumulative volume vs. that ticker's own trailing 20-trading-day average
daily volume, AND price above yesterday's close) evaluated at 4 points in
the session:
  morning   -- cumulative volume/price through the ~10:30 bar (~11:30 ET)
  midday    -- cumulative volume/price through the ~12:30 bar (~13:30 ET)
  afternoon -- cumulative volume/price through the ~14:30 bar (~15:30 ET)
  close     -- the full day's volume/price (what the live scanner sees
               late in the session)

"Hold" = same_session_close (exit that same day), next_day_close, or
2day_close. same_session_close is skipped for the `close` checkpoint --
entry and exit would be the same bar.

N_FOLDS=8, not this repo's usual 16: with only ~2 years of history (vs.
the 4-year window kangaroo_tail/verify.py and walk_forward_ab.py use),
16 folds would be ~45 calendar days each -- too thin a slice for a
~40-ticker universe to reliably have events in every fold. 8 folds
(~90 days each) trades fold count for per-fold sample size; rerun with
more folds later if event counts turn out to comfortably support it.

Usage:
    python volume_spike_intraday_backtest.py
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH
from market_data import DEFAULT_PROVIDER
from market_data_cache import sync_and_load
from time_utils import market_today

# Curated liquid, large/mid-cap TSX names spanning major sectors -- the
# same "quick to iterate on" tradeoff dip_grid_backtest.py made with its
# own 3-ticker AAPL/NVDA/AMD universe, just broader since spike events are
# ticker-specific and rare (more names -> more independent event
# instances to test against, which matters for the fold-level stats
# above). Not derived from conviction_watchlist.quality_filter (that
# module's own per-ticker yfinance .info fetch is a second, slow,
# unrelated cost this script doesn't need) -- a plain hand-picked list,
# reviewable here.
#
# Expanded from an original 46-name set to ~120 (2026-09) after a
# midday-checkpoint/same-session-close finding on the original 46
# REVERSED sign when tested against the full ~900-ticker
# VOLUME_SPIKE_TICKERS_URL universe (+0.24pp there vs -0.12pp on the full
# list, p=0.038). That full-universe test isn't a clean refutation,
# though -- it's dominated by junior/speculative TSXV/CSE names (~900 of
# them vs the original 46 blue chips), which spike far more often
# (~18 events/ticker vs ~2/ticker) and plausibly behave differently
# (a midday spike on a thinly-traded junior more often reads as a
# fade-prone news pump than the large-cap "real buying interest" read
# this signal is meant to capture). This expansion stays within the same
# liquid large/mid-cap character as the original list -- more names to
# actually test generalization, not a dilution with the population that
# produced the reversal.
TICKERS = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO", "EQB.TO",
    "SU.TO", "CNQ.TO", "IMO.TO", "CVE.TO", "TOU.TO", "ENB.TO", "TRP.TO", "PPL.TO",
    "ARX.TO", "MEG.TO", "WCP.TO", "BTE.TO", "PXT.TO", "KEY.TO", "PSK.TO", "GEI.TO",
    "ABX.TO", "AEM.TO", "FNV.TO", "WPM.TO", "TECK-B.TO", "NTR.TO",
    "K.TO", "CG.TO", "FM.TO", "LUN.TO", "CCO.TO", "MX.TO", "CAS.TO",
    "IMG.TO", "ELD.TO", "SSRM.TO", "PAAS.TO",
    "CNR.TO", "CP.TO", "WCN.TO", "TFII.TO", "CJT.TO",
    "STN.TO", "WSP.TO", "SNC.TO", "TIH.TO", "FTT.TO", "BYD.TO", "RUS.TO", "DOO.TO", "MG.TO",
    "BCE.TO", "T.TO", "RCI-B.TO", "QBR-B.TO", "CCA.TO", "CJR-B.TO",
    "SHOP.TO", "CSU.TO", "OTEX.TO", "DSG.TO", "KXS.TO", "LSPD.TO", "NVEI.TO", "TIXT.TO", "BB.TO", "ENGH.TO",
    "MFC.TO", "SLF.TO", "IFC.TO", "GWO.TO", "POW.TO", "FFH.TO", "ONEX.TO", "AGF-B.TO", "X.TO",
    "FTS.TO", "EMA.TO", "AQN.TO", "H.TO", "CU.TO", "BEP-UN.TO", "BIP-UN.TO", "BAM.TO", "BN.TO",
    "ATD.TO", "L.TO", "DOL.TO", "MRU.TO", "CTC-A.TO", "QSR.TO", "SAP.TO", "PBH.TO", "EMP-A.TO",
    "REI-UN.TO", "CAR-UN.TO", "AP-UN.TO", "SRU-UN.TO", "GRT-UN.TO",
    "WELL.TO",
    "AC.TO", "GIB-A.TO", "WN.TO",
]

INTRADAY_INTERVAL = "60m"
INTRADAY_PERIOD = "730d"          # yfinance's real free-tier ceiling for hourly bars
DAILY_HISTORY_DAYS = 800          # extra buffer before the hourly window so the 20-day rolling average has warm-up data

AVG_VOLUME_DAYS = 20              # matches volume_spike_scanner.py's production definition
ROUND_TRIP_COST_BPS = 5           # matches dip_grid_backtest.py's own assumption

# hour-of-day cutoff (bar START hour, America/Toronto) included in each
# checkpoint's cumulative volume/price read; None = every bar of the day.
CHECKPOINTS: dict[str, int | None] = {
    "open_hour": 9,   # ONLY the 9:30-10:30 bar -- cumulative volume/price through ~10:30, before "morning" below has even started accumulating its second hour
    "morning": 10,
    "midday": 12,
    "afternoon": 14,
    "close": None,
}

HOLDS = ["same_session_close", "next_day_close", "2day_close"]

N_FOLDS = 8
FORWARD_BUFFER_DAYS = 5  # daily data goes through today, so a small buffer is enough for the 2-day hold near the window's end
_today = market_today()
END = (_today - pd.Timedelta(days=FORWARD_BUFFER_DAYS)).strftime("%Y-%m-%d")
START = (_today - pd.Timedelta(days=720)).strftime("%Y-%m-%d")  # a bit inside the 730-day hourly ceiling
DAILY_FETCH_START = (_today - pd.Timedelta(days=DAILY_HISTORY_DAYS)).strftime("%Y-%m-%d")
DAILY_FETCH_END = _today.strftime("%Y-%m-%d")

RESULTS_CSV = OUT_PATH / "volume_spike_intraday_backtest_folds.csv"
SUMMARY_CSV = OUT_PATH / "volume_spike_intraday_backtest_summary.csv"


# ── Data prep ────────────────────────────────────────────────────────────────

def _load_daily(ticker: str, daily_by_ticker: dict) -> pd.DataFrame | None:
    df = daily_by_ticker.get(ticker)
    if df is None or df.empty:
        return None
    df = df.sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df


def _load_hourly(ticker: str) -> pd.DataFrame | None:
    try:
        df = DEFAULT_PROVIDER.download_bars(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df.sort_index()


def _checkpoint_reads(hourly: pd.DataFrame, daily: pd.DataFrame) -> dict[date, dict[str, tuple[float, float]]]:
    """{date: {checkpoint: (cumulative_volume, price_at_checkpoint)}} for
    every trading day present in `hourly`.

    Price is rescaled onto the SAME dividend/split-adjusted basis as
    `daily` -- yfinance's auto_adjust=True does not back-adjust intraday
    bars the way it does daily bars (confirmed empirically: hourly Close
    for a 2-year-old RY.TO bar came back ~6% above that day's adjusted
    daily Close, converging toward ~0% near today, exactly the signature
    of a raw-vs-adjusted mismatch for a dividend payer). Mixing the two
    directly would make every entry price systematically too HIGH for
    older events, producing a fake, calendar-position-dependent negative
    bias in every computed return regardless of hold length -- exactly
    what an earlier version of this script produced (~-2.4% on every
    single checkpoint/hold combo, including same-session holds, which
    should never move that uniformly). Fix: each day's own hourly EOD
    bar and that same day's daily Close represent the identical trading
    session, so their ratio is that day's adjustment factor -- applied to
    every intraday checkpoint price that day (dividends aren't granular
    by hour, so a same-day factor is accurate enough for this purpose).
    """
    out: dict[date, dict[str, tuple[float, float]]] = {}
    for day, group in hourly.groupby(hourly.index.date):
        group = group.sort_index()
        ts = pd.Timestamp(day)
        if ts not in daily.index:
            continue
        raw_eod_close = float(group["Close"].iloc[-1])
        adjusted_close = float(daily.loc[ts, "Close"])
        if raw_eod_close <= 0 or not np.isfinite(adjusted_close):
            continue
        adj_factor = adjusted_close / raw_eod_close

        day_reads = {}
        for cp_name, hour_cutoff in CHECKPOINTS.items():
            subset = group if hour_cutoff is None else group[group.index.hour <= hour_cutoff]
            if subset.empty:
                continue
            day_reads[cp_name] = (float(subset["Volume"].sum()), float(subset["Close"].iloc[-1]) * adj_factor)
        if day_reads:
            out[day] = day_reads
    return out


def _ticker_events(ticker: str, daily: pd.DataFrame, hourly: pd.DataFrame) -> list[dict]:
    """One row per (date, checkpoint, hold) this ticker actually has data
    for -- entry_price/exit_price/ret/is_event, long-format so it can be
    pooled into per-fold candidate/baseline groups downstream."""
    avg_volume_20 = daily["Volume"].rolling(AVG_VOLUME_DAYS).mean().shift(1)
    prev_close = daily["Close"].shift(1)
    session_close = daily["Close"]
    next_close = daily["Close"].shift(-1)
    close_2d = daily["Close"].shift(-2)

    cost = ROUND_TRIP_COST_BPS / 10_000
    reads = _checkpoint_reads(hourly, daily)

    rows = []
    for day, day_reads in reads.items():
        ts = pd.Timestamp(day)
        if ts not in daily.index:
            continue
        avg_vol = avg_volume_20.get(ts)
        p_close = prev_close.get(ts)
        if avg_vol is None or p_close is None or not np.isfinite(avg_vol) or not np.isfinite(p_close) or avg_vol <= 0:
            continue

        for cp_name, (cum_vol, cp_price) in day_reads.items():
            is_event = (cum_vol > avg_vol) and (cp_price > p_close)

            for hold in HOLDS:
                if hold == "same_session_close" and cp_name == "close":
                    continue  # entry == exit, degenerate
                exit_price = {
                    "same_session_close": session_close.get(ts),
                    "next_day_close": next_close.get(ts),
                    "2day_close": close_2d.get(ts),
                }[hold]
                if exit_price is None or not np.isfinite(exit_price) or cp_price <= 0:
                    continue
                ret = (exit_price - cp_price) / cp_price - cost
                rows.append({
                    "ticker": ticker, "date": ts, "checkpoint": cp_name, "hold": hold,
                    "is_event": is_event, "ret": ret,
                })
    return rows


# ── Folds ────────────────────────────────────────────────────────────────────

def _fold_bounds() -> np.ndarray:
    return pd.date_range(start=START, end=END, periods=N_FOLDS + 1).values


def _fold_of(dates: pd.DatetimeIndex, bounds: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bounds, dates.values, side="right") - 1
    idx[(dates.values < bounds[0]) | (dates.values > bounds[-1])] = -1
    idx[idx == N_FOLDS] = N_FOLDS - 1
    return idx


# ── Main sweep ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Universe: {len(TICKERS)} curated liquid TSX tickers  |  window {START} .. {END}  |  {N_FOLDS} folds")

    t0 = time.perf_counter()
    daily_by_ticker = sync_and_load(TICKERS, start=DAILY_FETCH_START, end=DAILY_FETCH_END, quiet=True)
    print(f"Daily bars loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    all_rows = []
    for i, ticker in enumerate(TICKERS, 1):
        daily = _load_daily(ticker, daily_by_ticker)
        hourly = _load_hourly(ticker)
        if daily is None or hourly is None or len(daily) < AVG_VOLUME_DAYS + 5:
            print(f"  [{i}/{len(TICKERS)}] {ticker}: skipped (insufficient data)")
            continue
        rows = _ticker_events(ticker, daily, hourly)
        print(f"  [{i}/{len(TICKERS)}] {ticker}: {len(rows)} ticker-day-checkpoint-hold rows")
        all_rows.extend(rows)
    print(f"Hourly bars fetched + events built in {time.perf_counter() - t0:.1f}s")

    table = pd.DataFrame(all_rows)
    if table.empty:
        print("No usable data at all -- aborting.")
        return

    bounds = _fold_bounds()
    table["fold"] = _fold_of(pd.DatetimeIndex(table["date"]), bounds)
    table = table[table["fold"] >= 0]

    summary_rows, detail_rows = [], []
    for cp_name in CHECKPOINTS:
        for hold in HOLDS:
            if hold == "same_session_close" and cp_name == "close":
                continue
            subset = table[(table["checkpoint"] == cp_name) & (table["hold"] == hold)]
            if subset.empty:
                continue

            diffs, wins, n_valid, n_events_used = [], 0, 0, 0
            cand_means, base_means = [], []
            for f in range(N_FOLDS):
                fold_rows = subset[subset["fold"] == f]
                cand = fold_rows.loc[fold_rows["is_event"], "ret"]
                base = fold_rows["ret"]  # every ticker-day at this checkpoint/hold, event or not
                if cand.empty or base.empty:
                    continue
                cand_mean, base_mean = float(cand.mean()), float(base.mean())
                diff = cand_mean - base_mean
                diffs.append(diff)
                wins += int(diff > 0)
                n_valid += 1
                n_events_used += len(cand)
                cand_means.append(cand_mean)
                base_means.append(base_mean)
                detail_rows.append({
                    "checkpoint": cp_name, "hold": hold, "fold": f + 1, "n_events": len(cand),
                    "candidate_ret_pct": round(cand_mean * 100, 3),
                    "baseline_ret_pct": round(base_mean * 100, 3),
                    "diff_pct": round(diff * 100, 3),
                })

            if n_valid >= 2:
                t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
                mean_diff = float(np.mean(diffs))
            else:
                t_stat, p_val = float("nan"), float("nan")
                mean_diff = float(np.mean(diffs)) if diffs else float("nan")

            summary_rows.append({
                "checkpoint": cp_name, "hold": hold,
                "total_events": int(subset["is_event"].sum()),
                "events_used": n_events_used, "folds_used": n_valid,
                "wins": wins,
                "candidate_ret_pct": round(float(np.mean(cand_means)) * 100, 3) if cand_means else float("nan"),
                "baseline_ret_pct": round(float(np.mean(base_means)) * 100, 3) if base_means else float("nan"),
                "mean_diff_pct": round(mean_diff * 100, 3) if np.isfinite(mean_diff) else float("nan"),
                "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else float("nan"),
                "p_value": round(p_val, 3) if np.isfinite(p_val) else float("nan"),
            })

    summary = pd.DataFrame(summary_rows).sort_values(["checkpoint", "hold"])
    print()
    print(tabulate(summary, headers="keys", tablefmt="simple", showindex=False))

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {RESULTS_CSV} and {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
