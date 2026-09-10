"""
Standalone research tool (not wired into any paper-trading sleeve): Elder-Ray
for a single ticker -- Alexander Elder's direct operationalization of "the
balance of power between bulls and bears" (Trading for a Living).

  EMA        = N-period EMA of Close (N=13 is Elder's own default; the trend
               filter -- rising EMA = uptrend, falling EMA = downtrend)
  Bull Power = High - EMA   (how far bulls pushed price above the trend)
  Bear Power = Low  - EMA   (how far bears pushed price below the trend)

Reading, per Elder's textbook rules:
  - Uptrend (EMA rising) + Bear Power negative but rising toward zero
    -> bears are losing strength inside an uptrend: BUY setup.
  - Downtrend (EMA falling) + Bull Power positive but falling toward zero
    -> bulls are losing strength inside a downtrend: SELL setup.
  - Both powers positive (Low > EMA)   -> bulls fully in control, no bears.
  - Both powers negative (High < EMA)  -> bears fully in control, no bulls.
  - Trend flat, or the two readings disagree with the trend -> neither camp
    has a clear edge: STAND ASIDE (Elder's own prescription for that case).

Weekly trend gate (Elder's "Triple Screen"): a daily BUY/SELL setup is only
a real signal when it agrees with the weekly-chart trend (Screen 1, the
"tide") -- Elder-Ray on the daily chart is Screen 2, the reflex within that
tide. A daily BUY setup against a falling weekly EMA, or a daily SELL setup
against a rising weekly EMA, is downgraded to STAND ASIDE with a note; pass
--no-weekly-gate to see the raw daily-only reads instead.

Per the book, Screen 1 is read from an already-*closed* week and then applied
to the days that follow -- you don't know how this week will close until it
closes. So each daily row is gated against the trend of the most recently
completed weekly bar as of that row's date (point-in-time, shown in the
`weekly_trend` column), never against a still-forming current week or a
single "as of today" snapshot re-applied to the whole printed history.

This is a research tool, not a production signal: no DB writes, no live
trading, just console output. Run standalone:

    python research/elder_ray.py SLF.TO
    python research/elder_ray.py AAPL --period 6mo --ema 13 --lookback 15
    python research/elder_ray.py AAPL --no-weekly-gate

Recommended swing-trader command (Elder's own daily/weekly parameters, with
the weekly Triple Screen gate left on):

    python research/elder_ray.py SLF.TO --period 1y --interval 1d --ema 13 \\
        --lookback 15 --weekly-ema 13 --weekly-period 2y
"""
import argparse

import pandas as pd
import yfinance as yf
from tabulate import tabulate

EMA_PERIOD = 13     # Elder's own default for daily bars
LOOKBACK = 10        # rows of history to print
TREND_LAG = 3         # EMA compared to this many bars back to call trend direction

WEEKLY_EMA_PERIOD = 13   # Elder's own default weekly EMA, mirroring the daily one
WEEKLY_PERIOD = "2y"      # yfinance history window for the weekly trend gate
WEEKLY_TREND_LAG = 3       # weekly EMA compared to this many weeks back for trend direction


def fetch_bars(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def compute_elder_ray(bars: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    out = bars.copy()
    out["EMA"] = out["Close"].ewm(span=ema_period, adjust=False).mean()
    out["BullPower"] = out["High"] - out["EMA"]
    out["BearPower"] = out["Low"] - out["EMA"]
    return out


def classify_row(row, prev_row, trend: str, weekly_trend: str = None) -> str:
    """One session's read, per Elder's textbook rules (see module docstring).

    `weekly_trend` ("up"/"down"/"flat"/None) is the Triple Screen gate: a daily
    BUY setup against a falling weekly EMA, or a daily SELL setup against a
    rising weekly EMA, is downgraded to STAND ASIDE. None (or "flat") skips
    the gate -- the raw daily-only read is returned unchanged.
    """
    bull, bear = row["BullPower"], row["BearPower"]

    if bull > 0 and bear > 0:
        return "BULLS IN CONTROL"
    if bull < 0 and bear < 0:
        return "BEARS IN CONTROL"

    if trend == "up" and bear < 0 and prev_row is not None and bear > prev_row["BearPower"]:
        if weekly_trend == "down":
            return "STAND ASIDE (weekly trend disagrees)"
        return "BUY SETUP (bears fading in uptrend)"
    if trend == "down" and bull > 0 and prev_row is not None and bull < prev_row["BullPower"]:
        if weekly_trend == "up":
            return "STAND ASIDE (weekly trend disagrees)"
        return "SELL SETUP (bulls fading in downtrend)"

    return "STAND ASIDE"


def trend_at(df: pd.DataFrame, i: int, lag: int) -> str:
    if i < lag:
        return "flat"
    ema_now, ema_then = df["EMA"].iloc[i], df["EMA"].iloc[i - lag]
    if ema_now > ema_then:
        return "up"
    if ema_now < ema_then:
        return "down"
    return "flat"


def weekly_trend_by_date(daily_index: pd.DatetimeIndex, weekly_df: pd.DataFrame, lag: int) -> pd.Series:
    """Point-in-time weekly trend for each daily date, per Elder's actual practice:
    read the weekly chart once it closes, then trade the days that follow against
    that already-closed read. A weekly bar's own trend reading only becomes
    "known" starting the following week -- never for days inside the week that
    produced it, which would mean judging a week's trend before it has closed.
    """
    daily_index = pd.DatetimeIndex(daily_index)
    weekly_index = pd.DatetimeIndex(weekly_df.index)
    if daily_index.tz is not None:
        daily_index = daily_index.tz_localize(None)
    if weekly_index.tz is not None:
        weekly_index = weekly_index.tz_localize(None)

    trends = pd.Series([trend_at(weekly_df, i, lag) for i in range(len(weekly_df))], index=weekly_index)
    known_as_of_next_week = trends.shift(1)  # a week's trend is knowable only once it has closed

    weekly_lookup = pd.DataFrame({"date": known_as_of_next_week.index, "weekly_trend": known_as_of_next_week.to_numpy()})
    daily_lookup = pd.DataFrame({"date": daily_index})
    merged = pd.merge_asof(daily_lookup.sort_values("date"), weekly_lookup.sort_values("date"),
                            on="date", direction="backward")
    return pd.Series(merged["weekly_trend"].to_numpy(), index=daily_index)


def main():
    parser = argparse.ArgumentParser(description="Elder-Ray bull/bear power for a single ticker")
    parser.add_argument("ticker", help="e.g. SLF.TO or AAPL")
    parser.add_argument("--period", default="1y", help="yfinance history window (default 1y)")
    parser.add_argument("--interval", default="1d", help="yfinance bar interval (default 1d)")
    parser.add_argument("--ema", type=int, default=EMA_PERIOD, help=f"EMA period (default {EMA_PERIOD})")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help=f"rows to print (default {LOOKBACK})")
    parser.add_argument("--weekly-ema", type=int, default=WEEKLY_EMA_PERIOD,
                         help=f"weekly EMA period for the Triple Screen trend gate (default {WEEKLY_EMA_PERIOD})")
    parser.add_argument("--weekly-period", default=WEEKLY_PERIOD,
                         help=f"yfinance history window for the weekly trend gate (default {WEEKLY_PERIOD})")
    parser.add_argument("--no-weekly-gate", action="store_true",
                         help="skip the weekly trend gate and show raw daily-only reads")
    args = parser.parse_args()

    if args.ema < 1:
        print(f"WARNING: --ema must be >= 1 (got {args.ema})")
        return
    if args.lookback < 1:
        print(f"WARNING: --lookback must be >= 1 (got {args.lookback})")
        return
    if not args.no_weekly_gate and args.weekly_ema < 1:
        print(f"WARNING: --weekly-ema must be >= 1 (got {args.weekly_ema})")
        return

    bars = fetch_bars(args.ticker, args.period, args.interval)
    if bars.empty:
        print(f"WARNING: no data returned for {args.ticker}")
        return
    if len(bars) < args.ema + TREND_LAG:
        print(f"WARNING: only {len(bars)} bars returned, need at least {args.ema + TREND_LAG} "
              f"for a meaningful EMA-{args.ema} trend read")
        return

    df = compute_elder_ray(bars, args.ema)

    weekly_trend_series = pd.Series([None] * len(df), index=df.index)
    weekly_gate_active = False
    if not args.no_weekly_gate:
        weekly_bars = fetch_bars(args.ticker, args.weekly_period, "1wk")
        if len(weekly_bars) < args.weekly_ema + WEEKLY_TREND_LAG:
            print(f"WARNING: only {len(weekly_bars)} weekly bars returned, need at least "
                  f"{args.weekly_ema + WEEKLY_TREND_LAG} for a weekly trend read -- "
                  "skipping the weekly gate")
        else:
            weekly_df = compute_elder_ray(weekly_bars, args.weekly_ema)
            weekly_trend_series = weekly_trend_by_date(df.index, weekly_df, WEEKLY_TREND_LAG)
            weekly_gate_active = True

    rows = []
    n = len(df)
    for i in range(max(0, n - args.lookback), n):
        trend = trend_at(df, i, TREND_LAG)
        prev_row = df.iloc[i - 1] if i > 0 else None
        weekly_trend = weekly_trend_series.iloc[i]
        read = classify_row(df.iloc[i], prev_row, trend, weekly_trend)
        rows.append({
            "date": df.index[i].date(),
            "close": df["Close"].iloc[i],
            f"ema{args.ema}": df["EMA"].iloc[i],
            "bull_power": df["BullPower"].iloc[i],
            "bear_power": df["BearPower"].iloc[i],
            "trend": trend,
            "weekly_trend": weekly_trend if weekly_trend is not None else "n/a",
            "read": read,
        })

    table = pd.DataFrame(rows)
    table_display = table.copy()
    for col in ["close", f"ema{args.ema}", "bull_power", "bear_power"]:
        table_display[col] = table_display[col].map(lambda x: f"{x:.2f}")

    print(f"{args.ticker}: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}, "
          f"EMA period={args.ema}")
    if weekly_gate_active:
        print(f"Weekly trend gate: point-in-time, last closed weekly bar as of each date "
              f"(EMA-{args.weekly_ema})")
    elif args.no_weekly_gate:
        print("Weekly trend gate: disabled (--no-weekly-gate)")
    else:
        print("Weekly trend gate: skipped (not enough weekly history, see warning above)")
    print()
    print(tabulate(table_display, headers="keys", tablefmt="github", showindex=False))

    last = rows[-1]
    print()
    print(f"Latest session ({last['date']}): {last['read']}")
    print("Elder's prescription: buy/hold if bulls are clearly stronger, sell/avoid if bears are "
          "clearly stronger, stand aside if neither side has a clear edge.")


if __name__ == "__main__":
    main()
