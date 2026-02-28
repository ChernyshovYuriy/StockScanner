"""
Canadian Stock Screener — Multi-Factor Momentum Scoring Engine
==============================================================
Detects top 10 stocks from a predefined TSX universe for maximum gain potential.

Strategy Stack:
  1. Weinstein Stage II Detection         (trending upward, not extended)
  2. RS Relative Strength vs XIU          (beating the benchmark)
  3. MACD Bullish Momentum                (trend confirmation)
  4. Volume Accumulation (OBV slope)      (smart money footprint)
  5. ADX Trend Strength                   (strong directional moves)
  6. Volatility-Adjusted Momentum (VAM)   (momentum per unit of risk)
  7. 52-Week High Proximity               (near breakout zone)

Usage:
  pip install yfinance pandas numpy scipy ta colorama tabulate
  python canadian_stock_screener.py
"""

import warnings

from config import SCREENER_OUTPUT, CAN_TICKERS_PATH

warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import linregress
from tabulate import tabulate
from colorama import Fore, Style, init
import time

init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE  (~200 TSX tickers — extend as needed)
# ─────────────────────────────────────────────────────────────────────────────
with open(CAN_TICKERS_PATH, "r") as f:
    TSX_UNIVERSE = [line.strip() for line in f if line.strip()]

# Benchmark ETF (TSX 60)
BENCHMARK = "XIU.TO"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "lookback_days": 504,  # ~2 years of data
    "top_n": 10,  # Final picks
    "min_avg_volume": 100_000,  # Minimum avg daily volume (liquidity filter)
    "min_price": 2.0,  # Min price filter (avoid penny stocks)
    "weights": {
        "stage2_score": 0.20,  # Weinstein Stage II alignment
        "rs_score": 0.20,  # Relative strength vs benchmark
        "macd_score": 0.15,  # MACD momentum
        "obv_score": 0.15,  # Volume accumulation
        "adx_score": 0.10,  # Trend strength
        "vam_score": 0.10,  # Volatility-adjusted momentum
        "breakout_score": 0.10,  # 52-week high proximity / breakout
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — uses EWM smoothing (alpha=1/period), matching standard charting platforms."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram)"""
    m = ema(series, fast) - ema(series, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    pos = np.where((up > down) & (up > 0), up, 0.0)
    neg = np.where((down > up) & (down > 0), down, 0.0)
    pdi = 100 * pd.Series(pos, index=high.index).ewm(span=period, adjust=False).mean() / atr
    ndi = 100 * pd.Series(neg, index=high.index).ewm(span=period, adjust=False).mean() / atr
    dx = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan))
    return dx.ewm(span=period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def linear_slope_r2(series: pd.Series) -> tuple:
    """Normalized slope and R² of a series (for trend quality)"""
    y = series.dropna().values
    if len(y) < 10:
        return 0.0, 0.0
    x = np.arange(len(y))
    sl, inter, r, *_ = linregress(x, y)
    norm_slope = sl / (np.mean(y) + 1e-9)
    return norm_slope, r ** 2


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS  (each returns 0-100)
# ─────────────────────────────────────────────────────────────────────────────

def score_stage2(close: pd.Series) -> float:
    """
    Weinstein Stage II: Price > 30w SMA AND 30w SMA rising AND 
    price not too extended above 30w SMA (not Stage III yet).
    """
    ma30w = sma(close, 150)  # ~30 weeks
    ma10w = sma(close, 50)
    last = close.iloc[-1]
    ma30 = ma30w.iloc[-1]
    ma10 = ma10w.iloc[-1]

    if pd.isna(ma30) or pd.isna(ma10) or ma30 == 0:
        return 0.0

    # 1. Is MA30 rising? (slope over last 10 bars)
    slope, r2 = linear_slope_r2(ma30w.iloc[-20:])
    ma_rising = slope > 0 and r2 > 0.5

    # 2. Price relationship to MA
    pct_above = (last - ma30) / ma30 * 100

    if not ma_rising or last < ma30:
        return 0.0

    # Penalize if too extended (Stage III risk).
    # Linear ramp: 0% above MA → 30, 25% above MA → 100, beyond 25% → discounted.
    if pct_above <= 25:
        stage_score = 30.0 + pct_above * (70.0 / 25.0)
    else:
        # Extended — discount linearly beyond 25%
        stage_score = max(0, 100 - (pct_above - 25) * 3)

    # Bonus: MA10 > MA30 (short MA above long MA)
    if ma10 > ma30:
        stage_score = min(100, stage_score + 10)

    return float(stage_score)


def score_relative_strength(close: pd.Series, bench: pd.Series,
                            periods=(20, 60, 120)) -> float:
    """
    Mansfield Relative Strength: stock return vs benchmark return
    across multiple timeframes, weighted toward recent.
    Weights are re-normalized if any period fails, so the score is always
    correctly scaled to 0-100.
    """
    raw_weights = [0.5, 0.3, 0.2]
    weighted_scores = []
    successful_weights = []
    for p, w in zip(periods, raw_weights):
        try:
            aligned = close.align(bench, join="inner")
            s_ret = aligned[0].iloc[-1] / aligned[0].iloc[-p] - 1
            b_ret = aligned[1].iloc[-1] / aligned[1].iloc[-p] - 1
            rs = (s_ret - b_ret) * 100  # RS in percentage points
            # Normalize: >+20pp = 100, 0pp = 50, <-20pp = 0
            norm = np.clip(50 + rs * 2.5, 0, 100)
            weighted_scores.append(norm * w)
            successful_weights.append(w)
        except Exception:
            pass
    if not weighted_scores:
        return 0.0
    total_weight = sum(successful_weights)
    return float(sum(weighted_scores) / total_weight)


def score_macd(close: pd.Series) -> float:
    """
    MACD signal: histogram trend + zero-line cross + momentum acceleration.
    Bidirectional: bullish conditions add to 50, bearish subtract from 50.
    """
    m_line, sig_line, hist = macd(close)
    if hist.dropna().__len__() < 5:
        return 0.0

    last_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    m_val = m_line.iloc[-1]
    sig_val = sig_line.iloc[-1]

    score = 50.0  # Neutral start

    # MACD line vs signal: +20 if bullish, -20 if bearish
    if m_val > sig_val:
        score += 20
    else:
        score -= 20

    # Histogram direction (momentum accelerating/decelerating): +15 / -15
    if last_hist > prev_hist:
        score += 15
    else:
        score -= 15

    # Histogram sign (above/below zero): +10 / -10
    if last_hist > 0:
        score += 10
    else:
        score -= 10

    # Recent bullish crossover bonus (within last 5 bars)
    crossover = ((hist.iloc[-5:] > 0) & (hist.shift().iloc[-5:] < 0)).any()
    if crossover:
        score += 5

    return float(np.clip(score, 0, 100))


def score_obv(close: pd.Series, volume: pd.Series) -> float:
    """
    OBV trend: rising OBV = accumulation (smart money buying).
    Compare OBV slope to price slope — divergences matter.
    """
    ob = obv(close, volume)
    if ob.dropna().__len__() < 20:
        return 0.0

    obv_slope, obv_r2 = linear_slope_r2(ob.iloc[-60:])
    price_slope, _ = linear_slope_r2(close.iloc[-60:])

    score = 50.0

    if obv_slope > 0:
        score += 25 * obv_r2  # OBV rising and consistent

    if obv_slope > 0 and price_slope > 0:
        score += 15  # Confirmed uptrend

    # OBV leading price (OBV making new highs faster)
    ob_norm = ob.iloc[-20:] / (ob.abs().max() + 1e-9)
    c_norm = close.iloc[-20:] / (close.abs().max() + 1e-9)
    diff = (ob_norm - c_norm).diff().mean()
    if diff > 0:
        score += 10

    return float(np.clip(score, 0, 100))


def score_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """ADX > 25 signals strong trend; rising ADX = strengthening trend"""
    adx_val = adx(high, low, close)
    if adx_val.dropna().__len__() < 5:
        return 0.0

    last_adx = adx_val.iloc[-1]
    prev_adx = adx_val.iloc[-5]

    score = 0.0
    if last_adx >= 40:
        score = 100
    elif last_adx >= 25:
        score = 60 + (last_adx - 25) * 2.67
    elif last_adx >= 20:
        score = 40 + (last_adx - 20) * 4
    else:
        score = last_adx * 2

    # Bonus: ADX is rising
    if last_adx > prev_adx:
        score = min(100, score + 10)

    return float(score)


def score_vam(close: pd.Series, periods=(20, 60)) -> float:
    """
    Volatility-Adjusted Momentum: annualized return / annualized vol.
    Both numerator and denominator are annualized for a true Sharpe-like ratio.
    """
    scores = []
    for p in periods:
        if len(close) < p + 1:
            continue
        rets = close.pct_change().dropna()
        raw_ret = close.iloc[-1] / close.iloc[-p] - 1
        # Annualize the return to match the annualized volatility
        ann_ret = (1 + raw_ret) ** (252 / p) - 1
        ann_vol = rets.iloc[-p:].std() * np.sqrt(252)
        if ann_vol < 0.001:
            continue
        vam = ann_ret / ann_vol
        # Normalize: vam of 1.0 = 100 (very high), 0 = 50, -1 = 0
        norm = np.clip(50 + vam * 50, 0, 100)
        scores.append(norm)
    return float(np.mean(scores)) if scores else 50.0


def score_breakout(close: pd.Series, high: pd.Series,
                   volume: pd.Series) -> float:
    """
    52-week high proximity and volume-confirmed breakout detection.
    Near a 52-week high with high volume = bullish.
    """
    if len(close) < 252:
        return 50.0

    high_52w = high.iloc[-252:].max()
    current = close.iloc[-1]
    pct_from_52 = (high_52w - current) / high_52w * 100  # 0 = AT high

    avg_vol = volume.iloc[-50:].mean()
    recent_vol = volume.iloc[-5:].mean()
    vol_ratio = recent_vol / (avg_vol + 1e-9)

    # Score by proximity to 52w high
    if pct_from_52 <= 2:  # At or near 52-week high
        score = 90
    elif pct_from_52 <= 5:
        score = 75
    elif pct_from_52 <= 10:
        score = 60
    elif pct_from_52 <= 20:
        score = 40
    else:
        score = max(0, 40 - (pct_from_52 - 20) * 1.5)

    # Volume surge bonus (breakout on volume)
    if vol_ratio > 1.5:
        score = min(100, score + 10)

    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_data(tickers: list, days: int) -> dict:
    end = datetime.today()
    start = end - timedelta(days=days + 60)  # extra buffer for MA warmup

    print(f"\n{Fore.CYAN}Downloading data for {len(tickers)} tickers...{Style.RESET_ALL}")
    data = {}

    # Download in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {batch[0]} ... {batch[-1]}")
        try:
            raw = yf.download(
                batch, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False, threads=True
            )
            for ticker in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df = pd.DataFrame({
                            "Open": raw["Open"][ticker],
                            "High": raw["High"][ticker],
                            "Low": raw["Low"][ticker],
                            "Close": raw["Close"][ticker],
                            "Volume": raw["Volume"][ticker],
                        }).dropna()
                    else:
                        # Single-ticker batch: flat DataFrame belongs only to batch[0]
                        if ticker != batch[0]:
                            continue
                        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
                    if len(df) > 100:
                        data[ticker] = df
                except Exception:
                    pass
        except Exception as e:
            print(f"  {Fore.RED}Batch error: {e}{Style.RESET_ALL}")
        time.sleep(0.5)

    print(f"  {Fore.GREEN}✓ Loaded {len(data)} tickers successfully{Style.RESET_ALL}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(ticker: str, df: pd.DataFrame,
                  bench_close: pd.Series, config: dict) -> dict:
    """Compute composite score for a single ticker"""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    w = config["weights"]

    # Liquidity / Price filters
    avg_vol = volume.iloc[-20:].mean()
    last_price = close.iloc[-1]
    if avg_vol < config["min_avg_volume"] or last_price < config["min_price"]:
        return None

    try:
        s2 = score_stage2(close)
        rs = score_relative_strength(close, bench_close)
        mac = score_macd(close)
        ob = score_obv(close, volume)
        adxs = score_adx(high, low, close)
        vams = score_vam(close)
        brk = score_breakout(close, high, volume)

        composite = (
                s2 * w["stage2_score"] +
                rs * w["rs_score"] +
                mac * w["macd_score"] +
                ob * w["obv_score"] +
                adxs * w["adx_score"] +
                vams * w["vam_score"] +
                brk * w["breakout_score"]
        )

        # ── Momentum context (for display) ──────────────────────────────────
        rsi_val = rsi(close).iloc[-1]
        ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else np.nan
        ma200 = sma(close, 200).iloc[-1]
        ma50 = sma(close, 50).iloc[-1]
        trend_tag = "▲ Stage II" if s2 > 60 else ("▶ Early" if s2 > 30 else "▼ Weak")

        return {
            "Ticker": ticker,
            "Price (CAD)": round(last_price, 2),
            "Score": round(composite, 1),
            "Stage II": round(s2, 1),
            "Rel.Str": round(rs, 1),
            "MACD": round(mac, 1),
            "OBV": round(ob, 1),
            "ADX": round(adxs, 1),
            "VAM": round(vams, 1),
            "Breakout": round(brk, 1),
            "RSI": round(rsi_val, 1),
            "Ret 1M%": round(ret_1m, 2),
            "Ret 3M%": round(ret_3m, 2) if not np.isnan(ret_3m) else "N/A",
            "MA50>MA200": "✓" if ma50 > ma200 else "✗",
            "Trend": trend_tag,
            "Avg Vol": f"{int(avg_vol):,}",
        }
    except Exception:
        return None


def run_screener(config: dict = CONFIG):
    print(f"\n{'=' * 65}")
    print(f"  {Fore.YELLOW}🇨🇦  Canadian Stock Screener — Multi-Factor Momentum Engine{Style.RESET_ALL}")
    print(f"{'=' * 65}")

    # 1. Download all data
    all_tickers = list(set(TSX_UNIVERSE + [BENCHMARK]))
    data = download_data(all_tickers, config["lookback_days"])

    # 2. Extract benchmark
    bench_close = data.get(BENCHMARK, pd.DataFrame())
    if bench_close.empty:
        print(f"{Fore.RED}Benchmark {BENCHMARK} failed to load!{Style.RESET_ALL}")
        return None
    bench_close = bench_close["Close"]

    # 3. Score each ticker
    print(f"\n{Fore.CYAN}Computing multi-factor scores...{Style.RESET_ALL}")
    results = []
    for ticker in TSX_UNIVERSE:
        if ticker not in data:
            continue
        result = compute_score(ticker, data[ticker], bench_close, config)
        if result:
            results.append(result)

    if not results:
        print(f"{Fore.RED}No results computed. Check tickers/data.{Style.RESET_ALL}")
        return None

    # 4. Sort and pick top N
    df_scores = pd.DataFrame(results).sort_values("Score", ascending=False)
    top10 = df_scores.head(config["top_n"])

    # 5. Display results
    display_cols = [
        "Ticker", "Price (CAD)", "Score", "Stage II", "Rel.Str",
        "MACD", "OBV", "ADX", "VAM", "Breakout",
        "RSI", "Ret 1M%", "Ret 3M%", "MA50>MA200", "Trend", "Avg Vol"
    ]
    print(f"\n{'=' * 65}")
    print(f"  {Fore.GREEN}🏆  TOP {config['top_n']} STOCKS TO WATCH{Style.RESET_ALL}")
    print(f"{'=' * 65}\n")
    print(tabulate(
        top10[display_cols],
        headers="keys",
        tablefmt="rounded_outline",
        showindex=False,
        numalign="right"
    ))

    # 6. Score breakdown explanation
    print(f"\n{Fore.YELLOW}Score Weights:{Style.RESET_ALL}")
    for k, v in config["weights"].items():
        print(f"  {k:<18}: {int(v * 100)}%")

    print(f"\n{Fore.YELLOW}Filters Applied:{Style.RESET_ALL}")
    print(f"  Min Avg Volume : {config['min_avg_volume']:,}")
    print(f"  Min Price      : ${config['min_price']}")
    print(f"  Universe Size  : {len(TSX_UNIVERSE)} tickers")
    print(f"  Screened       : {len(results)} (passed filters)")
    print(f"  Run Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    print(f"\n{Fore.RED}⚠  DISCLAIMER: For educational purposes only. "
          f"Not financial advice. Always do your own due diligence.{Style.RESET_ALL}\n")

    return top10


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    top10 = run_screener(CONFIG)

    # Save to CSV
    if top10 is not None:
        name = f"{SCREENER_OUTPUT}/{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        top10.to_csv(name, index=False)
        print(f"{Fore.CYAN}Results saved to {name}{Style.RESET_ALL}")
