"""
Canadian Stock Screener — Multi-Factor Momentum Scoring Engine (Enhanced)
========================================================================
Detects top stocks from TSX universe with robust technical analysis and risk metrics.

Strategy Stack Improvements:
  ✓ Weinstein Stage II Detection (weekly resampled data)
  ✓ Relative Strength percentile ranking across universe
  ✓ MACD with proper signal processing
  ✓ OBV with divergence detection
  ✓ ADX with Wilder's smoothing
  ✓ Volatility-Adjusted Momentum (Sharpe-like)
  ✓ 52-Week High Breakout with volume confirmation
  ✓ Risk metrics (Max DD, Sharpe, Win Rate)

Usage:
  pip install yfinance pandas numpy scipy ta tabulate colorama pydantic
  python canadian_stock_screener_enhanced.py
"""

import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Union

from config import CACHE_PATH
from market_data import DEFAULT_PROVIDER
from time_utils import market_now, date_to_iso_basic_minutes, date_to_iso_basic

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import linregress
from tabulate import tabulate
from colorama import Fore, Style, init
from pandas.tseries.offsets import BDay
from pydantic import BaseModel, validator, Field

init(autoreset=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class ScreenerConfig(BaseModel):
    """Configuration model with validation"""
    lookback_days: int = Field(504, ge=252, le=1000)
    top_n: int = Field(10, ge=1, le=50)
    min_avg_volume: int = Field(100000, ge=10000)
    min_price: float = Field(2.0, ge=0.5)
    min_composite_score: float = Field(60.0, ge=0.0, le=100.0)  # gate: skip mediocre setups
    weights: Dict[str, float]

    @validator('weights')
    def weights_sum_to_one(cls, v):
        if abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError(f'Weights must sum to 1.0, got {sum(v.values())}')
        return v


# Load or create config
CONFIG = ScreenerConfig(
    weights={
        "stage2_score": 0.20,
        "rs_score": 0.20,
        "macd_score": 0.15,
        "obv_score": 0.15,
        "adx_score": 0.10,
        "vam_score": 0.10,
        "breakout_score": 0.10,
    }
)

# Benchmark ETF
BENCHMARK = "XIU.TO"


# ─────────────────────────────────────────────────────────────────────────────
# DATA MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class DataManager:
    """Handles data downloading, caching, and preprocessing"""

    def __init__(self, tickers_source: Union[str, List[str]],
                 cache_dir: str = CACHE_PATH, provider=None):
        """
        Parameters
        ----------
        tickers_source : URL, local file path, or an already-loaded list of
                         ticker strings.
        cache_dir      : directory used for the daily parquet cache (live mode only)
        provider       : optional MarketDataProvider.
                         None  (default) → live mode: uses yfinance + parquet cache,
                         identical to the original behaviour.
                         MarketDataProvider instance → backtest mode: all data comes
                         from the provider; cache and yfinance are bypassed entirely.
        """
        self.tickers = self._load_tickers(tickers_source)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._provider = provider   # None = live mode

    def _load_tickers(self, source: Union[str, List[str]]) -> List[str]:
        """Load tickers from a URL, local file path, or an existing list."""
        if isinstance(source, list):
            return [t.strip() for t in source if t.strip()]
        if source.startswith("http://") or source.startswith("https://"):
            try:
                with urllib.request.urlopen(source) as resp:
                    content = resp.read().decode("utf-8")
                return [ln.strip() for ln in content.splitlines()
                        if ln.strip() and not ln.strip().startswith("#")]
            except Exception as e:
                print(f"{Fore.RED}Error fetching tickers from {source}: {e}. Using fallback tickers.{Style.RESET_ALL}")
                return ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO",
                        "ENB.TO", "SU.TO", "CNQ.TO", "CP.TO", "CNR.TO"]
        try:
            with open(source, "r") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"{Fore.RED}Error: {source} not found. Using fallback tickers.{Style.RESET_ALL}")
            return ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO",
                    "ENB.TO", "SU.TO", "CNQ.TO", "CP.TO", "CNR.TO"]

    def download_data(self, days: int, force_refresh: bool = False,
                      as_of=None) -> Dict[str, pd.DataFrame]:
        """Download data with caching.

        Parameters
        ----------
        days          : lookback window in calendar days
        force_refresh : ignore on-disk cache and re-download (live mode only)
        as_of         : backtest cutoff date (pd.Timestamp).  Passed through to
                        the provider when self._provider is set; ignored in live mode.
        """
        all_tickers = list(set(self.tickers + [BENCHMARK]))

        # ── Backtest mode: delegate entirely to the injected provider ─────────
        if self._provider is not None:
            return self._provider.download(all_tickers, days=days, as_of=as_of)
        cache_file = self.cache_dir / f"data_{date_to_iso_basic(market_now())}.parquet"

        # Try to load from cache
        if not force_refresh and cache_file.exists():
            try:
                print(f"{Fore.CYAN}Loading data from cache...{Style.RESET_ALL}")
                return pd.read_parquet(cache_file).to_dict()
            except Exception:
                pass

        # Download fresh data — via the shared DEFAULT_PROVIDER (LiveDataProvider)
        # instead of an inline yf.download batch loop, so live fetches go
        # through the same batching/quality-gate/cache-free path every other
        # live call site uses (see market_data.py).
        print(f"\n{Fore.CYAN}Downloading data for {len(all_tickers)} tickers...{Style.RESET_ALL}")
        data = DEFAULT_PROVIDER.download(all_tickers, days=days)
        failed_tickers = [t for t in all_tickers if t not in data]

        # Save to cache
        try:
            pd.DataFrame(data).to_parquet(cache_file)
        except Exception:
            pass

        if failed_tickers:
            print(f"  {Fore.YELLOW}Failed to load: {', '.join(failed_tickers[:5])}"
                  f"{'...' if len(failed_tickers) > 5 else ''}{Style.RESET_ALL}")

        print(f"  {Fore.GREEN}✓ Loaded {len(data)}/{len(all_tickers)} tickers successfully{Style.RESET_ALL}")
        return data


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

class TechnicalIndicators:
    """Stateless technical indicator calculations"""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI with proper smoothing"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))

        # Wilder's smoothing (equivalent to EMA with alpha=1/period)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        """MACD with proper alignment"""
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index with Wilder's smoothing"""
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing of TR
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth directional movements
        pos_dm_smooth = pd.Series(pos_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
        neg_dm_smooth = pd.Series(neg_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()

        # Directional Indicators
        pdi = 100 * pos_dm_smooth / atr
        ndi = 100 * neg_dm_smooth / atr

        # Directional Index
        dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)

        # ADX is smoothed DX
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume with proper handling"""
        price_change = close.diff()
        direction = np.sign(price_change)
        # No change days don't affect OBV
        direction[price_change == 0] = 0
        return (direction * volume).cumsum()

    @staticmethod
    def linear_regression_slope(series: pd.Series, period: int) -> float:
        """Calculate linear regression slope over period"""
        y = series.iloc[-period:].values
        if len(y) < 10:
            return 0.0
        x = np.arange(len(y))
        slope, _, r_value, _, _ = linregress(x, y)
        # Normalize slope by mean price
        norm_slope = slope / (np.mean(y) + 1e-9)
        return norm_slope

    @staticmethod
    def weekly_resample(daily_series: pd.Series) -> pd.Series:
        """Resample daily data to weekly (end of week)"""
        return daily_series.resample('W-FRI').last()


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCalculator:
    """Calculates all scoring components"""

    def __init__(self, config: ScreenerConfig):
        self.config = config
        self.ti = TechnicalIndicators()
        self.rs_scores_cache = {}  # For percentile ranking

    def score_stage2(self, close: pd.Series) -> float:
        """
        Enhanced Weinstein Stage II detection using weekly data
        """
        # Resample to weekly
        weekly_close = self.ti.weekly_resample(close)

        if len(weekly_close) < 40:
            return 0.0

        # Calculate weekly moving averages
        ma30w = self.ti.sma(weekly_close, 30)
        ma10w = self.ti.sma(weekly_close, 10)

        last_price = weekly_close.iloc[-1]
        last_ma30 = ma30w.iloc[-1]
        last_ma10 = ma10w.iloc[-1]

        if pd.isna(last_ma30) or pd.isna(last_ma10) or last_ma30 == 0:
            return 0.0

        # Check if MA30 is rising (slope over last 10 weeks)
        ma30_slope = self.ti.linear_regression_slope(ma30w.dropna(), 10)
        ma_rising = ma30_slope > 0.001

        # Check price position relative to MA30
        price_above_ma30 = last_price > last_ma30

        if not (ma_rising and price_above_ma30):
            return 0.0

        # Calculate how extended (percentage above MA30)
        pct_above = (last_price - last_ma30) / last_ma30 * 100

        # Score based on extension (Weinstein: 5-25% is sweet spot)
        if pct_above <= 5:
            stage_score = 30.0  # Just entering stage II
        elif pct_above <= 15:
            stage_score = 50.0 + (pct_above - 5) * 3  # 50 to 80
        elif pct_above <= 25:
            stage_score = 80.0 + (pct_above - 15) * 2  # 80 to 100
        else:
            stage_score = max(0, 100 - (pct_above - 25) * 4)  # Penalize extended

        # Bonus for MA10 > MA30 (strong trend)
        if last_ma10 > last_ma30:
            stage_score = min(100, stage_score + 10)

        # Bonus for price > MA10 (very strong)
        if last_price > last_ma10:
            stage_score = min(100, stage_score + 5)

        return float(stage_score)

    def score_relative_strength(self, close: pd.Series, bench: pd.Series,
                                all_rs_values: List[float] = None) -> float:
        """
        Enhanced relative strength using percentile ranking across universe
        """

        def calculate_rs(close_series, bench_series, period_days):
            """Calculate RS for a specific period"""
            try:
                # Align dates properly
                common_dates = close_series.index.intersection(bench_series.index)
                if len(common_dates) < period_days:
                    return None

                c_aligned = close_series.loc[common_dates]
                b_aligned = bench_series.loc[common_dates]

                # Get returns over exact business day period
                end_date = common_dates[-1]
                start_date = end_date - BDay(period_days)
                start_idx = common_dates.get_indexer([start_date], method='nearest')[0]

                if start_idx < 0:
                    return None

                c_ret = c_aligned.iloc[-1] / c_aligned.iloc[start_idx] - 1
                b_ret = b_aligned.iloc[-1] / b_aligned.iloc[start_idx] - 1

                return (c_ret - b_ret) * 100  # RS in percentage points

            except Exception:
                return None

        # Calculate RS for multiple periods
        periods = [20, 60, 120]
        weights = [0.5, 0.3, 0.2]

        weighted_rs = 0
        total_weight = 0

        for period, weight in zip(periods, weights):
            rs = calculate_rs(close, bench, period)
            if rs is not None:
                weighted_rs += rs * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0

        avg_rs = weighted_rs / total_weight

        # If we have universe RS values, use percentile ranking
        if all_rs_values and len(all_rs_values) > 10:
            percentile = stats.percentileofscore(all_rs_values, avg_rs)
            return float(percentile)
        else:
            # Fallback to linear scaling
            return float(np.clip(50 + avg_rs * 2.5, 0, 100))

    def score_macd(self, close: pd.Series) -> float:
        """Enhanced MACD scoring with multiple confirmations"""
        macd_line, signal_line, histogram = self.ti.macd(close)

        if histogram.dropna().empty:
            return 50.0

        # Get recent values
        last_macd = macd_line.iloc[-1]
        last_signal = signal_line.iloc[-1]
        last_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0

        # Recent histogram values for slope
        hist_slope = self.ti.linear_regression_slope(histogram.iloc[-5:], 5)

        score = 50.0  # Neutral start

        # MACD line vs signal (trend direction)
        if last_macd > last_signal:
            score += 20
        else:
            score -= 20

        # Histogram slope (momentum acceleration)
        if hist_slope > 0.001:
            score += 15
        elif hist_slope < -0.001:
            score -= 15

        # Zero line position
        if last_hist > 0:
            score += 10
        else:
            score -= 10

        # Histogram increasing (short-term momentum)
        if last_hist > prev_hist:
            score += 5
        else:
            score -= 5

        # Bullish cross in last 10 days
        recent_cross = ((histogram.iloc[-10:] > 0) &
                        (histogram.shift().iloc[-10:] < 0)).any()
        if recent_cross:
            score += 10

        return float(np.clip(score, 0, 100))

    def score_obv(self, close: pd.Series, volume: pd.Series) -> float:
        """Enhanced OBV scoring with divergence detection"""
        obv = self.ti.obv(close, volume)

        if len(obv) < 60:
            return 50.0

        # Calculate slopes over different periods
        obv_slope_20 = self.ti.linear_regression_slope(obv, 20)
        obv_slope_60 = self.ti.linear_regression_slope(obv, 60)
        price_slope_60 = self.ti.linear_regression_slope(close, 60)

        score = 50.0

        # OBV trend
        if obv_slope_60 > 0.001:
            score += 20
        elif obv_slope_60 < -0.001:
            score -= 20

        # Recent momentum
        if obv_slope_20 > obv_slope_60:
            score += 15  # Acceleration
        elif obv_slope_20 < obv_slope_60:
            score -= 15  # Deceleration

        # Positive divergence (price making lower lows, OBV making higher lows)
        price_low_60 = close.iloc[-60:].min()
        price_low_20 = close.iloc[-20:].min()
        obv_low_60 = obv.iloc[-60:].min()
        obv_low_20 = obv.iloc[-20:].min()

        if price_low_20 < price_low_60 and obv_low_20 > obv_low_60:
            score += 20  # Strong positive divergence

        # Negative divergence (price making higher highs, OBV making lower highs)
        price_high_60 = close.iloc[-60:].max()
        price_high_20 = close.iloc[-20:].max()
        obv_high_60 = obv.iloc[-60:].max()
        obv_high_20 = obv.iloc[-20:].max()

        if price_high_20 > price_high_60 and obv_high_20 < obv_high_60:
            score -= 20  # Strong negative divergence

        # Volume confirmation
        if price_slope_60 > 0 and obv_slope_60 > 0:
            score += 10

        return float(np.clip(score, 0, 100))

    def score_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Enhanced ADX scoring"""
        adx_values = self.ti.adx(high, low, close)

        if adx_values.dropna().empty:
            return 0.0

        last_adx = adx_values.iloc[-1]
        adx_slope = self.ti.linear_regression_slope(adx_values.iloc[-10:], 10)

        # Base score on ADX level
        if last_adx >= 40:
            score = 100
        elif last_adx >= 25:
            score = 60 + (last_adx - 25) * 2.67
        elif last_adx >= 20:
            score = 40 + (last_adx - 20) * 4
        else:
            score = last_adx * 2

        # Trend strength (rising ADX)
        if adx_slope > 0.1:
            score = min(100, score + 15)
        elif adx_slope < -0.1:
            score = max(0, score - 15)

        return float(score)

    def score_vam(self, close: pd.Series) -> float:
        """
        Volatility-Adjusted Momentum (Sharpe ratio over multiple periods)
        """
        returns = close.pct_change().dropna()

        if len(returns) < 60:
            return 50.0

        periods = [20, 60, 120]
        weights = [0.5, 0.3, 0.2]

        vam_scores = []
        total_weight = 0

        for period, weight in zip(periods, weights):
            if len(returns) < period:
                continue

            # Period returns
            period_returns = returns.iloc[-period:]

            # Annualized return
            total_ret = (1 + period_returns).prod() - 1
            ann_ret = (1 + total_ret) ** (252 / period) - 1

            # Annualized volatility
            ann_vol = period_returns.std() * np.sqrt(252)

            if ann_vol < 0.001 or np.isnan(ann_vol):
                continue

            # Sharpe ratio
            sharpe = ann_ret / ann_vol

            # Normalize: Sharpe of 2 = 100, 0 = 50, -2 = 0
            norm_sharpe = np.clip(50 + sharpe * 25, 0, 100)
            vam_scores.append(norm_sharpe * weight)
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return float(sum(vam_scores) / total_weight)

    def score_breakout(self, close: pd.Series, high: pd.Series,
                       volume: pd.Series) -> float:
        """
        Enhanced breakout detection with volume confirmation
        """
        if len(close) < 252:
            return 50.0

        # 52-week high (252 trading days)
        high_52w = high.iloc[-252:].max()
        current = close.iloc[-1]

        # Distance from 52-week high (0 = at high)
        pct_from_high = (high_52w - current) / high_52w * 100

        # Volume analysis
        avg_vol_50 = volume.iloc[-50:].mean()
        avg_vol_10 = volume.iloc[-10:].mean()
        vol_ratio = avg_vol_10 / (avg_vol_50 + 1e-9)

        # Recent volume spike
        recent_vol = volume.iloc[-5:].mean()
        vol_spike = recent_vol / (avg_vol_50 + 1e-9)

        # Breakout detection
        recent_highs = high.iloc[-20:]
        new_high_count = sum(recent_highs > high_52w * 0.95)

        # Score based on proximity to 52-week high
        if pct_from_high <= 2:  # At or near 52-week high
            score = 90
        elif pct_from_high <= 5:
            score = 75
        elif pct_from_high <= 10:
            score = 60
        elif pct_from_high <= 20:
            score = 40
        else:
            score = max(0, 40 - (pct_from_high - 20) * 1.5)

        # Volume confirmation bonus
        if vol_ratio > 1.5:
            score = min(100, score + 10)

        # Volume spike bonus (breakout confirmation)
        if vol_spike > 2.0:
            score = min(100, score + 15)

        # Multiple new highs bonus
        if new_high_count >= 3:
            score = min(100, score + 10)

        return float(score)

    def calculate_risk_metrics(self, close: pd.Series) -> Dict:
        """
        Calculate comprehensive risk metrics for a stock
        """
        returns = close.pct_change().dropna()

        if len(returns) < 60:
            return {}

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns)

        # Average win/loss
        avg_win = returns[returns > 0].mean() if any(returns > 0) else 0
        avg_loss = abs(returns[returns < 0].mean()) if any(returns < 0) else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else np.inf

        # Skewness and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Value at Risk (95%)
        var_95 = returns.quantile(0.05)

        # Calmar ratio (return / max drawdown)
        total_return = (close.iloc[-1] / close.iloc[0]) - 1
        calmar = abs(total_return / max_drawdown) if max_drawdown < 0 else 0

        return {
            "Max_DD": round(max_drawdown * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Win_Rate": round(win_rate * 100, 1),
            "Profit_Factor": round(profit_factor, 2) if profit_factor != np.inf else "Inf",
            "Skew": round(skew, 2),
            "Kurtosis": round(kurt, 2),
            "VaR_95": round(var_95 * 100, 2),
            "Calmar": round(calmar, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCREENING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StockResult:
    """Structured result for a screened stock"""
    ticker: str
    price: float
    composite_score: float
    stage2_score: float
    rs_score: float
    macd_score: float
    obv_score: float
    adx_score: float
    vam_score: float
    breakout_score: float
    rsi: float
    ret_1m: float
    ret_3m: float
    ma50_gt_ma200: bool
    trend_tag: str
    avg_volume: int
    risk_metrics: Dict
    last_close: float


class StockScreener:
    """Main screening engine"""

    def __init__(self, config: ScreenerConfig, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.score_calc = ScoreCalculator(config)
        self.ti = TechnicalIndicators()

    def _calculate_trend_tag(self, stage2_score: float, ma50: float,
                             ma200: float, last_price: float) -> str:
        """Generate trend description"""
        if stage2_score > 70 and last_price > ma50 > ma200:
            return "▲ Strong Stage II"
        elif stage2_score > 50:
            return "▶ Stage II"
        elif stage2_score > 30:
            return "→ Early Stage II"
        elif last_price > ma200:
            return "▼ Above 200MA"
        else:
            return "▼ Weak"

    def _collect_rs_values(self, data: Dict[str, pd.DataFrame],
                           bench_close: pd.Series) -> List[float]:
        """Collect RS values for percentile ranking"""
        rs_values = []

        for ticker, df in data.items():
            if ticker == BENCHMARK:
                continue

            try:
                # Quick RS calculation for ranking
                close = df["Close"]
                common_dates = close.index.intersection(bench_close.index)
                if len(common_dates) < 120:
                    continue

                c_aligned = close.loc[common_dates]
                b_aligned = bench_close.loc[common_dates]

                # 60-day RS
                ret_60d = c_aligned.iloc[-1] / c_aligned.iloc[-60] - 1
                bench_60d = b_aligned.iloc[-1] / b_aligned.iloc[-60] - 1
                rs = (ret_60d - bench_60d) * 100
                rs_values.append(rs)

            except Exception:
                continue

        return rs_values

    def analyze_stock(self, ticker: str, df: pd.DataFrame,
                      bench_close: pd.Series, rs_universe: List[float]) -> Optional[StockResult]:
        """Analyze a single stock and return structured result"""

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Liquidity and price filters
        avg_vol = volume.iloc[-20:].mean()
        last_price = close.iloc[-1]

        if avg_vol < self.config.min_avg_volume or last_price < self.config.min_price:
            return None

        try:
            # Calculate all scores
            s2 = self.score_calc.score_stage2(close)
            rs = self.score_calc.score_relative_strength(close, bench_close, rs_universe)
            mac = self.score_calc.score_macd(close)
            ob = self.score_calc.score_obv(close, volume)
            adxs = self.score_calc.score_adx(high, low, close)
            vams = self.score_calc.score_vam(close)
            brk = self.score_calc.score_breakout(close, high, volume)

            # Composite score
            w = self.config.weights
            composite = (
                    s2 * w["stage2_score"] +
                    rs * w["rs_score"] +
                    mac * w["macd_score"] +
                    ob * w["obv_score"] +
                    adxs * w["adx_score"] +
                    vams * w["vam_score"] +
                    brk * w["breakout_score"]
            )

            # Technical metrics
            rsi_val = self.ti.rsi(close).iloc[-1]
            ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan
            ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else np.nan

            ma200 = self.ti.sma(close, 200).iloc[-1]
            ma50 = self.ti.sma(close, 50).iloc[-1]
            ma50_gt_ma200 = ma50 > ma200 if not (pd.isna(ma50) or pd.isna(ma200)) else False

            trend_tag = self._calculate_trend_tag(s2, ma50, ma200, last_price)

            # Risk metrics
            risk_metrics = self.score_calc.calculate_risk_metrics(close)

            return StockResult(
                ticker=ticker,
                price=round(last_price, 2),
                composite_score=round(composite, 1),
                stage2_score=round(s2, 1),
                rs_score=round(rs, 1),
                macd_score=round(mac, 1),
                obv_score=round(ob, 1),
                adx_score=round(adxs, 1),
                vam_score=round(vams, 1),
                breakout_score=round(brk, 1),
                rsi=round(rsi_val, 1),
                ret_1m=round(ret_1m, 2) if not np.isnan(ret_1m) else 0,
                ret_3m=round(ret_3m, 2) if not np.isnan(ret_3m) else 0,
                ma50_gt_ma200=ma50_gt_ma200,
                trend_tag=trend_tag,
                avg_volume=int(avg_vol),
                risk_metrics=risk_metrics,
                last_close=last_price
            )

        except Exception as e:
            print(f"{Fore.RED}Error analyzing {ticker}: {str(e)[:50]}{Style.RESET_ALL}")
            return None

    def run(self, force_refresh: bool = False) -> pd.DataFrame:
        """Run the full screening process"""

        print(f"\n{'=' * 75}")
        print(f"  {Fore.YELLOW}🇨🇦  Enhanced Canadian Stock Screener — Multi-Factor Momentum{Style.RESET_ALL}")
        print(f"{'=' * 75}")

        # Download data
        data = self.data_manager.download_data(self.config.lookback_days, force_refresh)

        # Extract benchmark
        if BENCHMARK not in data:
            print(f"{Fore.RED}Benchmark {BENCHMARK} failed to load!{Style.RESET_ALL}")
            return pd.DataFrame()

        bench_close = data[BENCHMARK]["Close"]

        # Collect RS values for percentile ranking
        print(f"\n{Fore.CYAN}Calculating universe RS distribution...{Style.RESET_ALL}")
        rs_universe = self._collect_rs_values(data, bench_close)
        print(f"  Collected {len(rs_universe)} RS values for percentile ranking")

        # Analyze each stock
        print(f"\n{Fore.CYAN}Analyzing {len(self.data_manager.tickers)} stocks...{Style.RESET_ALL}")
        results = []

        for i, ticker in enumerate(self.data_manager.tickers, 1):
            if ticker not in data:
                continue

            if i % 20 == 0:
                print(f"  Progress: {i}/{len(self.data_manager.tickers)}")

            result = self.analyze_stock(ticker, data[ticker], bench_close, rs_universe)
            if result:
                results.append(result)

        if not results:
            print(f"{Fore.RED}No results computed. Check tickers/data.{Style.RESET_ALL}")
            return pd.DataFrame()

        # Convert to DataFrame and sort
        df_results = pd.DataFrame([r.__dict__ for r in results])
        df_results = df_results.sort_values("composite_score", ascending=False)

        # Extract risk metrics to separate columns
        for metric in ["Max_DD", "Sharpe", "Win_Rate", "Profit_Factor", "Skew", "Kurtosis", "VaR_95", "Calmar"]:
            df_results[metric] = df_results["risk_metrics"].apply(lambda x: x.get(metric, "N/A"))

        df_results = df_results.drop("risk_metrics", axis=1)

        # Apply minimum composite score gate — drop mediocre setups
        before_gate = len(df_results)
        df_results = df_results[df_results["composite_score"] >= self.config.min_composite_score]
        dropped = before_gate - len(df_results)
        if dropped:
            print(f"  {Fore.YELLOW}Score gate ({self.config.min_composite_score:.0f}): "
                  f"dropped {dropped} ticker(s) below threshold{Style.RESET_ALL}")

        return df_results


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY AND OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def display_results(df_results: pd.DataFrame, config: ScreenerConfig,
                    data_manager: DataManager):
    """Display screening results in formatted tables"""

    if df_results.empty:
        return

    top_n = df_results.head(config.top_n)

    # Main results table
    display_cols = [
        "ticker", "price", "composite_score", "stage2_score", "rs_score",
        "macd_score", "obv_score", "adx_score", "vam_score", "breakout_score",
        "rsi", "ret_1m", "ret_3m", "trend_tag", "avg_volume"
    ]

    col_names = [
        "Ticker", "Price", "Score", "Stage II", "Rel.Str",
        "MACD", "OBV", "ADX", "VAM", "Breakout",
        "RSI", "Ret 1M%", "Ret 3M%", "Trend", "Volume"
    ]

    print(f"\n{'=' * 75}")
    print(f"  {Fore.GREEN}🏆  TOP {config.top_n} STOCKS TO WATCH{Style.RESET_ALL}")
    print(f"{'=' * 75}\n")

    # Format volume with commas
    display_data = top_n[display_cols].copy()
    display_data["avg_volume"] = display_data["avg_volume"].apply(lambda x: f"{x:,}")
    display_data.columns = col_names

    print(tabulate(
        display_data,
        headers="keys",
        tablefmt="rounded_outline",
        showindex=False,
        numalign="right"
    ))

    # Risk metrics table
    risk_cols = ["ticker", "Max_DD", "Sharpe", "Win_Rate", "Profit_Factor", "Calmar"]
    risk_data = top_n[risk_cols].copy()
    risk_data.columns = ["Ticker", "Max DD%", "Sharpe", "Win Rate%", "Profit Factor", "Calmar"]

    print(f"\n{Fore.YELLOW}📊  RISK METRICS{Style.RESET_ALL}\n")
    print(tabulate(
        risk_data,
        headers="keys",
        tablefmt="rounded_outline",
        showindex=False,
        numalign="right"
    ))

    # Configuration summary
    print(f"\n{Fore.YELLOW}⚙️  Configuration:{Style.RESET_ALL}")
    print(f"  Weights:")
    for k, v in config.weights.items():
        print(f"    {k:<18}: {int(v * 100)}%")

    print(f"\n  Filters:")
    print(f"    Min Volume : {config.min_avg_volume:,}")
    print(f"    Min Price  : ${config.min_price}")
    print(f"    Min Score  : {config.min_composite_score:.0f}")
    print(f"    Universe   : {len(data_manager.tickers)} tickers")
    print(f"    Passed     : {len(df_results)} stocks")
    print(f"    Run Date   : {market_now()}")

    # Correlation warning
    print(f"\n{Fore.YELLOW}⚠️  Quick Stats:{Style.RESET_ALL}")
    print(f"  Top Score Range: {top_n['composite_score'].min():.1f} - {top_n['composite_score'].max():.1f}")
    print(f"  Avg Sharpe: {top_n['Sharpe'].mean():.2f}")
    print(f"  Avg Max DD: {top_n['Max_DD'].mean():.1f}%")

    print(f"\n{Fore.RED}⚠️  DISCLAIMER: For educational purposes only. "
          f"Not financial advice. Always do your own due diligence.{Style.RESET_ALL}\n")


def save_results(df_results: pd.DataFrame, output_dir: str = "screener_outputs"):
    """Save results to CSV and JSON"""
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = date_to_iso_basic_minutes(market_now())

    # CSV output
    csv_path = Path(output_dir) / f"{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"{Fore.CYAN}Results saved to {csv_path}{Style.RESET_ALL}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Canadian Stock Screener")
    from config import CAN_TICKERS_URL
    parser.add_argument("--tickers", type=str, default=CAN_TICKERS_URL,
                        help="URL or local file path for the ticker list (one per line)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top stocks to display")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh data cache")
    parser.add_argument("--output", type=str, default="screener_outputs",
                        help="Output directory for results")

    args = parser.parse_args()

    # Update config if needed
    if args.top != CONFIG.top_n:
        CONFIG.top_n = args.top

    # Initialize components
    data_manager = DataManager(args.tickers)
    screener = StockScreener(CONFIG, data_manager)

    try:
        # Run screening
        results = screener.run(force_refresh=args.refresh)

        # Display and save
        if not results.empty:
            display_results(results, CONFIG, data_manager)
            save_results(results, args.output)
        else:
            print(f"{Fore.RED}No results generated. Check your tickers and data connection.{Style.RESET_ALL}")

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Screening interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
