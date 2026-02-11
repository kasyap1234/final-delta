"""
Technical Indicators Module

This module provides functions for calculating various technical indicators
used in the trading strategy, including EMA, RSI, ATR, and Pivot Points.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from enum import Enum


class CrossoverType(Enum):
    """Types of EMA crossovers."""
    NONE = "none"
    BULLISH = "bullish"  # Fast EMA crosses above slow EMA
    BEARISH = "bearish"  # Fast EMA crosses below slow EMA


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    Formula: EMA = (Close * Multiplier) + (Previous EMA * (1 - Multiplier))
    where Multiplier = 2 / (period + 1)
    
    Args:
        prices: Array of price values
        period: EMA period
        
    Returns:
        Array of EMA values (same length as prices, with NaN for first period-1 values)
    """
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    
    multiplier = 2.0 / (period + 1.0)
    ema = np.zeros(len(prices))
    ema[:period] = np.nan
    
    # Initialize first EMA value with SMA
    ema[period - 1] = np.mean(prices[:period])
    
    # Calculate EMA for remaining values
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
    
    return ema


def calculate_ema_crossover(ema_fast: np.ndarray, ema_slow: np.ndarray) -> Tuple[CrossoverType, Optional[int]]:
    """
    Detect EMA crossover between fast and slow EMAs.
    
    Args:
        ema_fast: Array of fast EMA values
        ema_slow: Array of slow EMA values
        
    Returns:
        Tuple of (crossover_type, index_of_crossover)
        crossover_type: NONE, BULLISH, or BEARISH
        index: Index where crossover occurred, or None if no recent crossover
    """
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return CrossoverType.NONE, None
    
    # Get the last two values for comparison
    fast_prev, fast_curr = ema_fast[-2], ema_fast[-1]
    slow_prev, slow_curr = ema_slow[-2], ema_slow[-1]
    
    # Check for valid values (not NaN)
    if np.isnan(fast_prev) or np.isnan(fast_curr) or np.isnan(slow_prev) or np.isnan(slow_curr):
        return CrossoverType.NONE, None
    
    # Bullish crossover: fast was below slow, now above
    if fast_prev <= slow_prev and fast_curr > slow_curr:
        return CrossoverType.BULLISH, len(ema_fast) - 1
    
    # Bearish crossover: fast was above slow, now below
    if fast_prev >= slow_prev and fast_curr < slow_curr:
        return CrossoverType.BEARISH, len(ema_fast) - 1
    
    return CrossoverType.NONE, None


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Args:
        prices: Array of closing prices
        period: RSI period (default 14)
        
    Returns:
        Array of RSI values (0-100 range, with NaN for first period values)
    """
    if len(prices) <= period:
        return np.full(len(prices), np.nan)
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi = np.full(len(prices), np.nan)
    
    # Calculate first RSI value
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate remaining RSI values using smoothed averages
    for i in range(period + 1, len(prices)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        
        # Smoothed moving averages
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Formula:
    TR = max(high - low, |high - previous_close|, |low - previous_close|)
    ATR = SMA(TR, period)
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: ATR period (default 14)
        
    Returns:
        Array of ATR values
    """
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return np.full(len(highs), np.nan)
    
    # Calculate True Range for each period
    tr = np.zeros(len(highs))
    tr[0] = highs[0] - lows[0]  # First TR is just high - low
    
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]  # Current high - current low
        tr2 = abs(highs[i] - closes[i - 1])  # Current high - previous close
        tr3 = abs(lows[i] - closes[i - 1])  # Current low - previous close
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR using Wilder's smoothing method
    atr = np.full(len(highs), np.nan)
    
    # First ATR is simple average of first 'period' TR values
    if len(tr) >= period:
        atr[period - 1] = np.mean(tr[:period])
        
        # Remaining ATR values use smoothing
        for i in range(period, len(tr)):
            atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period
    
    return atr


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calculate classic pivot point and support/resistance levels.
    
    Formula:
    PP = (High + Low + Close) / 3
    R1 = (2 * PP) - Low
    S1 = (2 * PP) - High
    R2 = PP + (High - Low)
    S2 = PP - (High - Low)
    
    Args:
        high: High price for the period
        low: Low price for the period
        close: Close price for the period
        
    Returns:
        Dictionary with pivot point and support/resistance levels
    """
    pp = (high + low + close) / 3.0
    r1 = (2 * pp) - low
    s1 = (2 * pp) - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    
    return {
        'pivot': pp,
        'r1': r1,
        's1': s1,
        'r2': r2,
        's2': s2
    }


def calculate_pivot_points_from_ohlcv(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 10
) -> Dict[str, np.ndarray]:
    """
    Calculate pivot points for multiple periods from OHLCV data.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        lookback: Lookback period for calculating pivot range (default 10)
        
    Returns:
        Dictionary with arrays of pivot points and support/resistance levels
    """
    n = len(highs)
    
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        # Calculate pivot based on lookback period
        high = np.max(highs[i - lookback + 1:i + 1])
        low = np.min(lows[i - lookback + 1:i + 1])
        close = closes[i]
        
        levels = calculate_pivot_points(high, low, close)
        
        pivot[i] = levels['pivot']
        r1[i] = levels['r1']
        s1[i] = levels['s1']
        r2[i] = levels['r2']
        s2[i] = levels['s2']
    
    return {
        'pivot': pivot,
        'r1': r1,
        's1': s1,
        'r2': r2,
        's2': s2
    }


def is_near_resistance(
    price: float,
    resistance_levels: np.ndarray,
    threshold: float = 0.005
) -> Tuple[bool, Optional[float]]:
    """
    Check if price is near a resistance level.
    
    Args:
        price: Current price
        resistance_levels: Array of resistance level values
        threshold: Percentage threshold for "near" (default 0.5%)
        
    Returns:
        Tuple of (is_near, nearest_level)
        is_near: True if price is within threshold of any resistance level
        nearest_level: The nearest resistance level, or None
    """
    if len(resistance_levels) == 0 or np.all(np.isnan(resistance_levels)):
        return False, None
    
    # Filter out NaN values
    valid_levels = resistance_levels[~np.isnan(resistance_levels)]
    
    if len(valid_levels) == 0:
        return False, None
    
    # Calculate percentage distance to each level
    distances = np.abs((valid_levels - price) / price)
    
    # Find minimum distance
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    
    if min_distance <= threshold:
        return True, valid_levels[min_idx]
    
    return False, None


def is_near_support(
    price: float,
    support_levels: np.ndarray,
    threshold: float = 0.005
) -> Tuple[bool, Optional[float]]:
    """
    Check if price is near a support level.
    
    Args:
        price: Current price
        support_levels: Array of support level values
        threshold: Percentage threshold for "near" (default 0.5%)
        
    Returns:
        Tuple of (is_near, nearest_level)
        is_near: True if price is within threshold of any support level
        nearest_level: The nearest support level, or None
    """
    # Same logic as is_near_resistance
    return is_near_resistance(price, support_levels, threshold)


def detect_rsi_divergence(
    prices: np.ndarray,
    rsi_values: np.ndarray,
    lookback: int = 14
) -> Optional[str]:
    """
    Detect bullish or bearish RSI divergence.
    
    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high
    
    Args:
        prices: Array of price values
        rsi_values: Array of RSI values
        lookback: Lookback period for detecting divergence
        
    Returns:
        'bullish', 'bearish', or None if no divergence detected
    """
    if len(prices) < lookback * 2 or len(rsi_values) < lookback * 2:
        return None
    
    # Get recent and previous windows
    recent_prices = prices[-lookback:]
    previous_prices = prices[-lookback * 2:-lookback]
    recent_rsi = rsi_values[-lookback:]
    previous_rsi = rsi_values[-lookback * 2:-lookback]
    
    # Find local extrema
    recent_price_low = np.min(recent_prices)
    previous_price_low = np.min(previous_prices)
    recent_price_high = np.max(recent_prices)
    previous_price_high = np.max(previous_prices)
    
    recent_rsi_low = np.min(recent_rsi)
    previous_rsi_low = np.min(previous_rsi)
    recent_rsi_high = np.max(recent_rsi)
    previous_rsi_high = np.max(previous_rsi)
    
    # Check for bullish divergence
    if recent_price_low < previous_price_low and recent_rsi_low > previous_rsi_low:
        return 'bullish'
    
    # Check for bearish divergence
    if recent_price_high > previous_price_high and recent_rsi_high < previous_rsi_high:
        return 'bearish'
    
    return None


def get_trend_direction(ema_fast: float, ema_slow: float) -> str:
    """
    Determine trend direction based on EMA positions.
    
    Args:
        ema_fast: Fast EMA value
        ema_slow: Slow EMA value
        
    Returns:
        'uptrend', 'downtrend', or 'neutral'
    """
    if np.isnan(ema_fast) or np.isnan(ema_slow):
        return 'neutral'
    
    if ema_fast > ema_slow:
        return 'uptrend'
    elif ema_fast < ema_slow:
        return 'downtrend'
    else:
        return 'neutral'


def calculate_all_emas(prices: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate all EMA periods used in the strategy.
    
    Args:
        prices: Array of price values
        
    Returns:
        Dictionary with EMA 9, 21, 50, and 200
    """
    return {
        'ema_9': calculate_ema(prices, 9),
        'ema_21': calculate_ema(prices, 21),
        'ema_50': calculate_ema(prices, 50),
        'ema_200': calculate_ema(prices, 200)
    }


def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength regardless of direction.
    Values above 25 indicate a strong trend, below 20 indicate weak trend.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: ADX period (default 14)
        
    Returns:
        Array of ADX values (0-100 range)
    """
    if len(highs) < period * 2 + 1 or len(lows) < period * 2 + 1 or len(closes) < period * 2 + 1:
        return np.full(len(highs), np.nan)
    
    # Calculate True Range (TR)
    tr = np.zeros(len(highs))
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(len(highs))
    minus_dm = np.zeros(len(highs))
    
    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Calculate smoothed TR, +DM, -DM using Wilder's smoothing
    atr = np.full(len(highs), np.nan)
    plus_di = np.full(len(highs), np.nan)
    minus_di = np.full(len(highs), np.nan)
    dx = np.full(len(highs), np.nan)
    adx = np.full(len(highs), np.nan)
    
    # First values are simple averages
    if len(tr) >= period:
        atr[period - 1] = np.mean(tr[:period])
        plus_di[period - 1] = np.mean(plus_dm[:period])
        minus_di[period - 1] = np.mean(minus_dm[:period])
        
        # Calculate DX
        if plus_di[period - 1] + minus_di[period - 1] > 0:
            dx[period - 1] = 100 * abs(plus_di[period - 1] - minus_di[period - 1]) / (plus_di[period - 1] + minus_di[period - 1])
        else:
            dx[period - 1] = 0
        
        # Calculate remaining values using smoothing
        for i in range(period, len(highs)):
            atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period
            plus_di[i] = ((plus_di[i - 1] * (period - 1)) + plus_dm[i]) / period
            minus_di[i] = ((minus_di[i - 1] * (period - 1)) + minus_dm[i]) / period
            
            # Calculate DX
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        # Calculate ADX (smoothed DX) - need at least period*2 data points
        if len(dx) >= period * 2:
            adx[period * 2 - 1] = np.mean(dx[period:period * 2])
            
            for i in range(period * 2, len(highs)):
                adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period
    
    return adx
