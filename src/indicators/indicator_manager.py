"""
Indicator Manager Module

Manages all technical indicators for multiple symbols, including
calculation, caching, and updates from OHLCV data.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from datetime import datetime
import logging

from .technical_indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_pivot_points_from_ohlcv,
    calculate_all_emas,
    CrossoverType,
    calculate_ema_crossover
)

logger = logging.getLogger(__name__)


@dataclass
class IndicatorValues:
    """Container for all indicator values for a symbol."""
    
    # EMAs
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    
    # RSI
    rsi: Optional[float] = None
    rsi_period: int = 14
    
    # ATR
    atr: Optional[float] = None
    atr_period: int = 14
    
    # Pivot Points
    pivot: Optional[float] = None
    r1: Optional[float] = None
    s1: Optional[float] = None
    r2: Optional[float] = None
    s2: Optional[float] = None
    
    # Trend
    trend: str = 'neutral'  # 'uptrend', 'downtrend', 'neutral'
    
    # Crossover
    last_crossover: CrossoverType = CrossoverType.NONE
    crossover_index: Optional[int] = None
    
    # Candle tracking (for crossover recency + cooldown)
    candle_index: Optional[int] = None
    
    # Metadata
    timestamp: Optional[datetime] = None
    symbol: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert indicator values to dictionary."""
        return {
            'ema_9': self.ema_9,
            'ema_21': self.ema_21,
            'ema_50': self.ema_50,
            'ema_200': self.ema_200,
            'rsi': self.rsi,
            'rsi_period': self.rsi_period,
            'atr': self.atr,
            'atr_period': self.atr_period,
            'pivot': self.pivot,
            'r1': self.r1,
            's1': self.s1,
            'r2': self.r2,
            's2': self.s2,
            'trend': self.trend,
            'last_crossover': self.last_crossover.value,
            'crossover_index': self.crossover_index,
            'candle_index': self.candle_index,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol
        }


class IndicatorManager:
    """
    Manages technical indicators for multiple trading symbols.
    
    Handles calculation, caching, and updates of all indicators
    used in the trading strategy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the IndicatorManager.
        
        Args:
            config: Configuration dictionary with indicator settings
        """
        self.config = config or {}
        
        # Extract configuration values with defaults
        self.rsi_period = self.config.get('rsi_period', 14)
        self.atr_period = self.config.get('atr_period', 14)
        self.pivot_lookback = self.config.get('pivot_lookback', 10)
        self.ema_periods = self.config.get('ema_periods', [9, 21, 50, 200])
        
        # Cache for OHLCV data per symbol
        self._ohlcv_cache: Dict[str, List[List[float]]] = defaultdict(list)
        
        # Cache for indicator values per symbol
        self._indicator_cache: Dict[str, IndicatorValues] = {}
        
        # Cache for full indicator arrays (for crossover detection)
        self._ema_cache: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Maximum cache size per symbol
        self._max_cache_size = self.config.get('max_cache_size', 500)
        
        logger.info(f"IndicatorManager initialized with RSI({self.rsi_period}), "
                   f"ATR({self.atr_period}), Pivot({self.pivot_lookback})")
    
    def update_ohlcv(self, symbol: str, ohlcv: List[List[float]]) -> None:
        """
        Update OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            ohlcv: List of OHLCV candles [timestamp, open, high, low, close, volume]
        """
        if not ohlcv:
            return
        
        # Update cache
        self._ohlcv_cache[symbol] = ohlcv
        
        # Trim cache if too large
        if len(ohlcv) > self._max_cache_size:
            self._ohlcv_cache[symbol] = ohlcv[-self._max_cache_size:]
        
        logger.debug(f"Updated OHLCV for {symbol}: {len(ohlcv)} candles")
    
    def add_candle(self, symbol: str, candle: List[float]) -> None:
        """
        Add a single new candle to the OHLCV cache.
        
        Args:
            symbol: Trading pair symbol
            candle: Single OHLCV candle [timestamp, open, high, low, close, volume]
        """
        self._ohlcv_cache[symbol].append(candle)
        
        # Trim cache if too large
        if len(self._ohlcv_cache[symbol]) > self._max_cache_size:
            self._ohlcv_cache[symbol].pop(0)
    
    def calculate_all(self, symbol: str, ohlcv: Optional[List[List[float]]] = None) -> IndicatorValues:
        """
        Calculate all indicators for a symbol.
        
        Args:
            symbol: Trading pair symbol
            ohlcv: Optional OHLCV data. If None, uses cached data.
            
        Returns:
            IndicatorValues object with all calculated indicators
        """
        # Use provided OHLCV or cached data
        if ohlcv is not None:
            self.update_ohlcv(symbol, ohlcv)
        
        data = self._ohlcv_cache.get(symbol, [])
        
        if not data or len(data) < 10:
            logger.warning(f"Insufficient data for {symbol} to calculate indicators")
            return IndicatorValues(symbol=symbol)
        
        # Extract price arrays
        timestamps = np.array([c[0] for c in data])
        opens = np.array([c[1] for c in data])
        highs = np.array([c[2] for c in data])
        lows = np.array([c[3] for c in data])
        closes = np.array([c[4] for c in data])
        volumes = np.array([c[5] for c in data])
        
        # Create indicator values container
        indicators = IndicatorValues(symbol=symbol)
        indicators.timestamp = datetime.utcnow()
        indicators.candle_index = len(closes) - 1
        
        # Calculate EMAs
        emas = calculate_all_emas(closes)
        self._ema_cache[symbol] = emas
        
        indicators.ema_9 = emas['ema_9'][-1] if len(emas['ema_9']) > 0 and not np.isnan(emas['ema_9'][-1]) else None
        indicators.ema_21 = emas['ema_21'][-1] if len(emas['ema_21']) > 0 and not np.isnan(emas['ema_21'][-1]) else None
        indicators.ema_50 = emas['ema_50'][-1] if len(emas['ema_50']) > 0 and not np.isnan(emas['ema_50'][-1]) else None
        indicators.ema_200 = emas['ema_200'][-1] if len(emas['ema_200']) > 0 and not np.isnan(emas['ema_200'][-1]) else None
        
        # Calculate RSI
        rsi_values = calculate_rsi(closes, self.rsi_period)
        indicators.rsi = rsi_values[-1] if len(rsi_values) > 0 and not np.isnan(rsi_values[-1]) else None
        indicators.rsi_period = self.rsi_period
        
        # Calculate ATR
        atr_values = calculate_atr(highs, lows, closes, self.atr_period)
        indicators.atr = atr_values[-1] if len(atr_values) > 0 and not np.isnan(atr_values[-1]) else None
        indicators.atr_period = self.atr_period
        
        # Calculate Pivot Points
        pivot_data = calculate_pivot_points_from_ohlcv(highs, lows, closes, self.pivot_lookback)
        indicators.pivot = pivot_data['pivot'][-1] if len(pivot_data['pivot']) > 0 and not np.isnan(pivot_data['pivot'][-1]) else None
        indicators.r1 = pivot_data['r1'][-1] if len(pivot_data['r1']) > 0 and not np.isnan(pivot_data['r1'][-1]) else None
        indicators.s1 = pivot_data['s1'][-1] if len(pivot_data['s1']) > 0 and not np.isnan(pivot_data['s1'][-1]) else None
        indicators.r2 = pivot_data['r2'][-1] if len(pivot_data['r2']) > 0 and not np.isnan(pivot_data['r2'][-1]) else None
        indicators.s2 = pivot_data['s2'][-1] if len(pivot_data['s2']) > 0 and not np.isnan(pivot_data['s2'][-1]) else None
        
        # Determine trend
        if indicators.ema_50 is not None and indicators.ema_200 is not None:
            if indicators.ema_50 > indicators.ema_200:
                indicators.trend = 'uptrend'
            elif indicators.ema_50 < indicators.ema_200:
                indicators.trend = 'downtrend'
            else:
                indicators.trend = 'neutral'
        
        # Detect crossovers
        if len(emas['ema_9']) >= 2 and len(emas['ema_21']) >= 2:
            crossover_type, crossover_idx = calculate_ema_crossover(emas['ema_9'], emas['ema_21'])
            indicators.last_crossover = crossover_type
            indicators.crossover_index = crossover_idx
        
        # Cache the results
        self._indicator_cache[symbol] = indicators
        
        logger.debug(f"Calculated indicators for {symbol}: EMA9={indicators.ema_9}, "
                    f"RSI={indicators.rsi}, ATR={indicators.atr}")
        
        return indicators
    
    def get_latest(self, symbol: str) -> Optional[IndicatorValues]:
        """
        Get the latest cached indicator values for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            IndicatorValues object or None if not cached
        """
        return self._indicator_cache.get(symbol)
    
    def get_ema_arrays(self, symbol: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the full EMA arrays for a symbol (for crossover detection).
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary of EMA arrays or None
        """
        return self._ema_cache.get(symbol)
    
    def get_ohlcv(self, symbol: str) -> Optional[List[List[float]]]:
        """
        Get cached OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of OHLCV candles or None
        """
        return self._ohlcv_cache.get(symbol)
    
    def get_price_arrays(self, symbol: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get price arrays for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with price arrays or None
        """
        data = self._ohlcv_cache.get(symbol)
        if not data:
            return None
        
        return {
            'timestamps': np.array([c[0] for c in data]),
            'opens': np.array([c[1] for c in data]),
            'highs': np.array([c[2] for c in data]),
            'lows': np.array([c[3] for c in data]),
            'closes': np.array([c[4] for c in data]),
            'volumes': np.array([c[5] for c in data])
        }
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            self._ohlcv_cache.pop(symbol, None)
            self._indicator_cache.pop(symbol, None)
            self._ema_cache.pop(symbol, None)
            logger.info(f"Cleared cache for {symbol}")
        else:
            self._ohlcv_cache.clear()
            self._indicator_cache.clear()
            self._ema_cache.clear()
            logger.info("Cleared all indicator caches")
    
    def get_all_symbols(self) -> List[str]:
        """
        Get list of all symbols with cached data.
        
        Returns:
            List of symbol strings
        """
        return list(self._ohlcv_cache.keys())
    
    def is_data_sufficient(self, symbol: str, min_candles: int = 200) -> bool:
        """
        Check if sufficient data is available for a symbol.
        
        Args:
            symbol: Trading pair symbol
            min_candles: Minimum number of candles required
            
        Returns:
            True if sufficient data is available
        """
        data = self._ohlcv_cache.get(symbol, [])
        return len(data) >= min_candles
