"""Price history management for correlation calculations.

This module provides the PriceHistory class for storing and managing
historical price data for correlation calculations.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """A single price data point."""
    timestamp: datetime
    price: float


class PriceHistory:
    """Manages historical price data for correlation calculations.
    
    This class stores price data for multiple symbols and provides
    methods to retrieve price series for correlation calculations.
    It maintains a configurable lookback period and handles data gaps.
    
    Attributes:
        lookback_days: Number of days of historical data to maintain
        min_data_points: Minimum number of data points required for correlation
    """
    
    def __init__(
        self,
        lookback_days: int = 30,
        min_data_points: int = 20
    ):
        """Initialize the price history manager.
        
        Args:
            lookback_days: Number of days of historical data to maintain
            min_data_points: Minimum data points required for correlation
        """
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points
        self._price_data: Dict[str, deque] = {}
        self._max_size = lookback_days * 24 * 60  # Assume minute data as worst case
        
        logger.info(
            f"PriceHistory initialized with lookback_days={lookback_days}, "
            f"min_data_points={min_data_points}"
        )
    
    def add_price(
        self,
        symbol: str,
        timestamp: datetime,
        price: float
    ) -> None:
        """Add a new price data point for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTC/USD')
            timestamp: The timestamp of the price data
            price: The price value
        """
        if symbol not in self._price_data:
            self._price_data[symbol] = deque(maxlen=self._max_size)
            logger.debug(f"Created new price history for {symbol}")
        
        # Validate price
        if not isinstance(price, (int, float)) or np.isnan(price) or np.isinf(price):
            logger.warning(f"Invalid price {price} for {symbol}, skipping")
            return
        
        # Add price point
        price_point = PricePoint(timestamp=timestamp, price=float(price))
        self._price_data[symbol].append(price_point)
        
        # Clean old data
        self._clean_old_data(symbol)
    
    def add_prices(
        self,
        symbol: str,
        prices: List[Tuple[datetime, float]]
    ) -> None:
        """Add multiple price data points for a symbol.
        
        Args:
            symbol: The trading pair symbol
            prices: List of (timestamp, price) tuples
        """
        for timestamp, price in prices:
            self.add_price(symbol, timestamp, price)
    
    def get_price_series(
        self,
        symbol: str,
        lookback_days: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get price series for a symbol.
        
        Args:
            symbol: The trading pair symbol
            lookback_days: Override default lookback period
            
        Returns:
            Tuple of (timestamps array, prices array)
            
        Raises:
            KeyError: If symbol has no data
            ValueError: If insufficient data points
        """
        if symbol not in self._price_data:
            raise KeyError(f"No price data for symbol: {symbol}")
        
        lookback = lookback_days or self.lookback_days
        cutoff_time = datetime.utcnow() - timedelta(days=lookback)
        
        # Filter data within lookback period
        data = self._price_data[symbol]
        filtered_data = [
            (pp.timestamp, pp.price)
            for pp in data
            if pp.timestamp >= cutoff_time
        ]
        
        if len(filtered_data) < self.min_data_points:
            raise ValueError(
                f"Insufficient data for {symbol}: "
                f"{len(filtered_data)} points, need {self.min_data_points}"
            )
        
        timestamps = np.array([ts for ts, _ in filtered_data])
        prices = np.array([price for _, price in filtered_data])
        
        return timestamps, prices
    
    def get_aligned_price_series(
        self,
        symbol1: str,
        symbol2: str,
        lookback_days: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get aligned price series for two symbols.
        
        Returns price series that have matching timestamps,
        which is required for correlation calculation.
        
        Args:
            symbol1: First trading pair symbol
            symbol2: Second trading pair symbol
            lookback_days: Override default lookback period
            
        Returns:
            Tuple of (timestamps array, prices1 array, prices2 array)
            
        Raises:
            KeyError: If either symbol has no data
            ValueError: If insufficient aligned data points
        """
        ts1, prices1 = self.get_price_series(symbol1, lookback_days)
        ts2, prices2 = self.get_price_series(symbol2, lookback_days)
        
        # Find common timestamps (within 1 minute tolerance)
        aligned_ts = []
        aligned_p1 = []
        aligned_p2 = []
        
        # Create dictionaries for faster lookup
        ts2_dict = {ts: price for ts, price in zip(ts2, prices2)}
        
        for ts, p1 in zip(ts1, prices1):
            # Look for matching timestamp within 1 minute
            for offset in range(-60, 61):  # +/- 60 seconds
                check_ts = ts + timedelta(seconds=offset)
                if check_ts in ts2_dict:
                    aligned_ts.append(ts)
                    aligned_p1.append(p1)
                    aligned_p2.append(ts2_dict[check_ts])
                    break
        
        if len(aligned_ts) < self.min_data_points:
            raise ValueError(
                f"Insufficient aligned data points: "
                f"{len(aligned_ts)}, need {self.min_data_points}"
            )
        
        return (
            np.array(aligned_ts),
            np.array(aligned_p1),
            np.array(aligned_p2)
        )
    
    def has_sufficient_data(
        self,
        symbol: str,
        min_points: Optional[int] = None
    ) -> bool:
        """Check if symbol has sufficient data for correlation.
        
        Args:
            symbol: The trading pair symbol
            min_points: Minimum required data points (default: self.min_data_points)
            
        Returns:
            True if sufficient data exists
        """
        if symbol not in self._price_data:
            return False
        
        min_points = min_points or self.min_data_points
        cutoff_time = datetime.utcnow() - timedelta(days=self.lookback_days)
        
        data = self._price_data[symbol]
        recent_count = sum(
            1 for pp in data
            if pp.timestamp >= cutoff_time
        )
        
        return recent_count >= min_points
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with price data.
        
        Returns:
            List of symbol strings
        """
        return list(self._price_data.keys())
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Latest price or None if no data
        """
        if symbol not in self._price_data or not self._price_data[symbol]:
            return None
        
        return self._price_data[symbol][-1].price
    
    def get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the most recent timestamp for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Latest timestamp or None if no data
        """
        if symbol not in self._price_data or not self._price_data[symbol]:
            return None
        
        return self._price_data[symbol][-1].timestamp
    
    def clear_symbol(self, symbol: str) -> None:
        """Clear all data for a specific symbol.
        
        Args:
            symbol: The trading pair symbol to clear
        """
        if symbol in self._price_data:
            del self._price_data[symbol]
            logger.info(f"Cleared price history for {symbol}")
    
    def clear_all(self) -> None:
        """Clear all price data."""
        self._price_data.clear()
        logger.info("Cleared all price history")
    
    def _clean_old_data(self, symbol: str) -> None:
        """Remove data older than lookback period.
        
        Args:
            symbol: The symbol to clean
        """
        if symbol not in self._price_data:
            return
        
        cutoff_time = datetime.utcnow() - timedelta(days=self.lookback_days + 1)
        data = self._price_data[symbol]
        
        # Remove old data from the left side of deque
        while data and data[0].timestamp < cutoff_time:
            data.popleft()
    
    def get_data_stats(self) -> Dict[str, Dict]:
        """Get statistics about stored price data.
        
        Returns:
            Dictionary with stats per symbol
        """
        stats = {}
        cutoff_time = datetime.utcnow() - timedelta(days=self.lookback_days)
        
        for symbol, data in self._price_data.items():
            recent_data = [pp for pp in data if pp.timestamp >= cutoff_time]
            if recent_data:
                prices = [pp.price for pp in recent_data]
                stats[symbol] = {
                    'total_points': len(data),
                    'recent_points': len(recent_data),
                    'price_min': min(prices),
                    'price_max': max(prices),
                    'price_mean': np.mean(prices),
                    'last_update': recent_data[-1].timestamp
                }
        
        return stats
