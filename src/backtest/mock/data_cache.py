"""
Data cache module for backtesting.

This module provides a mock data cache that serves pre-loaded historical data.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
import logging

from src.data.data_cache import DataCache, OHLCV, PriceData

logger = logging.getLogger(__name__)


class BacktestDataCache:
    """
    Mock data cache that serves pre-loaded historical data.
    
    This class implements the same interface as DataCache but serves
    data from the historical dataset instead of caching real-time
    WebSocket data.
    """
    
    def __init__(self, historical_data: Dict[str, List[OHLCV]]):
        """
        Initialize the backtest data cache.
        
        Args:
            historical_data: Dictionary mapping symbols to OHLCV data
        """
        self.historical_data = historical_data
        self.current_time: Optional[datetime] = None
        self._cache: Dict[str, Dict[str, List[OHLCV]]] = {}
        
        # Initialize cache structure
        for symbol in historical_data:
            self._cache[symbol] = {}
        
        logger.info(f"BacktestDataCache initialized with {len(historical_data)} symbols")
    
    def set_current_time(self, current_time: datetime) -> None:
        """
        Set the current simulation time.
        
        Args:
            current_time: Current simulation time
        """
        self.current_time = current_time
    
    async def update_ohlcv(self, symbol: str, candle: OHLCV) -> None:
        """
        Update cache with new candle data.
        
        Called by BacktestStreamManager to simulate WebSocket updates.
        
        Args:
            symbol: Trading pair symbol
            candle: OHLCV candle data
        """
        if symbol not in self._cache:
            self._cache[symbol] = {}
        
        timeframe = candle.timeframe
        if timeframe not in self._cache[symbol]:
            self._cache[symbol][timeframe] = []
        
        # Check if we need to update existing candle or add new one
        candles = self._cache[symbol][timeframe]
        if candles:
            last_candle = candles[-1]
            if last_candle.timestamp == candle.timestamp:
                # Update existing candle
                candles[-1] = candle
            else:
                candles.append(candle)
        else:
            candles.append(candle)
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200
    ) -> List[OHLCV]:
        """
        Get OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string (e.g., '15m', '1h')
            limit: Maximum number of candles to return
            
        Returns:
            List of OHLCV objects, oldest first
        """
        # Get historical data for symbol
        historical = self.historical_data.get(symbol, [])
        
        # Filter by current time
        if self.current_time:
            filtered = [
                c for c in historical
                if c.timestamp <= self.current_time
            ]
        else:
            filtered = historical
        
        # Filter by timeframe
        filtered = [c for c in filtered if c.timeframe == timeframe]
        
        # Return last N candles
        return filtered[-limit:] if filtered else []
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Latest price as float, or None if not available
        """
        historical = self.historical_data.get(symbol, [])
        
        # Filter by current time
        if self.current_time:
            filtered = [
                c for c in historical
                if c.timestamp <= self.current_time
            ]
        else:
            filtered = historical
        
        if not filtered:
            return None
        
        # Return close price of latest candle
        return float(filtered[-1].close)
    
    def get_ticker(self, symbol: str) -> Optional[PriceData]:
        """
        Get ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            PriceData object or None
        """
        historical = self.historical_data.get(symbol, [])
        
        # Filter by current time
        if self.current_time:
            filtered = [
                c for c in historical
                if c.timestamp <= self.current_time
            ]
        else:
            filtered = historical
        
        if not filtered:
            return None
        
        latest = filtered[-1]
        
        # Simulate bid/ask spread
        close_price = float(latest.close)
        bid = close_price * 0.9999
        ask = close_price * 1.0001
        
        return PriceData(
            symbol=symbol,
            price=Decimal(str(close_price)),
            bid=Decimal(str(bid)),
            ask=Decimal(str(ask)),
            timestamp=latest.timestamp
        )
    
    def get_latest_candle(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[OHLCV]:
        """
        Get the latest candle for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            
        Returns:
            Latest OHLCV object or None
        """
        historical = self.historical_data.get(symbol, [])
        
        # Filter by current time and timeframe
        if self.current_time:
            filtered = [
                c for c in historical
                if c.timestamp <= self.current_time and c.timeframe == timeframe
            ]
        else:
            filtered = [c for c in historical if c.timeframe == timeframe]
        
        return filtered[-1] if filtered else None
    
    def get_candle_at_time(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime
    ) -> Optional[OHLCV]:
        """
        Get a specific candle at a given timestamp.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            timestamp: Candle timestamp
            
        Returns:
            OHLCV object or None
        """
        historical = self.historical_data.get(symbol, [])
        
        for candle in historical:
            if candle.timeframe == timeframe and candle.timestamp == timestamp:
                return candle
        
        return None
    
    def get_symbols(self) -> List[str]:
        """
        Get all available symbols.
        
        Returns:
            List of symbol strings
        """
        return list(self.historical_data.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_candles = sum(len(data) for data in self.historical_data.values())
        
        return {
            'symbols': len(self.historical_data),
            'total_candles': total_candles,
            'current_time': self.current_time.isoformat() if self.current_time else None
        }
    
    def reset(self) -> None:
        """Reset the cache."""
        self._cache.clear()
        self.current_time = None
        logger.info("BacktestDataCache reset")
