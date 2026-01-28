"""
Stream manager module for backtesting.

This module provides a mock stream manager that feeds historical data
instead of WebSocket streams.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, AsyncGenerator
import logging

from src.data.data_cache import OHLCV, PriceData
from src.backtest.mock.data_cache import BacktestDataCache

logger = logging.getLogger(__name__)


class BacktestStreamManager:
    """
    Mock stream manager that feeds historical data instead of WebSocket.
    
    Instead of subscribing to real-time WebSocket streams, this class
    provides the same interface but serves data from the pre-loaded
    historical dataset.
    """
    
    def __init__(
        self,
        historical_data: Dict[str, List[OHLCV]],
        data_cache: BacktestDataCache
    ):
        """
        Initialize the backtest stream manager.
        
        Args:
            historical_data: Dictionary mapping symbols to OHLCV data
            data_cache: Backtest data cache
        """
        self.historical_data = historical_data
        self.data_cache = data_cache
        self.current_time: Optional[datetime] = None
        self.subscribed_symbols: Set[str] = set()
        self._running = False
        
        logger.info("BacktestStreamManager initialized")
    
    async def start(self) -> None:
        """Start the stream (no-op for backtest)."""
        self._running = True
        logger.info("BacktestStreamManager started")
    
    async def stop(self) -> None:
        """Stop the stream (no-op for backtest)."""
        self._running = False
        logger.info("BacktestStreamManager stopped")
    
    async def subscribe_symbols(self, symbols: List[str]) -> None:
        """
        Subscribe to symbols (track for data serving).
        
        Args:
            symbols: List of symbols to subscribe to
        """
        self.subscribed_symbols.update(symbols)
        logger.debug(f"Subscribed to symbols: {symbols}")
    
    async def unsubscribe_symbols(self, symbols: List[str]) -> None:
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
        logger.debug(f"Unsubscribed from symbols: {symbols}")
    
    async def update_time(self, current_time: datetime) -> None:
        """
        Update the current time and push new candle data to cache.
        
        This is called by the backtest engine at each time step to
        simulate new data arriving from the WebSocket.
        
        Args:
            current_time: Current simulation time
        """
        self.current_time = current_time
        self.data_cache.set_current_time(current_time)
        
        # Push new candle data to cache for subscribed symbols
        for symbol in self.subscribed_symbols:
            data = self.historical_data.get(symbol, [])
            
            # Find the candle for current_time
            candle = next(
                (c for c in data if c.timestamp == current_time),
                None
            )
            
            if candle:
                # Push to cache (simulating WebSocket update)
                await self.data_cache.update_ohlcv(symbol, candle)
    
    async def watch_ohlcv(
        self,
        symbol: str,
        timeframe: str
    ) -> AsyncGenerator[OHLCV, None]:
        """
        Async generator that yields OHLCV data.
        
        In backtest mode, this yields data as time advances.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            
        Yields:
            OHLCV candle data
        """
        while self._running:
            if self.current_time:
                data = self.historical_data.get(symbol, [])
                candle = next(
                    (c for c in data 
                     if c.timestamp == self.current_time and c.timeframe == timeframe),
                    None
                )
                if candle:
                    yield candle
            
            await asyncio.sleep(0)  # Yield control
    
    async def watch_ticker(self, symbol: str) -> AsyncGenerator[PriceData, None]:
        """
        Async generator that yields ticker data.
        
        In backtest mode, this yields ticker data as time advances.
        
        Args:
            symbol: Trading pair symbol
            
        Yields:
            PriceData ticker data
        """
        while self._running:
            if self.current_time:
                ticker = self.data_cache.get_ticker(symbol)
                if ticker:
                    yield ticker
            
            await asyncio.sleep(0)  # Yield control
    
    def get_subscribed_symbols(self) -> Set[str]:
        """
        Get all subscribed symbols.
        
        Returns:
            Set of subscribed symbols
        """
        return self.subscribed_symbols.copy()
    
    def is_running(self) -> bool:
        """
        Check if the stream manager is running.
        
        Returns:
            True if running
        """
        return self._running
    
    def reset(self) -> None:
        """Reset the stream manager."""
        self.current_time = None
        self.subscribed_symbols.clear()
        self._running = False
        logger.info("BacktestStreamManager reset")
