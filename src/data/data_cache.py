"""
Data cache module for storing real-time market data.

This module provides thread-safe caching for WebSocket data including:
- Latest prices for multiple symbols
- Order book snapshots
- OHLCV/candlestick data
- Trade history

Features:
- Thread-safe access using asyncio locks
- Data expiration and cleanup
- Efficient lookups and updates
- Memory management with size limits
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Container for price/ticker data."""
    symbol: str
    price: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None
    timestamp: float = field(default_factory=time.time)
    
    def is_fresh(self, max_age_seconds: float = 60.0) -> bool:
        """Check if the price data is still fresh."""
        return (time.time() - self.timestamp) < max_age_seconds


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: Decimal
    size: Decimal
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "price": str(self.price),
            "size": str(self.size)
        }


@dataclass
class OrderBook:
    """Container for order book data."""
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    
    def is_fresh(self, max_age_seconds: float = 5.0) -> bool:
        """Check if the order book is still fresh."""
        return (time.time() - self.timestamp) < max_age_seconds
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get the best (highest) bid."""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get the best (lowest) ask."""
        return self.asks[0] if self.asks else None
    
    def get_spread(self) -> Optional[Decimal]:
        """Calculate the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Calculate the mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None


@dataclass
class OHLCV:
    """Container for OHLCV candlestick data."""
    symbol: str
    timeframe: str
    timestamp: float
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    def to_list(self) -> List[float]:
        """Convert to list format compatible with CCXT."""
        return [
            int(self.timestamp * 1000),  # milliseconds
            float(self.open),
            float(self.high),
            float(self.low),
            float(self.close),
            float(self.volume)
        ]


@dataclass
class Trade:
    """Container for individual trade data."""
    symbol: str
    trade_id: str
    price: Decimal
    size: Decimal
    side: str  # 'buy' or 'sell'
    timestamp: float
    
    def is_buy(self) -> bool:
        return self.side == 'buy'
    
    def is_sell(self) -> bool:
        return self.side == 'sell'


class DataCache:
    """
    Thread-safe cache for real-time market data.
    
    This class stores and manages:
    - Latest prices for symbols
    - Order book snapshots
    - OHLCV candlestick data
    - Recent trades
    
    All operations are thread-safe using asyncio locks.
    """
    
    def __init__(
        self,
        max_ohlcv_history: int = 1000,
        max_trade_history: int = 1000,
        default_expiry_seconds: float = 300.0
    ):
        """
        Initialize the data cache.
        
        Args:
            max_ohlcv_history: Maximum number of candles to keep per symbol/timeframe
            max_trade_history: Maximum number of trades to keep per symbol
            default_expiry_seconds: Default data expiration time
        """
        self._prices: Dict[str, PriceData] = {}
        self._orderbooks: Dict[str, OrderBook] = {}
        self._ohlcv: Dict[str, Dict[str, deque]] = {}  # symbol -> timeframe -> deque
        self._trades: Dict[str, deque] = {}  # symbol -> deque
        
        self._max_ohlcv_history = max_ohlcv_history
        self._max_trade_history = max_trade_history
        self._default_expiry = default_expiry_seconds
        
        self._lock = asyncio.Lock()
        
        # Callbacks for data updates
        self._price_callbacks: List[Callable[[str, PriceData], None]] = []
        self._orderbook_callbacks: List[Callable[[str, OrderBook], None]] = []
        self._ohlcv_callbacks: List[Callable[[str, str, OHLCV], None]] = []
        self._trade_callbacks: List[Callable[[str, Trade], None]] = []
    
    # Price/Ticker methods
    
    async def update_price(self, price_data: PriceData) -> None:
        """
        Update price data for a symbol.
        
        Args:
            price_data: PriceData object containing the latest price
        """
        async with self._lock:
            self._prices[price_data.symbol] = price_data
        
        # Notify callbacks
        for callback in self._price_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(price_data.symbol, price_data)
                else:
                    callback(price_data.symbol, price_data)
            except Exception as e:
                logger.error(f"Error in price callback: {e}")
    
    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get the last known price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            
        Returns:
            The last price as Decimal, or None if not available
        """
        price_data = self._prices.get(symbol)
        if price_data and price_data.is_fresh(self._default_expiry):
            return price_data.price
        return None
    
    def get_price_data(self, symbol: str) -> Optional[PriceData]:
        """
        Get full price data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            PriceData object or None
        """
        return self._prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, PriceData]:
        """Get all cached price data."""
        return dict(self._prices)
    
    # Order book methods
    
    async def update_orderbook(self, orderbook: OrderBook) -> None:
        """
        Update order book for a symbol.
        
        Args:
            orderbook: OrderBook object containing the snapshot
        """
        async with self._lock:
            self._orderbooks[orderbook.symbol] = orderbook
        
        # Notify callbacks
        for callback in self._orderbook_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(orderbook.symbol, orderbook)
                else:
                    callback(orderbook.symbol, orderbook)
            except Exception as e:
                logger.error(f"Error in orderbook callback: {e}")
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """
        Get the current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            OrderBook object or None if not available
        """
        return self._orderbooks.get(symbol)
    
    def get_best_bid_ask(self, symbol: str) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Get the best bid and ask prices for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (best_bid, best_ask) or (None, None)
        """
        orderbook = self._orderbooks.get(symbol)
        if orderbook:
            best_bid = orderbook.get_best_bid()
            best_ask = orderbook.get_best_ask()
            return (
                best_bid.price if best_bid else None,
                best_ask.price if best_ask else None
            )
        return None, None
    
    # OHLCV methods
    
    async def update_ohlcv(self, ohlcv: OHLCV) -> None:
        """
        Update OHLCV data for a symbol and timeframe.
        
        Args:
            ohlcv: OHLCV object containing candlestick data
        """
        async with self._lock:
            if ohlcv.symbol not in self._ohlcv:
                self._ohlcv[ohlcv.symbol] = {}
            
            if ohlcv.timeframe not in self._ohlcv[ohlcv.symbol]:
                self._ohlcv[ohlcv.symbol][ohlcv.timeframe] = deque(
                    maxlen=self._max_ohlcv_history
                )
            
            # Check if we need to update existing candle or add new one
            candles = self._ohlcv[ohlcv.symbol][ohlcv.timeframe]
            if candles:
                last_candle = candles[-1]
                if last_candle.timestamp == ohlcv.timestamp:
                    # Update existing candle
                    candles[-1] = ohlcv
                else:
                    candles.append(ohlcv)
            else:
                candles.append(ohlcv)
        
        # Notify callbacks
        for callback in self._ohlcv_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ohlcv.symbol, ohlcv.timeframe, ohlcv)
                else:
                    callback(ohlcv.symbol, ohlcv.timeframe, ohlcv)
            except Exception as e:
                logger.error(f"Error in OHLCV callback: {e}")
    
    def get_ohlcv(self, symbol: str, timeframe: str) -> List[OHLCV]:
        """
        Get OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string (e.g., '15m', '1h')
            
        Returns:
            List of OHLCV objects, oldest first
        """
        if symbol in self._ohlcv and timeframe in self._ohlcv[symbol]:
            return list(self._ohlcv[symbol][timeframe])
        return []
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[OHLCV]:
        """
        Get the latest candle for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            
        Returns:
            Latest OHLCV object or None
        """
        if symbol in self._ohlcv and timeframe in self._ohlcv[symbol]:
            candles = self._ohlcv[symbol][timeframe]
            return candles[-1] if candles else None
        return None
    
    # Trade methods
    
    async def add_trade(self, trade: Trade) -> None:
        """
        Add a trade to the cache.
        
        Args:
            trade: Trade object
        """
        async with self._lock:
            if trade.symbol not in self._trades:
                self._trades[trade.symbol] = deque(maxlen=self._max_trade_history)
            
            self._trades[trade.symbol].append(trade)
        
        # Notify callbacks
        for callback in self._trade_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(trade.symbol, trade)
                else:
                    callback(trade.symbol, trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of Trade objects, most recent first
        """
        if symbol in self._trades:
            trades = list(self._trades[symbol])
            return trades[-limit:][::-1]  # Return most recent first
        return []
    
    # Callback registration
    
    def on_price_update(self, callback: Callable[[str, PriceData], None]) -> None:
        """Register a callback for price updates."""
        self._price_callbacks.append(callback)
    
    def on_orderbook_update(self, callback: Callable[[str, OrderBook], None]) -> None:
        """Register a callback for order book updates."""
        self._orderbook_callbacks.append(callback)
    
    def on_ohlcv_update(self, callback: Callable[[str, str, OHLCV], None]) -> None:
        """Register a callback for OHLCV updates."""
        self._ohlcv_callbacks.append(callback)
    
    def on_trade(self, callback: Callable[[str, Trade], None]) -> None:
        """Register a callback for new trades."""
        self._trade_callbacks.append(callback)
    
    # Cleanup methods
    
    async def clear_symbol(self, symbol: str) -> None:
        """Clear all data for a specific symbol."""
        async with self._lock:
            self._prices.pop(symbol, None)
            self._orderbooks.pop(symbol, None)
            self._ohlcv.pop(symbol, None)
            self._trades.pop(symbol, None)
    
    async def clear_expired_data(self, max_age_seconds: Optional[float] = None) -> None:
        """
        Clear expired data from the cache.
        
        Args:
            max_age_seconds: Maximum age for data to be considered valid
        """
        max_age = max_age_seconds or self._default_expiry
        cutoff_time = time.time() - max_age
        
        async with self._lock:
            # Clear expired prices
            expired_prices = [
                symbol for symbol, data in self._prices.items()
                if data.timestamp < cutoff_time
            ]
            for symbol in expired_prices:
                del self._prices[symbol]
            
            # Clear expired order books
            expired_orderbooks = [
                symbol for symbol, data in self._orderbooks.items()
                if data.timestamp < cutoff_time
            ]
            for symbol in expired_orderbooks:
                del self._orderbooks[symbol]
    
    async def clear_all(self) -> None:
        """Clear all cached data."""
        async with self._lock:
            self._prices.clear()
            self._orderbooks.clear()
            self._ohlcv.clear()
            self._trades.clear()
    
    # Stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "prices_count": len(self._prices),
            "orderbooks_count": len(self._orderbooks),
            "ohlcv_symbols": len(self._ohlcv),
            "trades_symbols": len(self._trades),
            "symbols": list(self._prices.keys())
        }