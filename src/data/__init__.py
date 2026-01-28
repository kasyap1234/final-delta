"""
Data module for real-time market data handling.

This module provides WebSocket-based real-time market data feeds
from Delta Exchange, including:
- Ticker/price updates
- Order book updates
- Trade updates
- OHLCV/candlestick updates

Classes:
    WebSocketClient: Low-level WebSocket client for Delta Exchange
    DataCache: Thread-safe cache for market data
    StreamManager: High-level manager for multiple data streams

Example:
    from data import WebSocketClient, DataCache, StreamManager
    
    # Using StreamManager (recommended)
    cache = DataCache()
    manager = StreamManager(cache)
    
    await manager.start()
    await manager.subscribe_symbols(['BTC/USD', 'ETH/USD'])
    
    # Get data
    price = manager.get_last_price('BTC/USD')
    orderbook = manager.get_orderbook('BTC/USD')
    
    # Register callback
    manager.on_ticker(lambda symbol, price: print(f"{symbol}: {price}"))
    
    await manager.stop()
"""

from .data_cache import (
    DataCache,
    PriceData,
    OrderBook,
    OrderBookLevel,
    OHLCV,
    Trade,
)

from .websocket_client import (
    WebSocketClient,
    ConnectionState,
    SubscriptionType,
    Subscription,
    ReconnectConfig,
    MaxRetriesExceededError,
)

from .stream_manager import (
    StreamManager,
    StreamConfig,
    Subscriber,
)

__all__ = [
    # Data cache
    "DataCache",
    "PriceData",
    "OrderBook",
    "OrderBookLevel",
    "OHLCV",
    "Trade",
    
    # WebSocket client
    "WebSocketClient",
    "ConnectionState",
    "SubscriptionType",
    "Subscription",
    "ReconnectConfig",
    "MaxRetriesExceededError",
    
    # Stream manager
    "StreamManager",
    "StreamConfig",
    "Subscriber",
]