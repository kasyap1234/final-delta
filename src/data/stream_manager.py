"""
Stream manager module for managing multiple WebSocket data streams.

This module provides a high-level manager that:
- Coordinates multiple WebSocket connections if needed
- Manages symbol subscriptions across connections
- Distributes data to registered subscribers
- Handles connection pooling for high-throughput scenarios
- Provides a unified interface for data access
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Set, Any
from dataclasses import dataclass, field
from decimal import Decimal

from .websocket_client import WebSocketClient, ConnectionState, ReconnectConfig
from .data_cache import DataCache, PriceData, OrderBook, OHLCV, Trade

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for the stream manager."""
    # Maximum symbols per WebSocket connection
    max_symbols_per_connection: int = 50
    
    # Enable connection pooling for many symbols
    enable_connection_pooling: bool = True
    
    # Default timeframe for OHLCV subscriptions
    default_timeframe: str = "15m"
    
    # Auto-reconnect settings
    reconnect_config: ReconnectConfig = field(default_factory=ReconnectConfig)
    
    # Buffer data during reconnection
    buffer_during_reconnect: bool = True


@dataclass
class Subscriber:
    """Represents a data subscriber."""
    subscriber_id: str
    symbols: Set[str] = field(default_factory=set)
    data_types: Set[str] = field(default_factory=set)  # 'ticker', 'orderbook', 'trades', 'ohlcv'
    callback: Optional[Callable] = None


class StreamManager:
    """
    High-level manager for WebSocket data streams.
    
    This class provides:
    - Multi-symbol subscription management
    - Data distribution to subscribers
    - Connection pooling for large symbol lists
    - Unified interface for data access
    
    Example:
        cache = DataCache()
        manager = StreamManager(cache)
        
        # Start the manager
        await manager.start()
        
        # Subscribe to symbols
        await manager.subscribe_symbols(['BTC/USD', 'ETH/USD', 'SOL/USD'])
        
        # Register for data updates
        manager.on_ticker(lambda symbol, price: print(f"{symbol}: {price}"))
        
        # Get data
        price = manager.get_last_price('BTC/USD')
        
        # Stop
        await manager.stop()
    """
    
    def __init__(
        self,
        data_cache: Optional[DataCache] = None,
        config: Optional[StreamConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize the stream manager.
        
        Args:
            data_cache: DataCache instance for storing data
            config: Stream configuration
            api_key: API key for authenticated channels
            api_secret: API secret for authenticated channels
        """
        self.cache = data_cache or DataCache()
        self.config = config or StreamConfig()
        self.api_key = api_key
        self.api_secret = api_secret
        
        # WebSocket clients (supports pooling)
        self._clients: List[WebSocketClient] = []
        self._primary_client: Optional[WebSocketClient] = None
        
        # Subscription tracking
        self._subscribed_symbols: Set[str] = set()
        self._symbol_subscriptions: Dict[str, Dict[str, bool]] = {}  # symbol -> {data_type: subscribed}
        
        # Subscriber management
        self._subscribers: Dict[str, Subscriber] = {}
        self._subscriber_counter = 0
        
        # State
        self._running = False
        self._lock = asyncio.Lock()
        
        # Callbacks
        self._ticker_callbacks: List[Callable[[str, Decimal], None]] = []
        self._orderbook_callbacks: List[Callable[[str, OrderBook], None]] = []
        self._trade_callbacks: List[Callable[[str, Trade], None]] = []
        self._ohlcv_callbacks: List[Callable[[str, str, OHLCV], None]] = []
    
    async def start(self) -> None:
        """
        Start the stream manager and establish connections.
        
        This initializes the primary WebSocket connection.
        """
        async with self._lock:
            if self._running:
                logger.warning("Stream manager already running")
                return
            
            logger.info("Starting stream manager")
            
            # Create primary client
            self._primary_client = WebSocketClient(
                data_cache=self.cache,
                reconnect_config=self.config.reconnect_config,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Set up callback forwarding
            self._setup_client_callbacks(self._primary_client)
            
            # Connect
            await self._primary_client.connect()
            self._clients.append(self._primary_client)
            
            self._running = True
            logger.info("Stream manager started")
    
    async def stop(self) -> None:
        """
        Stop the stream manager and close all connections.
        
        This gracefully disconnects all WebSocket clients.
        """
        async with self._lock:
            if not self._running:
                return
            
            logger.info("Stopping stream manager")
            
            # Disconnect all clients
            for client in self._clients:
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting client: {e}")
            
            self._clients.clear()
            self._primary_client = None
            self._running = False
            
            logger.info("Stream manager stopped")
    
    async def restart(self) -> None:
        """Restart the stream manager."""
        await self.stop()
        await asyncio.sleep(1)
        await self.start()
        
        # Resubscribe to previous symbols
        if self._subscribed_symbols:
            symbols = list(self._subscribed_symbols)
            await self.subscribe_symbols(symbols)
    
    # Symbol subscription methods
    
    async def subscribe_symbols(
        self,
        symbols: List[str],
        ticker: bool = True,
        orderbook: bool = True,
        trades: bool = False,
        ohlcv: bool = True,
        timeframe: Optional[str] = None
    ) -> None:
        """
        Subscribe to multiple symbols with specified data types.
        
        Args:
            symbols: List of trading pair symbols
            ticker: Subscribe to ticker/price updates
            orderbook: Subscribe to order book updates
            trades: Subscribe to trade updates
            ohlcv: Subscribe to OHLCV/candlestick updates
            timeframe: Timeframe for OHLCV (default from config)
        """
        if not self._running:
            raise RuntimeError("Stream manager not started. Call start() first.")
        
        timeframe = timeframe or self.config.default_timeframe
        
        # Check if we need connection pooling
        total_symbols = len(self._subscribed_symbols) + len(symbols)
        
        if self.config.enable_connection_pooling and total_symbols > self.config.max_symbols_per_connection:
            await self._handle_pooled_subscriptions(
                symbols, ticker, orderbook, trades, ohlcv, timeframe
            )
        else:
            # Use primary client for all subscriptions
            for symbol in symbols:
                await self._subscribe_symbol(
                    symbol, ticker, orderbook, trades, ohlcv, timeframe,
                    self._primary_client
                )
    
    async def subscribe_symbol(
        self,
        symbol: str,
        ticker: bool = True,
        orderbook: bool = True,
        trades: bool = False,
        ohlcv: bool = True,
        timeframe: Optional[str] = None
    ) -> None:
        """
        Subscribe to a single symbol.
        
        Args:
            symbol: Trading pair symbol
            ticker: Subscribe to ticker updates
            orderbook: Subscribe to order book updates
            trades: Subscribe to trade updates
            ohlcv: Subscribe to OHLCV updates
            timeframe: Timeframe for OHLCV
        """
        await self.subscribe_symbols(
            [symbol], ticker, orderbook, trades, ohlcv, timeframe
        )
    
    async def _subscribe_symbol(
        self,
        symbol: str,
        ticker: bool,
        orderbook: bool,
        trades: bool,
        ohlcv: bool,
        timeframe: str,
        client: WebSocketClient
    ) -> None:
        """Internal method to subscribe a symbol to a specific client."""
        try:
            if symbol not in self._symbol_subscriptions:
                self._symbol_subscriptions[symbol] = {}
            
            if ticker:
                await client.subscribe_ticker(symbol)
                self._symbol_subscriptions[symbol]['ticker'] = True
            
            if orderbook:
                await client.subscribe_orderbook(symbol)
                self._symbol_subscriptions[symbol]['orderbook'] = True
            
            if trades:
                await client.subscribe_trades(symbol)
                self._symbol_subscriptions[symbol]['trades'] = True
            
            if ohlcv:
                await client.subscribe_ohlcv(symbol, timeframe)
                self._symbol_subscriptions[symbol]['ohlcv'] = True
                self._symbol_subscriptions[symbol]['timeframe'] = timeframe
            
            self._subscribed_symbols.add(symbol)
            logger.debug(f"Subscribed to {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            raise
    
    async def _handle_pooled_subscriptions(
        self,
        symbols: List[str],
        ticker: bool,
        orderbook: bool,
        trades: bool,
        ohlcv: bool,
        timeframe: str
    ) -> None:
        """Handle subscriptions across multiple connections."""
        current_client_idx = 0
        symbols_per_client = self.config.max_symbols_per_connection
        
        for i, symbol in enumerate(symbols):
            # Determine which client to use
            client_idx = i // symbols_per_client
            
            # Create new client if needed
            while client_idx >= len(self._clients):
                new_client = WebSocketClient(
                    data_cache=self.cache,
                    reconnect_config=self.config.reconnect_config,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                self._setup_client_callbacks(new_client)
                await new_client.connect()
                self._clients.append(new_client)
                logger.info(f"Created additional WebSocket client #{len(self._clients)}")
            
            client = self._clients[client_idx]
            await self._subscribe_symbol(
                symbol, ticker, orderbook, trades, ohlcv, timeframe, client
            )
    
    async def unsubscribe_symbol(self, symbol: str) -> None:
        """
        Unsubscribe from a symbol.
        
        Args:
            symbol: Trading pair symbol to unsubscribe
        """
        if symbol not in self._subscribed_symbols:
            return
        
        # Find which client has this symbol and unsubscribe
        for client in self._clients:
            try:
                subs = self._symbol_subscriptions.get(symbol, {})
                
                if subs.get('ticker'):
                    await client.unsubscribe_ticker(symbol)
                if subs.get('orderbook'):
                    await client.unsubscribe_orderbook(symbol)
                if subs.get('trades'):
                    await client.unsubscribe_trades(symbol)
                if subs.get('ohlcv'):
                    timeframe = subs.get('timeframe', self.config.default_timeframe)
                    await client.unsubscribe_ohlcv(symbol, timeframe)
                    
            except Exception as e:
                logger.error(f"Error unsubscribing {symbol}: {e}")
        
        self._subscribed_symbols.discard(symbol)
        self._symbol_subscriptions.pop(symbol, None)
        logger.debug(f"Unsubscribed from {symbol}")
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all symbols."""
        symbols = list(self._subscribed_symbols)
        for symbol in symbols:
            await self.unsubscribe_symbol(symbol)
    
    # Subscriber management
    
    def register_subscriber(
        self,
        symbols: List[str],
        data_types: List[str],
        callback: Optional[Callable] = None
    ) -> str:
        """
        Register a subscriber for specific symbols and data types.
        
        Args:
            symbols: List of symbols to subscribe to
            data_types: List of data types ('ticker', 'orderbook', 'trades', 'ohlcv')
            callback: Optional callback function for updates
            
        Returns:
            Subscriber ID for later reference
        """
        self._subscriber_counter += 1
        subscriber_id = f"sub_{self._subscriber_counter}"
        
        subscriber = Subscriber(
            subscriber_id=subscriber_id,
            symbols=set(symbols),
            data_types=set(data_types),
            callback=callback
        )
        
        self._subscribers[subscriber_id] = subscriber
        
        # Auto-subscribe to symbols if manager is running
        if self._running:
            asyncio.create_task(self.subscribe_symbols(symbols))
        
        return subscriber_id
    
    def unregister_subscriber(self, subscriber_id: str) -> None:
        """
        Unregister a subscriber.
        
        Args:
            subscriber_id: Subscriber ID to remove
        """
        self._subscribers.pop(subscriber_id, None)
    
    def update_subscriber_symbols(self, subscriber_id: str, symbols: List[str]) -> None:
        """
        Update the symbols for a subscriber.
        
        Args:
            subscriber_id: Subscriber ID
            symbols: New list of symbols
        """
        if subscriber_id in self._subscribers:
            self._subscribers[subscriber_id].symbols = set(symbols)
    
    # Callback registration
    
    def on_ticker(self, callback: Callable[[str, Decimal], None]) -> None:
        """Register a callback for ticker updates."""
        self._ticker_callbacks.append(callback)
        
        # Also register with cache for direct updates
        self.cache.on_price_update(
            lambda symbol, data: callback(symbol, data.price)
        )
    
    def on_orderbook(self, callback: Callable[[str, OrderBook], None]) -> None:
        """Register a callback for order book updates."""
        self._orderbook_callbacks.append(callback)
        self.cache.on_orderbook_update(callback)
    
    def on_trades(self, callback: Callable[[str, Trade], None]) -> None:
        """Register a callback for trade updates."""
        self._trade_callbacks.append(callback)
        self.cache.on_trade(callback)
    
    def on_ohlcv(self, callback: Callable[[str, str, OHLCV], None]) -> None:
        """Register a callback for OHLCV updates."""
        self._ohlcv_callbacks.append(callback)
        self.cache.on_ohlcv_update(callback)
    
    def _setup_client_callbacks(self, client: WebSocketClient) -> None:
        """Set up callbacks for a WebSocket client."""
        # Forward callbacks
        for callback in self._ticker_callbacks:
            client.on_ticker(callback)
        
        client.on_error(self._handle_client_error)
        client.on_state_change(self._handle_state_change)
    
    def _handle_client_error(self, error: Exception) -> None:
        """Handle errors from WebSocket clients."""
        logger.error(f"WebSocket client error: {error}")
    
    def _handle_state_change(self, state: ConnectionState) -> None:
        """Handle connection state changes."""
        logger.debug(f"Client state changed to: {state.value}")
    
    # Data access methods
    
    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get the last known price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Last price as Decimal, or None
        """
        return self.cache.get_last_price(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """
        Get the current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            OrderBook object or None
        """
        return self.cache.get_orderbook(symbol)
    
    def get_best_bid_ask(self, symbol: str) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Get the best bid and ask prices.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (best_bid, best_ask)
        """
        return self.cache.get_best_bid_ask(symbol)
    
    def get_ohlcv(self, symbol: str, timeframe: Optional[str] = None) -> List[OHLCV]:
        """
        Get OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            
        Returns:
            List of OHLCV objects
        """
        timeframe = timeframe or self.config.default_timeframe
        return self.cache.get_ohlcv(symbol, timeframe)
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades
            
        Returns:
            List of Trade objects
        """
        return self.cache.get_recent_trades(symbol, limit)
    
    # Status and monitoring
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the stream manager.
        
        Returns:
            Dictionary with status information
        """
        client_states = []
        for i, client in enumerate(self._clients):
            client_states.append({
                "index": i,
                "state": client.state.value,
                "is_connected": client.is_connected,
                "subscriptions": len(client.get_subscriptions())
            })
        
        return {
            "running": self._running,
            "num_clients": len(self._clients),
            "client_states": client_states,
            "subscribed_symbols": len(self._subscribed_symbols),
            "symbols": list(self._subscribed_symbols),
            "subscribers": len(self._subscribers),
            "cache_stats": self.cache.get_stats()
        }
    
    def is_healthy(self) -> bool:
        """
        Check if the stream manager is healthy.
        
        Returns:
            True if at least one client is connected
        """
        if not self._running:
            return False
        
        return any(client.is_connected for client in self._clients)
    
    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for at least one connection to be established.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected, False if timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.is_healthy():
                return True
            await asyncio.sleep(0.1)
        
        return False
    
    async def wait_for_data(self, symbol: str, timeout: float = 30.0) -> bool:
        """
        Wait for data to be available for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeout: Maximum time to wait
            
        Returns:
            True if data available, False if timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.cache.get_last_price(symbol) is not None:
                return True
            await asyncio.sleep(0.1)
        
        return False