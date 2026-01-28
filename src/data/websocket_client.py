"""
WebSocket client module for Delta Exchange real-time market data.

This module provides a robust WebSocket client that:
- Connects to Delta Exchange WebSocket API
- Handles automatic reconnection with exponential backoff
- Manages subscriptions for multiple data types and symbols
- Processes incoming messages and updates the data cache
- Implements heartbeat/ping-pong for connection health
- Provides graceful shutdown capabilities
"""

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from .data_cache import DataCache, PriceData, OrderBook, OrderBookLevel, OHLCV, Trade

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


class SubscriptionType(Enum):
    """Types of WebSocket subscriptions."""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    OHLCV = "candlestick"


@dataclass
class Subscription:
    """Represents a single subscription."""
    subscription_type: SubscriptionType
    symbol: str
    timeframe: Optional[str] = None  # For OHLCV subscriptions
    
    @property
    def channel_name(self) -> str:
        """Generate the channel name for this subscription."""
        base = f"{self.subscription_type.value}:{self.symbol}"
        if self.timeframe:
            base += f":{self.timeframe}"
        return base
    
    def __hash__(self):
        return hash(self.channel_name)
    
    def __eq__(self, other):
        if isinstance(other, Subscription):
            return self.channel_name == other.channel_name
        return False


@dataclass
class ReconnectConfig:
    """Configuration for reconnection behavior."""
    max_retries: int = 10
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class WebSocketClient:
    """
    WebSocket client for Delta Exchange real-time market data.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping-pong handling
    - Multi-symbol subscription management
    - Data parsing and cache updates
    - Event callbacks for new data
    - Graceful shutdown
    
    Example:
        cache = DataCache()
        client = WebSocketClient(cache)
        
        await client.connect()
        await client.subscribe_ticker('BTC/USD')
        await client.subscribe_orderbook('BTC/USD')
    """
    
    # Delta Exchange WebSocket URL
    WS_URL = "wss://api.delta.exchange/ws/v2"
    
    # Heartbeat interval (seconds)
    HEARTBEAT_INTERVAL = 30.0
    
    # Connection timeout (seconds)
    CONNECTION_TIMEOUT = 10.0
    
    def __init__(
        self,
        data_cache: Optional[DataCache] = None,
        reconnect_config: Optional[ReconnectConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize the WebSocket client.
        
        Args:
            data_cache: DataCache instance for storing received data
            reconnect_config: Reconnection configuration
            api_key: API key for authenticated channels (optional)
            api_secret: API secret for authenticated channels (optional)
        """
        self.cache = data_cache or DataCache()
        self.reconnect_config = reconnect_config or ReconnectConfig()
        self.api_key = api_key
        self.api_secret = api_secret
        
        # WebSocket connection
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._state = ConnectionState.DISCONNECTED
        
        # Subscription management
        self._subscriptions: Set[Subscription] = set()
        self._pending_subscriptions: Set[Subscription] = set()
        
        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Reconnection state
        self._reconnect_attempts = 0
        self._should_reconnect = True
        self._shutdown_event = asyncio.Event()
        
        # Message buffer during reconnection
        self._message_buffer: List[Dict] = []
        self._buffering = False
        
        # Callbacks
        self._message_callbacks: List[Callable[[Dict], None]] = []
        self._ticker_callbacks: List[Callable[[str, Decimal], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        self._state_callbacks: List[Callable[[ConnectionState], None]] = []
        
        # Locks
        self._lock = asyncio.Lock()
        self._subscription_lock = asyncio.Lock()
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._state == ConnectionState.CONNECTED and self._ws is not None
    
    # Connection management
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection.
        
        Raises:
            ConnectionError: If connection fails after max retries
        """
        if self.is_connected:
            logger.warning("Already connected")
            return
        
        async with self._lock:
            await self._set_state(ConnectionState.CONNECTING)
            
            try:
                logger.info(f"Connecting to {self.WS_URL}")
                
                self._ws = await asyncio.wait_for(
                    websockets.connect(self.WS_URL),
                    timeout=self.CONNECTION_TIMEOUT
                )
                
                await self._set_state(ConnectionState.CONNECTED)
                self._reconnect_attempts = 0
                self._should_reconnect = True
                
                # Start background tasks
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Resubscribe to previous subscriptions
                await self._resubscribe_all()
                
                logger.info("WebSocket connected successfully")
                
            except Exception as e:
                await self._set_state(ConnectionState.DISCONNECTED)
                logger.error(f"Failed to connect: {e}")
                raise ConnectionError(f"Failed to connect to WebSocket: {e}")
    
    async def disconnect(self) -> None:
        """
        Close connection gracefully.
        
        This method will:
        - Stop reconnection attempts
        - Cancel background tasks
        - Close the WebSocket connection
        """
        logger.info("Disconnecting WebSocket client")
        
        self._should_reconnect = False
        self._shutdown_event.set()
        
        async with self._lock:
            await self._set_state(ConnectionState.CLOSING)
            
            # Cancel background tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None
            
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                self._receive_task = None
            
            # Close WebSocket
            if self._ws:
                try:
                    await self._ws.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                self._ws = None
            
            await self._set_state(ConnectionState.DISCONNECTED)
        
        logger.info("WebSocket disconnected")
    
    async def _reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if not self._should_reconnect:
            return
        
        async with self._lock:
            if self._state in (ConnectionState.RECONNECTING, ConnectionState.CLOSING):
                return
            
            await self._set_state(ConnectionState.RECONNECTING)
        
        # Calculate delay with exponential backoff
        delay = min(
            self.reconnect_config.base_delay * (
                self.reconnect_config.exponential_base ** self._reconnect_attempts
            ),
            self.reconnect_config.max_delay
        )
        
        if self.reconnect_config.jitter:
            delay *= (0.5 + 0.5 * (asyncio.get_event_loop().time() % 1))
        
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts + 1})")
        
        await asyncio.sleep(delay)
        
        if not self._should_reconnect:
            return
        
        try:
            # Close existing connection if any
            if self._ws:
                try:
                    await self._ws.close()
                except:
                    pass
                self._ws = None
            
            await self.connect()
            
        except Exception as e:
            self._reconnect_attempts += 1
            
            if self._reconnect_attempts >= self.reconnect_config.max_retries:
                logger.error(f"Max reconnection attempts reached: {e}")
                await self._set_state(ConnectionState.DISCONNECTED)
                self._notify_error(MaxRetriesExceededError("Max reconnection attempts reached"))
            else:
                # Schedule another reconnection attempt
                asyncio.create_task(self._reconnect())
    
    async def _receive_loop(self) -> None:
        """Main receive loop for WebSocket messages."""
        try:
            while self.is_connected and self._ws:
                try:
                    message = await self._ws.recv()
                    await self._handle_message(message)
                    
                except ConnectionClosed as e:
                    logger.warning(f"Connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
            raise
        finally:
            # Trigger reconnection if needed
            if self._should_reconnect and self._state != ConnectionState.CLOSING:
                asyncio.create_task(self._reconnect())
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat pings."""
        try:
            while self.is_connected and self._ws:
                try:
                    # Delta Exchange uses ping messages
                    ping_msg = {"type": "ping"}
                    await self._send_message(ping_msg)
                    
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.HEARTBEAT_INTERVAL
                    )
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
            raise
    
    # Message handling
    
    async def _handle_message(self, message: str) -> None:
        """
        Parse and handle incoming WebSocket messages.
        
        Args:
            message: Raw WebSocket message string
        """
        try:
            data = json.loads(message)
            
            # Notify raw message callbacks
            for callback in self._message_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
            
            # Handle different message types
            msg_type = data.get("type", "").lower()
            
            if msg_type == "pong":
                logger.debug("Received pong")
                
            elif msg_type == "error":
                logger.error(f"Server error: {data}")
                
            elif msg_type == "subscriptions":
                logger.debug(f"Subscription confirmation: {data}")
                
            elif msg_type in ("ticker", "tickers"):
                await self._handle_ticker(data)
                
            elif msg_type in ("orderbook", "l2_orderbook"):
                await self._handle_orderbook(data)
                
            elif msg_type in ("trade", "trades"):
                await self._handle_trade(data)
                
            elif msg_type in ("candlestick", "candle", "ohlcv"):
                await self._handle_ohlcv(data)
                
            else:
                logger.debug(f"Unhandled message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_ticker(self, data: Dict) -> None:
        """Handle ticker/price update messages."""
        try:
            symbol = data.get("symbol") or data.get("product_id")
            if not symbol:
                return
            
            # Normalize symbol format
            symbol = self._normalize_symbol(symbol)
            
            price_data = PriceData(
                symbol=symbol,
                price=Decimal(str(data.get("price", 0) or data.get("mark_price", 0))),
                bid=Decimal(str(data.get("bid", 0))) if data.get("bid") else None,
                ask=Decimal(str(data.get("ask", 0))) if data.get("ask") else None,
                volume_24h=Decimal(str(data.get("volume_24h", 0))) if data.get("volume_24h") else None,
                change_24h=Decimal(str(data.get("change_24h", 0))) if data.get("change_24h") else None,
                timestamp=time.time()
            )
            
            await self.cache.update_price(price_data)
            
            # Notify ticker callbacks
            for callback in self._ticker_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, price_data.price)
                    else:
                        callback(symbol, price_data.price)
                except Exception as e:
                    logger.error(f"Error in ticker callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling ticker: {e}")
    
    async def _handle_orderbook(self, data: Dict) -> None:
        """Handle order book update messages."""
        try:
            symbol = data.get("symbol") or data.get("product_id")
            if not symbol:
                return
            
            symbol = self._normalize_symbol(symbol)
            
            # Parse bids and asks
            bids = []
            asks = []
            
            raw_bids = data.get("bids", data.get("buy", []))
            raw_asks = data.get("asks", data.get("sell", []))
            
            for level in raw_bids:
                if isinstance(level, list) and len(level) >= 2:
                    bids.append(OrderBookLevel(
                        price=Decimal(str(level[0])),
                        size=Decimal(str(level[1]))
                    ))
                elif isinstance(level, dict):
                    bids.append(OrderBookLevel(
                        price=Decimal(str(level.get("price", 0))),
                        size=Decimal(str(level.get("size", 0)))
                    ))
            
            for level in raw_asks:
                if isinstance(level, list) and len(level) >= 2:
                    asks.append(OrderBookLevel(
                        price=Decimal(str(level[0])),
                        size=Decimal(str(level[1]))
                    ))
                elif isinstance(level, dict):
                    asks.append(OrderBookLevel(
                        price=Decimal(str(level.get("price", 0))),
                        size=Decimal(str(level.get("size", 0)))
                    ))
            
            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=time.time(),
                sequence=data.get("sequence", 0)
            )
            
            await self.cache.update_orderbook(orderbook)
            
        except Exception as e:
            logger.error(f"Error handling orderbook: {e}")
    
    async def _handle_trade(self, data: Dict) -> None:
        """Handle trade update messages."""
        try:
            symbol = data.get("symbol") or data.get("product_id")
            if not symbol:
                return
            
            symbol = self._normalize_symbol(symbol)
            
            trades_data = data.get("trades", [data]) if "trades" in data else [data]
            
            for trade_data in trades_data:
                trade = Trade(
                    symbol=symbol,
                    trade_id=str(trade_data.get("id", trade_data.get("trade_id", ""))),
                    price=Decimal(str(trade_data.get("price", 0))),
                    size=Decimal(str(trade_data.get("size", trade_data.get("amount", 0)))),
                    side=trade_data.get("side", "buy").lower(),
                    timestamp=trade_data.get("timestamp", time.time())
                )
                
                await self.cache.add_trade(trade)
                
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
    
    async def _handle_ohlcv(self, data: Dict) -> None:
        """Handle OHLCV/candlestick update messages."""
        try:
            symbol = data.get("symbol") or data.get("product_id")
            timeframe = data.get("timeframe") or data.get("resolution", "15m")
            
            if not symbol:
                return
            
            symbol = self._normalize_symbol(symbol)
            
            # Handle different candle data formats
            candles = data.get("candles", data.get("data", [data]))
            if not isinstance(candles, list):
                candles = [candles]
            
            for candle in candles:
                ohlcv = OHLCV(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=candle.get("timestamp", data.get("timestamp", time.time())),
                    open=Decimal(str(candle.get("open", candle.get("o", 0)))),
                    high=Decimal(str(candle.get("high", candle.get("h", 0)))),
                    low=Decimal(str(candle.get("low", candle.get("l", 0)))),
                    close=Decimal(str(candle.get("close", candle.get("c", 0)))),
                    volume=Decimal(str(candle.get("volume", candle.get("v", 0))))
                )
                
                await self.cache.update_ohlcv(ohlcv)
                
        except Exception as e:
            logger.error(f"Error handling OHLCV: {e}")
    
    # Subscription methods
    
    async def subscribe_ticker(self, symbol: str) -> None:
        """
        Subscribe to ticker/price updates for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
        """
        subscription = Subscription(
            subscription_type=SubscriptionType.TICKER,
            symbol=symbol
        )
        await self._subscribe(subscription)
    
    async def subscribe_orderbook(self, symbol: str) -> None:
        """
        Subscribe to order book updates for a symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        subscription = Subscription(
            subscription_type=SubscriptionType.ORDERBOOK,
            symbol=symbol
        )
        await self._subscribe(subscription)
    
    async def subscribe_trades(self, symbol: str) -> None:
        """
        Subscribe to trade updates for a symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        subscription = Subscription(
            subscription_type=SubscriptionType.TRADES,
            symbol=symbol
        )
        await self._subscribe(subscription)
    
    async def subscribe_ohlcv(self, symbol: str, timeframe: str = "15m") -> None:
        """
        Subscribe to OHLCV/candlestick updates for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (e.g., '15m', '1h', '1d')
        """
        subscription = Subscription(
            subscription_type=SubscriptionType.OHLCV,
            symbol=symbol,
            timeframe=timeframe
        )
        await self._subscribe(subscription)
    
    async def _subscribe(self, subscription: Subscription) -> None:
        """
        Internal method to handle subscription.
        
        Args:
            subscription: Subscription object
        """
        async with self._subscription_lock:
            self._subscriptions.add(subscription)
            
            if self.is_connected:
                await self._send_subscription(subscription)
            else:
                self._pending_subscriptions.add(subscription)
    
    async def _send_subscription(self, subscription: Subscription) -> None:
        """Send subscription message to server."""
        # Delta Exchange format
        channel_map = {
            SubscriptionType.TICKER: "ticker",
            SubscriptionType.ORDERBOOK: "l2_orderbook",
            SubscriptionType.TRADES: "trades",
            SubscriptionType.OHLCV: "candlestick"
        }
        
        channel = channel_map[subscription.subscription_type]
        symbol = self._format_symbol_for_api(subscription.symbol)
        
        msg = {
            "type": "subscribe",
            "payload": {
                "channels": [{
                    "name": channel,
                    "symbols": [symbol]
                }]
            }
        }
        
        # Add timeframe for OHLCV
        if subscription.timeframe:
            msg["payload"]["channels"][0]["timeframe"] = subscription.timeframe
        
        await self._send_message(msg)
        logger.debug(f"Subscribed to {subscription.channel_name}")
    
    async def _resubscribe_all(self) -> None:
        """Resubscribe to all active subscriptions after reconnection."""
        async with self._subscription_lock:
            all_subs = self._subscriptions | self._pending_subscriptions
            self._pending_subscriptions.clear()
            
            for subscription in all_subs:
                try:
                    await self._send_subscription(subscription)
                    await asyncio.sleep(0.05)  # Rate limiting
                except Exception as e:
                    logger.error(f"Failed to resubscribe to {subscription.channel_name}: {e}")
    
    # Unsubscription methods
    
    async def unsubscribe_ticker(self, symbol: str) -> None:
        """Unsubscribe from ticker updates for a symbol."""
        subscription = Subscription(
            subscription_type=SubscriptionType.TICKER,
            symbol=symbol
        )
        await self._unsubscribe(subscription)
    
    async def unsubscribe_orderbook(self, symbol: str) -> None:
        """Unsubscribe from order book updates for a symbol."""
        subscription = Subscription(
            subscription_type=SubscriptionType.ORDERBOOK,
            symbol=symbol
        )
        await self._unsubscribe(subscription)
    
    async def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trade updates for a symbol."""
        subscription = Subscription(
            subscription_type=SubscriptionType.TRADES,
            symbol=symbol
        )
        await self._unsubscribe(subscription)
    
    async def unsubscribe_ohlcv(self, symbol: str, timeframe: str = "15m") -> None:
        """Unsubscribe from OHLCV updates for a symbol."""
        subscription = Subscription(
            subscription_type=SubscriptionType.OHLCV,
            symbol=symbol,
            timeframe=timeframe
        )
        await self._unsubscribe(subscription)
    
    async def _unsubscribe(self, subscription: Subscription) -> None:
        """Internal method to handle unsubscription."""
        async with self._subscription_lock:
            self._subscriptions.discard(subscription)
            self._pending_subscriptions.discard(subscription)
            
            if self.is_connected:
                msg = {
                    "type": "unsubscribe",
                    "payload": {
                        "channels": [subscription.channel_name]
                    }
                }
                await self._send_message(msg)
    
    # Callback registration
    
    def on_message(self, callback: Callable[[Dict], None]) -> None:
        """Register a callback for all raw messages."""
        self._message_callbacks.append(callback)
    
    def on_ticker(self, callback: Callable[[str, Decimal], None]) -> None:
        """Register a callback for ticker/price updates."""
        self._ticker_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback for errors."""
        self._error_callbacks.append(callback)
    
    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register a callback for connection state changes."""
        self._state_callbacks.append(callback)
    
    # Helper methods
    
    async def _send_message(self, message: Dict) -> None:
        """Send a message to the WebSocket server."""
        if not self._ws:
            raise ConnectionError("Not connected")
        
        try:
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def _set_state(self, state: ConnectionState) -> None:
        """Update connection state and notify callbacks."""
        old_state = self._state
        self._state = state
        
        if old_state != state:
            logger.debug(f"State changed: {old_state.value} -> {state.value}")
            
            for callback in self._state_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(state)
                    else:
                        callback(state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")
    
    def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(error))
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format (e.g., 'BTCUSD' -> 'BTC/USD')."""
        # Handle various symbol formats
        if "/" not in symbol:
            # Try to split common patterns
            for quote in ["USD", "USDT", "BTC", "ETH", "INR"]:
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    return f"{base}/{quote}"
        return symbol
    
    def _format_symbol_for_api(self, symbol: str) -> str:
        """Format symbol for Delta Exchange API (remove slash)."""
        return symbol.replace("/", "")
    
    # Public getters
    
    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """Get the last known price for a symbol from cache."""
        return self.cache.get_last_price(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """Get the current order book for a symbol from cache."""
        return self.cache.get_orderbook(symbol)
    
    def get_subscriptions(self) -> List[str]:
        """Get list of active subscription channel names."""
        return [sub.channel_name for sub in self._subscriptions]


class MaxRetriesExceededError(Exception):
    """Exception raised when max reconnection attempts are exceeded."""
    pass