"""
Exchange client module for Delta Exchange India.

This module provides a robust CCXT-based client for interacting with
Delta Exchange India API. It includes:
- Exchange initialization with API credentials
- Market data fetching (OHLCV candles, tickers)
- Order placement and management
- Account balance and position queries
- Rate limit handling with token bucket algorithm
- Error handling with retries and circuit breaker pattern
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass
from functools import wraps

import ccxt.async_support as ccxt
from ccxt.base.errors import (
    NetworkError as CCXTNetworkError,
    ExchangeError as CCXTExchangeError,
    RateLimitExceeded as CCXTRateLimitExceeded,
    AuthenticationError as CCXTAuthenticationError,
    InsufficientFunds as CCXTInsufficientFunds,
    InvalidOrder as CCXTInvalidOrder,
    OrderNotFound as CCXTOrderNotFound,
    ExchangeNotAvailable as CCXTExchangeNotAvailable,
    BadSymbol as CCXTBadSymbol,
)

from .exceptions import (
    ExchangeError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
    InsufficientFundsError,
    InvalidOrderError,
    OrderNotFoundError,
    ExchangeNotAvailableError,
    SymbolNotFoundError,
    CircuitBreakerOpenError,
    MaxRetriesExceededError,
    PositionNotFoundError,
)
from .rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    reset_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for handling repeated failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold reached, requests are blocked
    - HALF_OPEN: Testing if service has recovered
    
    This prevents cascading failures and gives the exchange time to recover.
    """
    
    STATE_CLOSED = 'closed'
    STATE_OPEN = 'open'
    STATE_HALF_OPEN = 'half_open'
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize the circuit breaker.
        
        Args:
            config: Circuit breaker configuration.
        """
        self.config = config or CircuitBreakerConfig()
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function with circuit breaker protection.
        
        Args:
            func: The function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        
        Returns:
            The result of the function call.
        
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open.
            Exception: Any exception raised by the function.
        """
        async with self._lock:
            if self._state == self.STATE_OPEN:
                # Check if we should transition to half-open
                if time.monotonic() - self._last_failure_time >= self.config.reset_timeout:
                    logger.info("Circuit breaker entering half-open state")
                    self._state = self.STATE_HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitBreakerOpenError(
                        message="Circuit breaker is open",
                        failure_count=self._failure_count,
                        last_failure_time=self._last_failure_time,
                        reset_timeout=self.config.reset_timeout
                    )
            
            elif self._state == self.STATE_HALF_OPEN:
                if self._success_count >= self.config.half_open_max_calls:
                    logger.info("Circuit breaker closing after successful half-open tests")
                    self._state = self.STATE_CLOSED
                    self._failure_count = 0
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._success_count += 1
            else:
                self._failure_count = max(0, self._failure_count - 1)
    
    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            
            if self._failure_count >= self.config.failure_threshold:
                if self._state != self.STATE_OPEN:
                    logger.warning(
                        f"Circuit breaker opening after {self._failure_count} failures"
                    )
                    self._state = self.STATE_OPEN
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info("Circuit breaker manually reset")


@dataclass
class ExchangeClientConfig:
    """Configuration for the exchange client."""
    
    # Exchange settings
    exchange_id: str = 'delta'
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True
    testnet: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 30.0
    
    # Rate limiting
    enable_rate_limiter: bool = True
    rate_limit_config: Optional[RateLimitConfig] = None
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    
    # Timeouts (in milliseconds)
    timeout: int = 30000


class ExchangeClient:
    """
    CCXT-based exchange client for Delta Exchange India.
    
    This client provides a unified interface for:
    - Market data: OHLCV, tickers, order books
    - Account: balances, positions
    - Trading: order creation, cancellation, status
    
    Features:
    - Async/await support
    - Automatic rate limiting
    - Retry logic with exponential backoff
    - Circuit breaker pattern for resilience
    - Comprehensive error handling
    
    Example:
        ```python
        from exchange import ExchangeClient
        from config import load_config
        
        config = load_config('config/config.yaml')
        client = ExchangeClient.from_config(config)
        
        await client.connect()
        
        # Fetch market data
        ohlcv = await client.fetch_ohlcv('BTC/USD', '15m', limit=200)
        
        # Place an order
        order = await client.create_order(
            symbol='BTC/USD',
            side='buy',
            amount=0.1,
            price=50000,
            params={'postOnly': True}
        )
        
        await client.close()
        ```
    """
    
    def __init__(self, config: Optional[ExchangeClientConfig] = None):
        """
        Initialize the exchange client.
        
        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self.config = config or ExchangeClientConfig()
        self._exchange: Optional[ccxt.Exchange] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._connected: bool = False
        self._markets_loaded: bool = False
    
    @classmethod
    def from_config(cls, config: Any) -> 'ExchangeClient':
        """
        Create an ExchangeClient from a configuration object.
        
        Args:
            config: Configuration object with exchange settings.
        
        Returns:
            Configured ExchangeClient instance.
        """
        exchange_settings = config.exchange
        
        client_config = ExchangeClientConfig(
            exchange_id=exchange_settings.exchange_id,
            api_key=exchange_settings.api_key,
            api_secret=exchange_settings.api_secret,
            sandbox=exchange_settings.sandbox,
            testnet=exchange_settings.testnet,
        )
        
        return cls(client_config)
    
    async def connect(self) -> None:
        """
        Initialize and connect to the exchange.
        
        This method:
        1. Creates the CCXT exchange instance
        2. Sets up authentication
        3. Enables sandbox mode if configured
        4. Loads markets
        5. Initializes rate limiter and circuit breaker
        
        Raises:
            AuthenticationError: If API credentials are invalid.
            ExchangeNotAvailableError: If exchange is not accessible.
        """
        try:
            logger.info(f"Connecting to {self.config.exchange_id} exchange...")
            
            # Create exchange instance
            exchange_class = getattr(ccxt, self.config.exchange_id)
            
            exchange_config = {
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'timeout': self.config.timeout,
                'enableRateLimit': False,  # We handle rate limiting ourselves
                'options': {
                    'defaultType': 'swap',  # Delta is primarily a derivatives exchange
                }
            }
            
            self._exchange = exchange_class(exchange_config)
            
            # Enable sandbox/testnet mode
            if self.config.sandbox or self.config.testnet:
                logger.info("Enabling sandbox/testnet mode")
                self._exchange.set_sandbox_mode(True)
            
            # Load markets
            await self._exchange.load_markets()
            self._markets_loaded = True
            logger.info(f"Loaded {len(self._exchange.markets)} markets")
            
            # Initialize rate limiter
            if self.config.enable_rate_limiter:
                rate_limit_config = self.config.rate_limit_config or RateLimitConfig()
                self._rate_limiter = RateLimiter(rate_limit_config)
                await self._rate_limiter.start()
                logger.info("Rate limiter initialized")
            
            # Initialize circuit breaker
            if self.config.enable_circuit_breaker:
                cb_config = self.config.circuit_breaker_config or CircuitBreakerConfig()
                self._circuit_breaker = CircuitBreaker(cb_config)
                logger.info("Circuit breaker initialized")
            
            self._connected = True
            logger.info("Successfully connected to exchange")
            
        except CCXTAuthenticationError as e:
            raise AuthenticationError(
                message=f"Authentication failed: {str(e)}",
                details={'original_error': str(e)}
            ) from e
        except CCXTExchangeNotAvailable as e:
            raise ExchangeNotAvailableError(
                message=f"Exchange not available: {str(e)}",
                details={'original_error': str(e)}
            ) from e
        except Exception as e:
            raise ExchangeError(
                message=f"Failed to connect: {str(e)}",
                details={'original_error': str(e)}
            ) from e
    
    async def close(self) -> None:
        """Close the exchange connection and cleanup resources."""
        logger.info("Closing exchange connection...")
        
        if self._rate_limiter:
            await self._rate_limiter.stop()
            self._rate_limiter = None
        
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
        
        self._connected = False
        self._markets_loaded = False
        logger.info("Exchange connection closed")
    
    async def _execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an exchange method with retry logic and error handling.
        
        Args:
            func: The exchange method to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        
        Returns:
            The result of the method execution.
        
        Raises:
            Various ExchangeError subclasses based on the error type.
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Apply rate limiting
                if self._rate_limiter:
                    await self._rate_limiter.acquire()
                
                try:
                    # Apply circuit breaker
                    if self._circuit_breaker:
                        return await self._circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                        
                finally:
                    if self._rate_limiter:
                        self._rate_limiter.release()
                        
            except CCXTRateLimitExceeded as e:
                last_exception = e
                retry_after = getattr(e, 'retry_after', None)
                
                if attempt < self.config.max_retries:
                    delay = retry_after or self._calculate_backoff_delay(attempt)
                    logger.warning(f"Rate limit exceeded, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise RateLimitError(
                        message=f"Rate limit exceeded: {str(e)}",
                        retry_after=retry_after,
                        details={'attempt': attempt, 'max_retries': self.config.max_retries}
                    ) from e
                    
            except CCXTNetworkError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Network error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise NetworkError(
                        message=f"Network error after {self.config.max_retries} retries: {str(e)}",
                        details={'attempt': attempt}
                    ) from e
                    
            except CCXTExchangeNotAvailable as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Exchange not available, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise ExchangeNotAvailableError(
                        message=f"Exchange not available: {str(e)}",
                        details={'attempt': attempt}
                    ) from e
                    
            except CCXTAuthenticationError as e:
                # Don't retry authentication errors
                raise AuthenticationError(
                    message=f"Authentication error: {str(e)}",
                    details={'original_error': str(e)}
                ) from e
                
            except CCXTInsufficientFunds as e:
                # Don't retry insufficient funds errors
                raise InsufficientFundsError(
                    message=f"Insufficient funds: {str(e)}",
                    details={'original_error': str(e)}
                ) from e
                
            except CCXTInvalidOrder as e:
                # Don't retry invalid order errors
                raise InvalidOrderError(
                    message=f"Invalid order: {str(e)}",
                    details={'original_error': str(e)}
                ) from e
                
            except CCXTOrderNotFound as e:
                # Don't retry order not found errors
                raise OrderNotFoundError(
                    message=f"Order not found: {str(e)}",
                    details={'original_error': str(e)}
                ) from e
                
            except CCXTBadSymbol as e:
                raise SymbolNotFoundError(
                    message=f"Symbol not found: {str(e)}",
                    details={'original_error': str(e)}
                ) from e
                
            except CCXTExchangeError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Exchange error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise ExchangeError(
                        message=f"Exchange error: {str(e)}",
                        details={'original_error': str(e)}
                    ) from e
        
        # Should not reach here, but just in case
        raise MaxRetriesExceededError(
            message="Maximum retries exceeded",
            original_error=last_exception,
            attempts=self.config.max_retries + 1
        )
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: The current retry attempt (0-indexed).
        
        Returns:
            Delay in seconds.
        """
        delay = self.config.retry_delay_base * (2 ** attempt)
        return min(delay, self.config.retry_delay_max)
    
    def _ensure_connected(self) -> None:
        """Ensure the client is connected to the exchange."""
        if not self._connected or not self._exchange:
            raise ExchangeError("Client not connected. Call connect() first.")
    
    # ==================== Market Data Methods ====================
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[List[float]]:
        """
        Fetch OHLCV (candlestick) data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD').
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h', '1d').
            since: Timestamp in milliseconds to fetch from.
            limit: Maximum number of candles to fetch.
            params: Additional exchange-specific parameters.
        
        Returns:
            List of OHLCV candles, each as [timestamp, open, high, low, close, volume].
        
        Raises:
            SymbolNotFoundError: If the symbol is not found.
            ExchangeError: If the request fails.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.debug(f"Fetching OHLCV for {symbol} ({timeframe})")
        
        return await self._execute_with_retry(
            self._exchange.fetch_ohlcv,
            symbol,
            timeframe,
            since,
            limit,
            params
        )
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD').
        
        Returns:
            Ticker data including bid, ask, last price, volume, etc.
        
        Raises:
            SymbolNotFoundError: If the symbol is not found.
            ExchangeError: If the request fails.
        """
        self._ensure_connected()
        
        logger.debug(f"Fetching ticker for {symbol}")
        
        return await self._execute_with_retry(
            self._exchange.fetch_ticker,
            symbol
        )
    
    async def fetch_order_book(
        self,
        symbol: str,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch the order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD').
            limit: Maximum number of bids/asks to fetch.
            params: Additional exchange-specific parameters.
        
        Returns:
            Order book data with bids and asks.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.debug(f"Fetching order book for {symbol}")
        
        return await self._execute_with_retry(
            self._exchange.fetch_order_book,
            symbol,
            limit,
            params
        )
    
    async def fetch_markets(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch available markets from the exchange.
        
        Args:
            params: Additional exchange-specific parameters.
        
        Returns:
            List of market information dictionaries.
        """
        self._ensure_connected()
        params = params or {}
        
        return await self._execute_with_retry(
            self._exchange.fetch_markets,
            params
        )
    
    # ==================== Account Methods ====================
    
    async def fetch_balance(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Args:
            params: Additional exchange-specific parameters.
        
        Returns:
            Account balance information including available and total balances.
        
        Raises:
            AuthenticationError: If not authenticated.
            ExchangeError: If the request fails.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.debug("Fetching account balance")
        
        return await self._execute_with_retry(
            self._exchange.fetch_balance,
            params
        )
    
    async def fetch_positions(
        self,
        symbols: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch open positions.
        
        Args:
            symbols: List of symbols to filter positions. If None, fetches all.
            params: Additional exchange-specific parameters.
        
        Returns:
            List of position information dictionaries.
        
        Raises:
            AuthenticationError: If not authenticated.
            ExchangeError: If the request fails.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.debug(f"Fetching positions for symbols: {symbols}")
        
        return await self._execute_with_retry(
            self._exchange.fetch_positions,
            symbols,
            params
        )
    
    async def fetch_position(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch a specific position by symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD').
            params: Additional exchange-specific parameters.
        
        Returns:
            Position information dictionary.
        
        Raises:
            PositionNotFoundError: If the position is not found.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.debug(f"Fetching position for {symbol}")
        
        try:
            return await self._execute_with_retry(
                self._exchange.fetch_position,
                symbol,
                params
            )
        except ExchangeError as e:
            if "not found" in str(e).lower():
                raise PositionNotFoundError(
                    message=f"Position not found for {symbol}",
                    symbol=symbol
                ) from e
            raise
    
    # ==================== Order Methods ====================
    
    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD').
            type: Order type ('market', 'limit', 'stop', etc.).
            side: Order side ('buy' or 'sell').
            amount: Order amount in base currency.
            price: Order price (required for limit orders).
            params: Additional exchange-specific parameters (e.g., {'postOnly': True}).
        
        Returns:
            Order information dictionary.
        
        Raises:
            InvalidOrderError: If order parameters are invalid.
            InsufficientFundsError: If not enough balance.
            ExchangeError: If the request fails.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.info(
            f"Creating {type} {side} order for {amount} {symbol} "
            f"at {price if price else 'market price'}"
        )
        
        return await self._execute_with_retry(
            self._exchange.create_order,
            symbol,
            type,
            side,
            amount,
            price,
            params
        )
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel.
            symbol: Trading pair symbol (recommended for some exchanges).
            params: Additional exchange-specific parameters.
        
        Returns:
            Cancellation result dictionary.
        
        Raises:
            OrderNotFoundError: If the order is not found.
            ExchangeError: If the request fails.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.info(f"Cancelling order {order_id}")
        
        return await self._execute_with_retry(
            self._exchange.cancel_order,
            order_id,
            symbol,
            params
        )
    
    async def fetch_order(
        self,
        order_id: str,
        symbol: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch information about a specific order.
        
        Args:
            order_id: The ID of the order to fetch.
            symbol: Trading pair symbol (recommended for some exchanges).
            params: Additional exchange-specific parameters.
        
        Returns:
            Order information dictionary.
        
        Raises:
            OrderNotFoundError: If the order is not found.
            ExchangeError: If the request fails.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.debug(f"Fetching order {order_id}")
        
        return await self._execute_with_retry(
            self._exchange.fetch_order,
            order_id,
            symbol,
            params
        )
    
    async def fetch_open_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all open orders.
        
        Args:
            symbol: Filter by trading pair symbol.
            since: Timestamp in milliseconds to fetch from.
            limit: Maximum number of orders to fetch.
            params: Additional exchange-specific parameters.
        
        Returns:
            List of open order information dictionaries.
        """
        self._ensure_connected()
        params = params or {}
        
        logger.debug(f"Fetching open orders for {symbol or 'all symbols'}")
        
        return await self._execute_with_retry(
            self._exchange.fetch_open_orders,
            symbol,
            since,
            limit,
            params
        )
    
    async def fetch_closed_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch closed orders.
        
        Args:
            symbol: Filter by trading pair symbol.
            since: Timestamp in milliseconds to fetch from.
            limit: Maximum number of orders to fetch.
            params: Additional exchange-specific parameters.
        
        Returns:
            List of closed order information dictionaries.
        """
        self._ensure_connected()
        params = params or {}
        
        return await self._execute_with_retry(
            self._exchange.fetch_closed_orders,
            symbol,
            since,
            limit,
            params
        )
    
    async def fetch_my_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch personal trading history.
        
        Args:
            symbol: Filter by trading pair symbol.
            since: Timestamp in milliseconds to fetch from.
            limit: Maximum number of trades to fetch.
            params: Additional exchange-specific parameters.
        
        Returns:
            List of trade information dictionaries.
        """
        self._ensure_connected()
        params = params or {}
        
        return await self._execute_with_retry(
            self._exchange.fetch_my_trades,
            symbol,
            since,
            limit,
            params
        )
    
    # ==================== Fee Methods ====================
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trading fee structure.
        
        Args:
            symbol: Trading pair symbol. If None, returns general fee structure.
        
        Returns:
            Fee structure dictionary with maker and taker fees.
        """
        self._ensure_connected()
        
        logger.debug(f"Fetching trading fees for {symbol or 'all markets'}")
        
        # Try to fetch from exchange if method exists
        if hasattr(self._exchange, 'fetch_trading_fee'):
            if symbol:
                return await self._execute_with_retry(
                    self._exchange.fetch_trading_fee,
                    symbol
                )
        
        if hasattr(self._exchange, 'fetch_trading_fees'):
            fees = await self._execute_with_retry(
                self._exchange.fetch_trading_fees
            )
            if symbol and symbol in fees:
                return fees[symbol]
            return fees
        
        # Fallback to markets data
        if symbol and symbol in self._exchange.markets:
            market = self._exchange.markets[symbol]
            return {
                'symbol': symbol,
                'maker': market.get('maker', 0),
                'taker': market.get('taker', 0),
            }
        
        return {
            'maker': self._exchange.fees.get('trading', {}).get('maker', 0),
            'taker': self._exchange.fees.get('trading', {}).get('taker', 0),
        }
    
    # ==================== Utility Methods ====================
    
    async def check_connection(self) -> bool:
        """
        Check if the exchange connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise.
        """
        if not self._connected or not self._exchange:
            return False
        
        try:
            # Try to fetch server time as a lightweight health check
            if hasattr(self._exchange, 'fetch_time'):
                await self._exchange.fetch_time()
            else:
                # Fallback to fetching a ticker
                markets = list(self._exchange.markets.keys())
                if markets:
                    await self._exchange.fetch_ticker(markets[0])
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about the connected exchange.
        
        Returns:
            Dictionary with exchange information.
        """
        if not self._exchange:
            return {}
        
        return {
            'id': self._exchange.id,
            'name': self._exchange.name,
            'version': self._exchange.version,
            'sandbox': self._exchange.sandbox if hasattr(self._exchange, 'sandbox') else False,
            'timeframes': self._exchange.timeframes if hasattr(self._exchange, 'timeframes') else {},
            'has': self._exchange.has if hasattr(self._exchange, 'has') else {},
            'urls': self._exchange.urls if hasattr(self._exchange, 'urls') else {},
        }
    
    def get_rate_limiter_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get rate limiter statistics.
        
        Returns:
            Rate limiter statistics or None if rate limiter is disabled.
        """
        if self._rate_limiter:
            return self._rate_limiter.get_stats()
        return None
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to the exchange."""
        return self._connected
    
    @property
    def markets(self) -> Dict[str, Any]:
        """Get loaded markets."""
        if self._exchange:
            return self._exchange.markets
        return {}
    
    @property
    def symbols(self) -> List[str]:
        """Get list of available symbols."""
        if self._exchange:
            return list(self._exchange.markets.keys())
        return []
