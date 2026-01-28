"""
Exchange package for Delta Exchange India Trading Bot.

This package provides a robust CCXT-based exchange client with:
- Exchange initialization with API credentials
- Market data fetching (OHLCV candles)
- Order placement and management
- Account balance and position queries
- Rate limit handling
- Error handling and retries
"""

from .exceptions import (
    ExchangeError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
    InsufficientFundsError,
    InvalidOrderError,
    OrderNotFoundError,
    ExchangeNotAvailableError,
    CircuitBreakerOpenError,
)

from .rate_limiter import RateLimiter

from .exchange_client import ExchangeClient

__all__ = [
    # Exceptions
    'ExchangeError',
    'NetworkError',
    'RateLimitError',
    'AuthenticationError',
    'InsufficientFundsError',
    'InvalidOrderError',
    'OrderNotFoundError',
    'ExchangeNotAvailableError',
    'CircuitBreakerOpenError',
    # Rate Limiter
    'RateLimiter',
    # Exchange Client
    'ExchangeClient',
]
