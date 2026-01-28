"""
Custom exceptions for the exchange client module.

This module defines a hierarchy of exceptions for handling various error scenarios
when interacting with the Delta Exchange API through CCXT.
"""


class ExchangeError(Exception):
    """Base exception for all exchange-related errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class NetworkError(ExchangeError):
    """
    Exception raised for network-related errors.
    
    These errors are typically transient and can be retried.
    Examples: connection timeouts, DNS resolution failures, connection refused.
    """
    
    def __init__(self, message: str = "Network error occurred", details: dict = None):
        super().__init__(message, error_code="NETWORK_ERROR", details=details)


class RateLimitError(ExchangeError):
    """
    Exception raised when API rate limits are exceeded.
    
    This error indicates that too many requests have been made in a short period.
    The client should back off and retry after the rate limit reset time.
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = None,
        details: dict = None
    ):
        super().__init__(message, error_code="RATE_LIMIT_ERROR", details=details)
        self.retry_after = retry_after  # Seconds to wait before retrying


class AuthenticationError(ExchangeError):
    """
    Exception raised for authentication failures.
    
    Examples: invalid API key, expired credentials, insufficient permissions.
    These errors should not be retried without fixing the credentials.
    """
    
    def __init__(self, message: str = "Authentication failed", details: dict = None):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", details=details)


class InsufficientFundsError(ExchangeError):
    """
    Exception raised when there are insufficient funds for an operation.
    
    This occurs when trying to place an order or transfer funds without
    adequate balance in the relevant account.
    """
    
    def __init__(
        self,
        message: str = "Insufficient funds",
        currency: str = None,
        required: float = None,
        available: float = None,
        details: dict = None
    ):
        super().__init__(message, error_code="INSUFFICIENT_FUNDS", details=details)
        self.currency = currency
        self.required = required
        self.available = available


class InvalidOrderError(ExchangeError):
    """
    Exception raised for invalid order parameters.
    
    Examples: invalid symbol, invalid price/amount, order would trigger immediately
    for post-only orders, etc.
    """
    
    def __init__(
        self,
        message: str = "Invalid order parameters",
        symbol: str = None,
        details: dict = None
    ):
        super().__init__(message, error_code="INVALID_ORDER", details=details)
        self.symbol = symbol


class OrderNotFoundError(ExchangeError):
    """
    Exception raised when a requested order cannot be found.
    
    This can occur when trying to fetch, cancel, or modify an order that
    doesn't exist or has already been filled/cancelled.
    """
    
    def __init__(
        self,
        message: str = "Order not found",
        order_id: str = None,
        symbol: str = None,
        details: dict = None
    ):
        super().__init__(message, error_code="ORDER_NOT_FOUND", details=details)
        self.order_id = order_id
        self.symbol = symbol


class ExchangeNotAvailableError(ExchangeError):
    """
    Exception raised when the exchange is unavailable.
    
    Examples: exchange maintenance, DDoS protection, temporary outage.
    These errors can typically be retried after a delay.
    """
    
    def __init__(
        self,
        message: str = "Exchange is not available",
        retry_after: int = None,
        details: dict = None
    ):
        super().__init__(message, error_code="EXCHANGE_NOT_AVAILABLE", details=details)
        self.retry_after = retry_after


class SymbolNotFoundError(ExchangeError):
    """
    Exception raised when a trading symbol is not found.
    
    This occurs when trying to trade a symbol that doesn't exist on the exchange
    or has been delisted.
    """
    
    def __init__(self, message: str = "Symbol not found", symbol: str = None, details: dict = None):
        super().__init__(message, error_code="SYMBOL_NOT_FOUND", details=details)
        self.symbol = symbol


class CircuitBreakerOpenError(ExchangeError):
    """
    Exception raised when the circuit breaker is open.
    
    The circuit breaker pattern prevents cascading failures by temporarily
    blocking requests after a series of consecutive failures.
    """
    
    def __init__(
        self,
        message: str = "Circuit breaker is open",
        failure_count: int = None,
        last_failure_time: float = None,
        reset_timeout: int = None,
        details: dict = None
    ):
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", details=details)
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time
        self.reset_timeout = reset_timeout


class MaxRetriesExceededError(ExchangeError):
    """
    Exception raised when the maximum number of retry attempts is exceeded.
    
    This is typically raised after all retry attempts for a transient error
    have been exhausted.
    """
    
    def __init__(
        self,
        message: str = "Maximum retry attempts exceeded",
        original_error: Exception = None,
        attempts: int = None,
        details: dict = None
    ):
        super().__init__(message, error_code="MAX_RETRIES_EXCEEDED", details=details)
        self.original_error = original_error
        self.attempts = attempts


class PositionNotFoundError(ExchangeError):
    """
    Exception raised when a requested position cannot be found.
    
    This occurs when trying to access or modify a position that doesn't exist.
    """
    
    def __init__(
        self,
        message: str = "Position not found",
        symbol: str = None,
        position_id: str = None,
        details: dict = None
    ):
        super().__init__(message, error_code="POSITION_NOT_FOUND", details=details)
        self.symbol = symbol
        self.position_id = position_id
