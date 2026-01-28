"""
Rate limiter implementation for the exchange client.

This module provides rate limiting functionality using the token bucket algorithm
to ensure API requests stay within exchange limits while optimizing throughput.
"""

import asyncio
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Token bucket settings
    max_requests_per_second: float = 10.0
    max_burst: int = 20
    
    # Queue settings
    max_queue_size: int = 100
    queue_timeout: float = 30.0
    
    # Backoff settings
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class RateLimiter:
    """
    Token bucket rate limiter with request queuing.
    
    This class implements a token bucket algorithm to control the rate of API
    requests. It supports:
    - Token bucket for smooth rate limiting
    - Request queueing when rate limited
    - Exponential backoff for retries
    - Per-endpoint rate limiting
    
    Example:
        ```python
        limiter = RateLimiter(max_requests_per_second=10.0)
        
        # Acquire permission to make a request
        await limiter.acquire()
        try:
            # Make API request
            response = await exchange.fetch_ticker('BTC/USD')
        finally:
            limiter.release()
        ```
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.
        
        Args:
            config: Rate limiting configuration. Uses defaults if not provided.
        """
        self.config = config or RateLimitConfig()
        
        # Token bucket state
        self._tokens: float = self.config.max_burst
        self._last_update: float = time.monotonic()
        self._lock = asyncio.Lock()
        
        # Request queue
        self._queue: deque[asyncio.Future] = deque()
        self._queue_size: int = 0
        
        # Statistics
        self._total_requests: int = 0
        self._throttled_requests: int = 0
        self._dropped_requests: int = 0
        
        # Running state
        self._running: bool = False
        self._refill_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the rate limiter background tasks."""
        if self._running:
            return
        
        self._running = True
        self._refill_task = asyncio.create_task(self._refill_loop())
        logger.debug("Rate limiter started")
    
    async def stop(self) -> None:
        """Stop the rate limiter and clean up resources."""
        if not self._running:
            return
        
        self._running = False
        
        if self._refill_task:
            self._refill_task.cancel()
            try:
                await self._refill_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all pending queue items
        while self._queue:
            future = self._queue.popleft()
            if not future.done():
                future.set_exception(asyncio.CancelledError("Rate limiter stopped"))
        
        logger.debug("Rate limiter stopped")
    
    async def _refill_loop(self) -> None:
        """Background task to refill tokens at the configured rate."""
        while self._running:
            try:
                await asyncio.sleep(0.1)  # Update every 100ms
                
                async with self._lock:
                    now = time.monotonic()
                    elapsed = now - self._last_update
                    self._last_update = now
                    
                    # Add tokens based on elapsed time
                    tokens_to_add = elapsed * self.config.max_requests_per_second
                    self._tokens = min(
                        self.config.max_burst,
                        self._tokens + tokens_to_add
                    )
                    
                    # Process queued requests
                    await self._process_queue()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limiter refill loop: {e}")
    
    async def _process_queue(self) -> None:
        """Process queued requests when tokens are available."""
        while self._queue and self._tokens >= 1.0:
            future = self._queue.popleft()
            self._queue_size -= 1
            
            if not future.done():
                self._tokens -= 1.0
                future.set_result(True)
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait for a token. Uses config default if None.
        
        Returns:
            True if permission granted, False if timed out.
        
        Raises:
            asyncio.TimeoutError: If timeout is reached waiting for a token.
        """
        timeout = timeout or self.config.queue_timeout
        
        async with self._lock:
            self._total_requests += 1
            
            # Check if we can proceed immediately
            if self._tokens >= 1.0 and not self._queue:
                self._tokens -= 1.0
                return True
            
            # Check queue capacity
            if self._queue_size >= self.config.max_queue_size:
                self._dropped_requests += 1
                logger.warning("Rate limiter queue full, dropping request")
                raise asyncio.TimeoutError("Rate limiter queue is full")
            
            self._throttled_requests += 1
        
        # Queue the request
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            self._queue.append(future)
            self._queue_size += 1
        
        try:
            await asyncio.wait_for(future, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            # Remove from queue if still there
            async with self._lock:
                if future in self._queue:
                    self._queue.remove(future)
                    self._queue_size -= 1
            if not future.done():
                future.cancel()
            raise
    
    def release(self) -> None:
        """
        Release a token back to the bucket.
        
        Note: In the token bucket algorithm, tokens are typically not returned.
        This method exists for compatibility with semaphore-like patterns.
        """
        # Token bucket doesn't return tokens, but we could implement
        # a more complex algorithm that does
        pass
    
    def calculate_backoff_delay(
        self,
        attempt: int,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None
    ) -> float:
        """
        Calculate exponential backoff delay for retries.
        
        Args:
            attempt: The current retry attempt (0-indexed).
            base_delay: Base delay in seconds. Uses config default if None.
            max_delay: Maximum delay in seconds. Uses config default if None.
        
        Returns:
            Delay in seconds before the next retry attempt.
        """
        base = base_delay or self.config.base_delay
        max_d = max_delay or self.config.max_delay
        
        delay = base * (self.config.exponential_base ** attempt)
        return min(delay, max_d)
    
    async def execute_with_rate_limit(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with rate limiting.
        
        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            The result of the function execution.
        """
        await self.acquire()
        try:
            return await func(*args, **kwargs)
        finally:
            self.release()
    
    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        max_retries: int = 3,
        retry_exceptions: tuple = (Exception,),
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with rate limiting and retry logic.
        
        Args:
            func: The function to execute.
            max_retries: Maximum number of retry attempts.
            retry_exceptions: Tuple of exception types to retry on.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            The result of the function execution.
        
        Raises:
            The last exception if all retries are exhausted.
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                await self.acquire()
                try:
                    return await func(*args, **kwargs)
                finally:
                    self.release()
                    
            except retry_exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self.calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Function failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Function failed after {max_retries + 1} attempts: {e}")
        
        raise last_exception
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with rate limiting statistics.
        """
        return {
            'total_requests': self._total_requests,
            'throttled_requests': self._throttled_requests,
            'dropped_requests': self._dropped_requests,
            'queue_size': self._queue_size,
            'available_tokens': self._tokens,
            'throttle_rate': (
                self._throttled_requests / max(self._total_requests, 1)
            ),
        }
    
    def reset_stats(self) -> None:
        """Reset rate limiter statistics."""
        self._total_requests = 0
        self._throttled_requests = 0
        self._dropped_requests = 0


class EndpointRateLimiter:
    """
    Per-endpoint rate limiter for different API endpoints with different limits.
    
    Some exchanges have different rate limits for different endpoints
    (e.g., trading endpoints vs. market data endpoints).
    """
    
    def __init__(self):
        """Initialize the endpoint rate limiter."""
        self._limiters: dict[str, RateLimiter] = {}
        self._default_limiter: Optional[RateLimiter] = None
    
    def add_endpoint(
        self,
        endpoint: str,
        config: Optional[RateLimitConfig] = None
    ) -> RateLimiter:
        """
        Add a rate limiter for a specific endpoint.
        
        Args:
            endpoint: The endpoint identifier (e.g., 'public', 'private', 'trading').
            config: Rate limiting configuration for this endpoint.
        
        Returns:
            The created RateLimiter instance.
        """
        limiter = RateLimiter(config)
        self._limiters[endpoint] = limiter
        return limiter
    
    def set_default_limiter(self, limiter: RateLimiter) -> None:
        """
        Set the default rate limiter for endpoints without specific limits.
        
        Args:
            limiter: The default RateLimiter instance.
        """
        self._default_limiter = limiter
    
    def get_limiter(self, endpoint: str) -> RateLimiter:
        """
        Get the rate limiter for a specific endpoint.
        
        Args:
            endpoint: The endpoint identifier.
        
        Returns:
            The RateLimiter for the endpoint, or the default limiter.
        """
        return self._limiters.get(endpoint, self._default_limiter)
    
    async def start_all(self) -> None:
        """Start all rate limiters."""
        if self._default_limiter:
            await self._default_limiter.start()
        
        for limiter in self._limiters.values():
            await limiter.start()
    
    async def stop_all(self) -> None:
        """Stop all rate limiters."""
        if self._default_limiter:
            await self._default_limiter.stop()
        
        for limiter in self._limiters.values():
            await limiter.stop()
