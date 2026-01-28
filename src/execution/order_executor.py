"""
Order executor module for placing and managing orders.

This module provides the OrderExecutor class for placing post-only limit orders,
market orders, and managing order lifecycle with retry logic and error handling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Callable, List

# Exception classes defined locally to avoid import issues
class ExecutionError(Exception):
    """Base exception for execution errors."""
    pass

class ExecutionInvalidOrderError(ExecutionError):
    """Invalid order error."""
    pass

class ExecutionInsufficientFundsError(ExecutionError):
    """Insufficient funds error."""
    pass

class ExecutionOrderNotFoundError(ExecutionError):
    """Order not found error."""
    pass

class ExecutionNetworkError(ExecutionError):
    """Network error."""
    pass

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "limit"
    MARKET = "market"
    POST_ONLY_LIMIT = "post_only_limit"


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    order_type: Optional[str] = None
    amount: Optional[float] = None
    price: Optional[float] = None
    filled: float = 0.0
    remaining: float = 0.0
    status: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED.value or (
            self.filled > 0 and self.remaining == 0
        )
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (open or partially filled)."""
        return self.status in (
            OrderStatus.OPEN.value,
            OrderStatus.PARTIALLY_FILLED.value,
            OrderStatus.PENDING.value
        )


@dataclass
class ExecutorConfig:
    """Configuration for OrderExecutor."""
    max_retries: int = 3
    retry_delay_base: float = 1.0
    default_timeout: float = 60.0
    post_only_retry_attempts: int = 3
    post_only_retry_delay: float = 0.5
    enable_price_adjustment: bool = True
    price_adjustment_step: float = 0.005  # 0.005% adjustment per retry
    max_price_adjustment: float = 0.02    # Max 0.02% adjustment


class OrderExecutor:
    """
    Order executor for placing and managing orders.
    
    This class handles:
    - Post-only limit orders (primary)
    - Regular limit orders (fallback)
    - Market orders (emergency/close only)
    - Order lifecycle management
    - Retry logic with price adjustment
    - Order cancellation and modification
    
    Example:
        ```python
        from execution.order_executor import OrderExecutor
        from execution.price_calculator import PriceCalculator
        from exchange.exchange_client import ExchangeClient
        
        exchange = ExchangeClient(config)
        price_calc = PriceCalculator()
        executor = OrderExecutor(exchange, price_calc)
        
        # Place post-only limit order
        result = await executor.place_limit_order(
            symbol='BTC/USD',
            side='buy',
            amount=0.1,
            price=50000,
            post_only=True
        )
        
        # Wait for fill
        filled = await executor.wait_for_fill(result.order_id, 'BTC/USD', timeout=60)
        ```
    """
    
    def __init__(
        self,
        exchange_client: Any,
        price_calculator: Any,
        config: Optional[ExecutorConfig] = None
    ):
        """
        Initialize the order executor.
        
        Args:
            exchange_client: Exchange client instance for API calls.
            price_calculator: PriceCalculator instance for price calculations.
            config: Executor configuration. Uses defaults if not provided.
        """
        self.exchange = exchange_client
        self.price_calc = price_calculator
        self.config = config or ExecutorConfig()
        self._pending_orders: Dict[str, Dict[str, Any]] = {}
        
        logger.info("OrderExecutor initialized")
    
    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        post_only: bool = True,
        time_in_force: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> OrderResult:
        """
        Place a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD').
            side: Order side ('buy' or 'sell').
            amount: Order amount in base currency.
            price: Limit price.
            post_only: If True, order will be rejected if it would match immediately.
            time_in_force: Time in force (e.g., 'GTC', 'IOC', 'FOK').
            params: Additional exchange-specific parameters.
        
        Returns:
            OrderResult with order details.
        """
        params = params or {}
        side = side.lower()
        
        # Validate inputs
        if side not in ('buy', 'sell'):
            return OrderResult(
                success=False,
                error_message=f"Invalid side: {side}. Must be 'buy' or 'sell'."
            )
        
        if amount <= 0:
            return OrderResult(
                success=False,
                error_message=f"Invalid amount: {amount}. Must be positive."
            )
        
        if price <= 0:
            return OrderResult(
                success=False,
                error_message=f"Invalid price: {price}. Must be positive."
            )
        
        # Add post-only flag
        if post_only:
            params['postOnly'] = True
        
        if time_in_force:
            params['timeInForce'] = time_in_force
        
        logger.info(
            f"Placing {'post-only ' if post_only else ''}limit order: "
            f"{side} {amount} {symbol} @ {price}"
        )
        
        # Attempt to place order with retries
        last_error = None
        current_price = price
        
        for attempt in range(self.config.max_retries):
            try:
                order = await self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=amount,
                    price=current_price,
                    params=params
                )
                
                result = self._parse_order_response(order, symbol, side, 'limit', amount, current_price)
                
                if result.success:
                    logger.info(f"Order placed successfully: {result.order_id}")
                    self._pending_orders[result.order_id] = {
                        'symbol': symbol,
                        'side': side,
                        'amount': amount,
                        'price': current_price,
                        'placed_at': time.time(),
                        'post_only': post_only
                    }
                    return result
                else:
                    logger.warning(f"Order placement returned unsuccessful: {result.error_message}")
                    
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if post-only rejection
                if post_only and ('post only' in error_str or 'would match' in error_str or 'invalid order' in error_str):
                    logger.warning(f"Post-only order rejected (attempt {attempt + 1}): {e}")
                    
                    if self.config.enable_price_adjustment and attempt < self.config.max_retries - 1:
                        # Adjust price and retry
                        current_price = self._adjust_price_for_retry(side, current_price, attempt)
                        logger.info(f"Adjusting price to {current_price} and retrying...")
                        await asyncio.sleep(self.config.post_only_retry_delay)
                        continue
                
                # Check for specific error types
                if 'insufficient fund' in error_str:
                    logger.error(f"Insufficient funds: {e}")
                    return OrderResult(
                        success=False,
                        error_message=f"Insufficient funds: {str(e)}"
                    )
                
                if 'invalid order' in error_str:
                    logger.error(f"Invalid order error: {e}")
                    return OrderResult(
                        success=False,
                        error_message=f"Invalid order: {str(e)}"
                    )
                
                # Network or other exchange errors - retry
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_base * (2 ** attempt)
                    logger.warning(f"Order placement failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Order placement failed after {self.config.max_retries} attempts: {e}")
                    return OrderResult(
                        success=False,
                        error_message=f"Failed after {self.config.max_retries} attempts: {str(e)}"
                    )
        
        return OrderResult(
            success=False,
            error_message=f"Failed to place order: {str(last_error) if last_error else 'Unknown error'}"
        )
    
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None
    ) -> OrderResult:
        """
        Place a market order (emergency/close only).
        
        Market orders execute immediately at the best available price.
        Use sparingly as they incur taker fees.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD').
            side: Order side ('buy' or 'sell').
            amount: Order amount in base currency.
            params: Additional exchange-specific parameters.
        
        Returns:
            OrderResult with order details.
        """
        params = params or {}
        side = side.lower()
        
        if side not in ('buy', 'sell'):
            return OrderResult(
                success=False,
                error_message=f"Invalid side: {side}. Must be 'buy' or 'sell'."
            )
        
        if amount <= 0:
            return OrderResult(
                success=False,
                error_message=f"Invalid amount: {amount}. Must be positive."
            )
        
        logger.warning(
            f"Placing MARKET order (taker fees apply): {side} {amount} {symbol}"
        )
        
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params=params
            )
            
            result = self._parse_order_response(order, symbol, side, 'market', amount, None)
            
            if result.success:
                logger.info(f"Market order executed: {result.order_id}, filled: {result.filled}")
            
            return result
            
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return OrderResult(
                success=False,
                error_message=f"Market order failed: {str(e)}"
            )
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        params: Optional[Dict[str, Any]] = None
    ) -> OrderResult:
        """
        Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel.
            symbol: Trading pair symbol.
            params: Additional exchange-specific parameters.
        
        Returns:
            OrderResult with cancellation result.
        """
        params = params or {}
        
        logger.info(f"Cancelling order {order_id} for {symbol}")
        
        try:
            result = await self.exchange.cancel_order(order_id, symbol, params)
            
            # Remove from pending orders
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            
            return OrderResult(
                success=True,
                order_id=order_id,
                symbol=symbol,
                status=OrderStatus.CANCELLED.value,
                raw_response=result,
                timestamp=time.time()
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if 'not found' in error_str:
                logger.warning(f"Order {order_id} not found for cancellation: {e}")
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    symbol=symbol,
                    error_message=f"Order not found: {str(e)}"
                )
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                symbol=symbol,
                error_message=f"Cancel failed: {str(e)}"
            )
    
    async def cancel_all_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[OrderResult]:
        """
        Cancel all open orders for a symbol (or all symbols).
        
        Args:
            symbol: Trading pair symbol. If None, cancels all orders.
        
        Returns:
            List of OrderResult for each cancellation attempt.
        """
        logger.info(f"Cancelling all orders for {symbol or 'all symbols'}")
        
        results = []
        
        try:
            # Fetch open orders
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
            if not open_orders:
                logger.info("No open orders to cancel")
                return results
            
            logger.info(f"Found {len(open_orders)} open orders to cancel")
            
            # Cancel each order
            for order in open_orders:
                order_id = order.get('id')
                order_symbol = order.get('symbol', symbol)
                
                if order_id and order_symbol:
                    result = await self.cancel_order(order_id, order_symbol)
                    results.append(result)
            
            # Clear pending orders tracking
            if symbol:
                self._pending_orders = {
                    k: v for k, v in self._pending_orders.items()
                    if v.get('symbol') != symbol
                }
            else:
                self._pending_orders.clear()
            
        except Exception as e:
            logger.error(f"Failed to fetch/cancel orders: {e}")
            results.append(OrderResult(
                success=False,
                error_message=f"Failed to cancel all orders: {str(e)}"
            ))
        
        return results
    
    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        new_price: float,
        params: Optional[Dict[str, Any]] = None
    ) -> OrderResult:
        """
        Modify an existing order by cancelling and replacing it.
        
        Note: Not all exchanges support true order modification.
        This implementation cancels the old order and places a new one.
        
        Args:
            order_id: The ID of the order to modify.
            symbol: Trading pair symbol.
            new_price: New limit price.
            params: Additional exchange-specific parameters.
        
        Returns:
            OrderResult with new order details.
        """
        params = params or {}
        
        logger.info(f"Modifying order {order_id} for {symbol} to price {new_price}")
        
        # First, get current order details
        try:
            current_order = await self.get_order_status(order_id, symbol)
            
            if not current_order.success:
                return OrderResult(
                    success=False,
                    error_message=f"Could not fetch order details: {current_order.error_message}"
                )
            
            if not current_order.is_active:
                return OrderResult(
                    success=False,
                    error_message=f"Order is not active (status: {current_order.status})"
                )
            
            # Cancel the existing order
            cancel_result = await self.cancel_order(order_id, symbol)
            
            if not cancel_result.success:
                return OrderResult(
                    success=False,
                    error_message=f"Failed to cancel existing order: {cancel_result.error_message}"
                )
            
            # Place new order with remaining amount
            remaining = current_order.remaining if current_order.remaining > 0 else current_order.amount
            
            if remaining and remaining > 0:
                new_result = await self.place_limit_order(
                    symbol=symbol,
                    side=current_order.side or 'buy',
                    amount=remaining,
                    price=new_price,
                    post_only=True,
                    params=params
                )
                
                logger.info(f"Order modified: old={order_id}, new={new_result.order_id}")
                return new_result
            else:
                return OrderResult(
                    success=False,
                    error_message="Order already filled, nothing to modify"
                )
                
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return OrderResult(
                success=False,
                error_message=f"Modify failed: {str(e)}"
            )
    
    async def get_order_status(
        self,
        order_id: str,
        symbol: str,
        params: Optional[Dict[str, Any]] = None
    ) -> OrderResult:
        """
        Get the current status of an order.
        
        Args:
            order_id: The ID of the order to check.
            symbol: Trading pair symbol.
            params: Additional exchange-specific parameters.
        
        Returns:
            OrderResult with current order status.
        """
        params = params or {}
        
        try:
            order = await self.exchange.fetch_order(order_id, symbol, params)
            return self._parse_order_response(order, symbol, None, None, None, None)
            
        except Exception as e:
            error_str = str(e).lower()
            if 'not found' in error_str:
                logger.warning(f"Order {order_id} not found: {e}")
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    symbol=symbol,
                    status=OrderStatus.CANCELLED.value,
                    error_message=f"Order not found: {str(e)}"
                )
            logger.error(f"Failed to fetch order status for {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                symbol=symbol,
                error_message=f"Failed to fetch status: {str(e)}"
            )
    
    async def wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: Optional[float] = None,
        poll_interval: float = 1.0,
        params: Optional[Dict[str, Any]] = None
    ) -> OrderResult:
        """
        Wait for an order to be filled.
        
        Polls the order status until it's filled, cancelled, or times out.
        
        Args:
            order_id: The ID of the order to wait for.
            symbol: Trading pair symbol.
            timeout: Maximum time to wait in seconds. Uses default if None.
            poll_interval: Time between status checks in seconds.
            params: Additional exchange-specific parameters.
        
        Returns:
            OrderResult with final order status.
        """
        timeout = timeout or self.config.default_timeout
        params = params or {}
        
        logger.info(f"Waiting for order {order_id} to fill (timeout: {timeout}s)")
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            result = await self.get_order_status(order_id, symbol, params)
            
            if not result.success and result.status != OrderStatus.CANCELLED.value:
                logger.error(f"Failed to get order status: {result.error_message}")
                await asyncio.sleep(poll_interval)
                continue
            
            # Log status changes
            if result.status != last_status:
                logger.info(f"Order {order_id} status: {result.status}, filled: {result.filled}/{result.amount}")
                last_status = result.status
            
            # Check if order is complete
            if result.is_filled:
                logger.info(f"Order {order_id} completely filled: {result.filled}")
                return result
            
            if not result.is_active:
                logger.info(f"Order {order_id} no longer active (status: {result.status})")
                return result
            
            await asyncio.sleep(poll_interval)
        
        # Timeout reached
        logger.warning(f"Timeout waiting for order {order_id} to fill")
        
        # Get final status
        final_result = await self.get_order_status(order_id, symbol, params)
        final_result.error_message = f"Timeout after {timeout}s"
        
        return final_result
    
    def _parse_order_response(
        self,
        order: Dict[str, Any],
        symbol: Optional[str],
        side: Optional[str],
        order_type: Optional[str],
        amount: Optional[float],
        price: Optional[float]
    ) -> OrderResult:
        """
        Parse exchange order response into OrderResult.
        
        Args:
            order: Raw order response from exchange.
            symbol: Expected symbol.
            side: Expected side.
            order_type: Expected order type.
            amount: Expected amount.
            price: Expected price.
        
        Returns:
            Parsed OrderResult.
        """
        if not order:
            return OrderResult(success=False, error_message="Empty order response")
        
        # Extract fields with fallbacks
        order_id = order.get('id') or order.get('orderId')
        order_symbol = order.get('symbol', symbol)
        order_side = (order.get('side') or side or '').lower()
        order_type_parsed = (order.get('type') or order_type or '').lower()
        
        # Parse amounts
        try:
            order_amount = float(order.get('amount') or amount or 0)
        except (ValueError, TypeError):
            order_amount = amount or 0
        
        try:
            order_price = float(order.get('price') or price or 0)
        except (ValueError, TypeError):
            order_price = price or 0
        
        try:
            filled = float(order.get('filled') or 0)
        except (ValueError, TypeError):
            filled = 0
        
        try:
            remaining = float(order.get('remaining') or 0)
        except (ValueError, TypeError):
            remaining = order_amount - filled if order_amount > 0 else 0
        
        # Parse status
        status = order.get('status', 'unknown')
        
        # Map common status values
        status_mapping = {
            'open': OrderStatus.OPEN.value,
            'closed': OrderStatus.FILLED.value,
            'canceled': OrderStatus.CANCELLED.value,
            'cancelled': OrderStatus.CANCELLED.value,
            'pending': OrderStatus.PENDING.value,
            'expired': OrderStatus.EXPIRED.value,
            'rejected': OrderStatus.REJECTED.value,
        }
        normalized_status = status_mapping.get(status.lower(), status)
        
        # Determine if partially filled
        if filled > 0 and remaining > 0:
            normalized_status = OrderStatus.PARTIALLY_FILLED.value
        
        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=order_symbol,
            side=order_side,
            order_type=order_type_parsed,
            amount=order_amount,
            price=order_price,
            filled=filled,
            remaining=remaining,
            status=normalized_status,
            raw_response=order,
            timestamp=time.time()
        )
    
    def _adjust_price_for_retry(
        self,
        side: str,
        current_price: float,
        attempt: int
    ) -> float:
        """
        Adjust price for post-only retry.
        
        When a post-only order is rejected, we adjust the price further
        from the market to ensure it won't match immediately.
        
        Args:
            side: Order side ('buy' or 'sell').
            current_price: Current price.
            attempt: Retry attempt number.
        
        Returns:
            Adjusted price.
        """
        adjustment = self.config.price_adjustment_step * (attempt + 1)
        adjustment = min(adjustment, self.config.max_price_adjustment)
        
        adjustment_multiplier = adjustment / 100.0
        
        if side == 'buy':
            # Move buy price lower
            new_price = current_price * (1 - adjustment_multiplier)
        else:
            # Move sell price higher
            new_price = current_price * (1 + adjustment_multiplier)
        
        logger.debug(f"Price adjusted from {current_price} to {new_price} (attempt {attempt + 1})")
        
        return new_price
    
    def get_pending_orders(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all pending orders being tracked.
        
        Returns:
            Dictionary of order_id -> order info.
        """
        return self._pending_orders.copy()
