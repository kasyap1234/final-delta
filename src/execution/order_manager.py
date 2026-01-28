"""
Order manager module for tracking and managing order lifecycle.

This module provides the OrderManager class for submitting orders,
tracking their status, handling partial fills, and maintaining order history.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from collections import defaultdict

from .order_executor import OrderExecutor, OrderResult, OrderStatus

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Extended order state for internal tracking."""
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class OrderRequest:
    """Request to submit a new order."""
    symbol: str
    side: str
    amount: float
    price: Optional[float] = None
    order_type: str = "limit"  # 'limit', 'market', 'post_only_limit'
    post_only: bool = True
    time_in_force: Optional[str] = "GTC"
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.side = self.side.lower()
        self.order_type = self.order_type.lower()


@dataclass
class TrackedOrder:
    """Order being tracked by the manager."""
    order_id: str
    request: OrderRequest
    state: OrderState
    symbol: str
    side: str
    amount: float
    price: Optional[float]
    filled: float = 0.0
    remaining: float = 0.0
    avg_fill_price: Optional[float] = None
    fee: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.state in (
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
        )
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or cancelled)."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        )
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.amount == 0:
            return 0.0
        return (self.filled / self.amount) * 100.0


@dataclass
class ManagerConfig:
    """Configuration for OrderManager."""
    update_interval: float = 1.0  # Seconds between order status updates
    max_open_orders: int = 100
    enable_auto_update: bool = True
    partial_fill_threshold: float = 0.01  # 1% threshold to consider partial fill
    order_timeout: Optional[float] = None  # Auto-cancel after timeout (None = no timeout)
    fee_tracking_enabled: bool = True


class OrderManager:
    """
    Manager for order lifecycle and tracking.
    
    This class provides:
    - Order submission and tracking
    - Automatic order status updates
    - Partial fill handling
    - Order history management
    - Fee tracking
    
    Example:
        ```python
        from execution.order_manager import OrderManager, OrderRequest
        from execution.order_executor import OrderExecutor
        
        executor = OrderExecutor(exchange, price_calc)
        manager = OrderManager(executor)
        
        # Submit an order
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            amount=0.1,
            price=50000,
            post_only=True
        )
        order = await manager.submit_order(request)
        
        # Get order updates
        await manager.update_orders()
        
        # Handle partial fills
        if order.state == OrderState.PARTIALLY_FILLED:
            await manager.handle_partial_fill(order.order_id, order.filled)
        ```
    """
    
    def __init__(
        self,
        executor: OrderExecutor,
        config: Optional[ManagerConfig] = None
    ):
        """
        Initialize the order manager.
        
        Args:
            executor: OrderExecutor instance for placing orders.
            config: Manager configuration. Uses defaults if not provided.
        """
        self.executor = executor
        self.config = config or ManagerConfig()
        
        # Order tracking
        self._orders: Dict[str, TrackedOrder] = {}
        self._order_history: List[TrackedOrder] = []
        self._symbol_orders: Dict[str, set] = defaultdict(set)
        
        # Callbacks
        self._on_fill_callbacks: List[Callable[[TrackedOrder], None]] = []
        self._on_partial_fill_callbacks: List[Callable[[TrackedOrder, float], None]] = []
        self._on_cancel_callbacks: List[Callable[[TrackedOrder], None]] = []
        
        # Update task
        self._update_task: Optional[asyncio.Task] = None
        self._running: bool = False
        
        # Statistics
        self._stats = {
            'total_submitted': 0,
            'total_filled': 0,
            'total_cancelled': 0,
            'total_rejected': 0,
            'total_fees': 0.0,
        }
        
        logger.info("OrderManager initialized")
    
    async def start(self) -> None:
        """Start the order manager background tasks."""
        if self._running:
            return
        
        self._running = True
        
        if self.config.enable_auto_update:
            self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info("OrderManager started")
    
    async def stop(self) -> None:
        """Stop the order manager background tasks."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        logger.info("OrderManager stopped")
    
    async def submit_order(
        self,
        order_request: OrderRequest,
        wait_for_fill: bool = False,
        timeout: Optional[float] = None
    ) -> TrackedOrder:
        """
        Submit a new order and track it.
        
        Args:
            order_request: The order request to submit.
            wait_for_fill: If True, wait for order to fill before returning.
            timeout: Timeout for waiting (if wait_for_fill is True).
        
        Returns:
            TrackedOrder with order details.
        """
        logger.info(
            f"Submitting {order_request.order_type} order: "
            f"{order_request.side} {order_request.amount} {order_request.symbol}"
        )
        
        self._stats['total_submitted'] += 1
        
        # Create tracking entry
        temp_id = order_request.client_order_id or f"temp_{time.time()}"
        tracked = TrackedOrder(
            order_id=temp_id,
            request=order_request,
            state=OrderState.SUBMITTING,
            symbol=order_request.symbol,
            side=order_request.side,
            amount=order_request.amount,
            price=order_request.price,
            remaining=order_request.amount
        )
        self._orders[temp_id] = tracked
        self._symbol_orders[order_request.symbol].add(temp_id)
        
        try:
            # Place the order
            if order_request.order_type == 'market':
                result = await self.executor.place_market_order(
                    symbol=order_request.symbol,
                    side=order_request.side,
                    amount=order_request.amount
                )
            else:
                result = await self.executor.place_limit_order(
                    symbol=order_request.symbol,
                    side=order_request.side,
                    amount=order_request.amount,
                    price=order_request.price or 0,
                    post_only=order_request.post_only,
                    time_in_force=order_request.time_in_force
                )
            
            if result.success:
                # Update tracking with real order ID
                old_id = tracked.order_id
                tracked.order_id = result.order_id
                tracked.state = OrderState.OPEN
                tracked.filled = result.filled
                tracked.remaining = result.remaining
                tracked.updated_at = time.time()
                
                # Update indices
                del self._orders[old_id]
                self._orders[tracked.order_id] = tracked
                self._symbol_orders[order_request.symbol].remove(old_id)
                self._symbol_orders[order_request.symbol].add(tracked.order_id)
                
                logger.info(f"Order submitted successfully: {tracked.order_id}")
                
                # Check if already filled
                if result.is_filled:
                    tracked.state = OrderState.FILLED
                    tracked.filled = result.amount or order_request.amount
                    tracked.remaining = 0
                    tracked.closed_at = time.time()
                    self._stats['total_filled'] += 1
                    self._trigger_fill_callbacks(tracked)
                elif result.filled > 0:
                    tracked.state = OrderState.PARTIALLY_FILLED
                    self._trigger_partial_fill_callbacks(tracked, result.filled)
                
                # Wait for fill if requested
                if wait_for_fill and tracked.is_active:
                    await self.wait_for_order_fill(tracked.order_id, timeout)
                
            else:
                # Order submission failed
                tracked.state = OrderState.REJECTED
                tracked.error_message = result.error_message
                tracked.closed_at = time.time()
                self._stats['total_rejected'] += 1
                logger.error(f"Order submission failed: {result.error_message}")
            
            return tracked
            
        except Exception as e:
            tracked.state = OrderState.ERROR
            tracked.error_message = str(e)
            tracked.closed_at = time.time()
            logger.error(f"Exception submitting order: {e}")
            return tracked
    
    async def update_orders(self) -> None:
        """
        Poll and update status of all active orders.
        
        This method fetches the latest status for all open orders
        and updates their tracking information.
        """
        active_orders = [
            order for order in self._orders.values()
            if order.is_active
        ]
        
        if not active_orders:
            return
        
        logger.debug(f"Updating {len(active_orders)} active orders")
        
        for order in active_orders:
            try:
                result = await self.executor.get_order_status(order.order_id, order.symbol)
                
                if result.success:
                    self._update_order_from_result(order, result)
                else:
                    # Order not found - might have been cancelled externally
                    error_msg = (result.error_message or '').lower()
                    if 'not found' in error_msg:
                        logger.warning(f"Order {order.order_id} not found, marking as cancelled")
                        order.state = OrderState.CANCELLED
                        order.closed_at = time.time()
                        self._move_to_history(order)
                    else:
                        logger.warning(f"Failed to update order {order.order_id}: {result.error_message}")
                        
            except Exception as e:
                logger.error(f"Error updating order {order.order_id}: {e}")
    
    async def handle_partial_fill(
        self,
        order_id: str,
        filled_amount: float
    ) -> Optional[TrackedOrder]:
        """
        Handle a partial fill event.
        
        This method can be called when a partial fill is detected,
        or can be overridden to implement custom partial fill logic.
        
        Args:
            order_id: The ID of the partially filled order.
            filled_amount: The amount that has been filled.
        
        Returns:
            Updated TrackedOrder or None if not found.
        """
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Cannot handle partial fill for unknown order: {order_id}")
            return None
        
        logger.info(
            f"Handling partial fill for {order_id}: "
            f"{filled_amount}/{order.amount} ({order.fill_percentage:.2f}%)"
        )
        
        # Update order state
        previous_filled = order.filled
        order.filled = filled_amount
        order.remaining = order.amount - filled_amount
        order.state = OrderState.PARTIALLY_FILLED if order.remaining > 0 else OrderState.FILLED
        order.updated_at = time.time()
        
        # Calculate newly filled amount
        newly_filled = filled_amount - previous_filled
        
        if newly_filled > 0:
            # Record the fill
            order.fills.append({
                'amount': newly_filled,
                'timestamp': time.time(),
            })
            
            # Trigger callbacks
            self._trigger_partial_fill_callbacks(order, newly_filled)
            
            # Check if now completely filled
            if order.remaining <= 0:
                order.state = OrderState.FILLED
                order.closed_at = time.time()
                self._stats['total_filled'] += 1
                self._trigger_fill_callbacks(order)
                self._move_to_history(order)
        
        return order
    
    async def cancel_order(
        self,
        order_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: The ID of the order to cancel.
            reason: Optional reason for cancellation.
        
        Returns:
            True if cancellation was successful.
        """
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Cannot cancel unknown order: {order_id}")
            return False
        
        if not order.is_active:
            logger.warning(f"Cannot cancel order {order_id} in state {order.state.value}")
            return False
        
        logger.info(f"Cancelling order {order_id}" + (f" (reason: {reason})" if reason else ""))
        
        order.state = OrderState.CANCELLING
        
        result = await self.executor.cancel_order(order_id, order.symbol)
        
        if result.success:
            order.state = OrderState.CANCELLED
            order.closed_at = time.time()
            order.remaining = 0  # Cancelled orders have no remaining
            self._stats['total_cancelled'] += 1
            self._trigger_cancel_callbacks(order)
            self._move_to_history(order)
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        else:
            # Revert state if cancellation failed
            order.state = OrderState.OPEN
            logger.error(f"Failed to cancel order {order_id}: {result.error_message}")
            return False
    
    async def cancel_all_orders(
        self,
        symbol: Optional[str] = None,
        reason: Optional[str] = None
    ) -> int:
        """
        Cancel all active orders for a symbol (or all symbols).
        
        Args:
            symbol: Symbol to cancel orders for. If None, cancels all.
            reason: Optional reason for cancellation.
        
        Returns:
            Number of orders cancelled.
        """
        if symbol:
            order_ids = [
                oid for oid in self._symbol_orders.get(symbol, set())
                if oid in self._orders and self._orders[oid].is_active
            ]
        else:
            order_ids = [
                oid for oid, order in self._orders.items()
                if order.is_active
            ]
        
        logger.info(f"Cancelling {len(order_ids)} orders" + (f" for {symbol}" if symbol else ""))
        
        cancelled_count = 0
        for order_id in order_ids:
            if await self.cancel_order(order_id, reason):
                cancelled_count += 1
        
        return cancelled_count
    
    def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[TrackedOrder]:
        """
        Get all open orders.
        
        Args:
            symbol: Filter by symbol. If None, returns all open orders.
        
        Returns:
            List of active TrackedOrder objects.
        """
        if symbol:
            return [
                self._orders[oid] for oid in self._symbol_orders.get(symbol, set())
                if oid in self._orders and self._orders[oid].is_active
            ]
        else:
            return [order for order in self._orders.values() if order.is_active]
    
    def get_order(self, order_id: str) -> Optional[TrackedOrder]:
        """
        Get a specific order by ID.
        
        Args:
            order_id: The order ID.
        
        Returns:
            TrackedOrder or None if not found.
        """
        return self._orders.get(order_id)
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TrackedOrder]:
        """
        Get filled/cancelled order history.
        
        Args:
            symbol: Filter by symbol.
            limit: Maximum number of orders to return.
        
        Returns:
            List of completed TrackedOrder objects.
        """
        history = self._order_history
        
        if symbol:
            history = [o for o in history if o.symbol == symbol]
        
        # Sort by closed time (most recent first)
        history = sorted(
            history,
            key=lambda o: o.closed_at or 0,
            reverse=True
        )
        
        if limit:
            history = history[:limit]
        
        return history
    
    async def wait_for_order_fill(
        self,
        order_id: str,
        timeout: Optional[float] = None
    ) -> Optional[TrackedOrder]:
        """
        Wait for an order to be filled.
        
        Args:
            order_id: The order ID to wait for.
            timeout: Maximum time to wait in seconds.
        
        Returns:
            TrackedOrder when filled, or None if timeout/error.
        """
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Cannot wait for unknown order: {order_id}")
            return None
        
        if order.is_complete:
            return order
        
        timeout = timeout or 60.0
        start_time = time.time()
        poll_interval = min(self.config.update_interval, 1.0)
        
        while time.time() - start_time < timeout:
            await self.update_orders()
            
            order = self._orders.get(order_id)
            if not order:
                return None
            
            if order.is_complete:
                return order
            
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"Timeout waiting for order {order_id} to fill")
        return order
    
    def on_fill(self, callback: Callable[[TrackedOrder], None]) -> None:
        """
        Register a callback for order fill events.
        
        Args:
            callback: Function to call when an order is filled.
        """
        self._on_fill_callbacks.append(callback)
    
    def on_partial_fill(self, callback: Callable[[TrackedOrder, float], None]) -> None:
        """
        Register a callback for partial fill events.
        
        Args:
            callback: Function to call when an order is partially filled.
                      Receives (order, newly_filled_amount).
        """
        self._on_partial_fill_callbacks.append(callback)
    
    def on_cancel(self, callback: Callable[[TrackedOrder], None]) -> None:
        """
        Register a callback for order cancellation events.
        
        Args:
            callback: Function to call when an order is cancelled.
        """
        self._on_cancel_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get order statistics.
        
        Returns:
            Dictionary with order statistics.
        """
        stats = self._stats.copy()
        stats['open_orders'] = len(self.get_open_orders())
        stats['total_tracked'] = len(self._orders)
        stats['history_size'] = len(self._order_history)
        return stats
    
    def _update_order_from_result(self, order: TrackedOrder, result: OrderResult) -> None:
        """Update tracked order from executor result."""
        previous_filled = order.filled
        
        order.filled = result.filled
        order.remaining = result.remaining
        order.updated_at = time.time()
        
        # Update state based on status
        if result.status == OrderStatus.FILLED.value:
            order.state = OrderState.FILLED
            order.closed_at = time.time()
            order.remaining = 0
            self._stats['total_filled'] += 1
            self._trigger_fill_callbacks(order)
            self._move_to_history(order)
        elif result.status == OrderStatus.PARTIALLY_FILLED.value:
            order.state = OrderState.PARTIALLY_FILLED
            newly_filled = result.filled - previous_filled
            if newly_filled > 0:
                self._trigger_partial_fill_callbacks(order, newly_filled)
        elif result.status == OrderStatus.CANCELLED.value:
            order.state = OrderState.CANCELLED
            order.closed_at = time.time()
            self._stats['total_cancelled'] += 1
            self._trigger_cancel_callbacks(order)
            self._move_to_history(order)
        elif result.status == OrderStatus.REJECTED.value:
            order.state = OrderState.REJECTED
            order.closed_at = time.time()
            self._stats['total_rejected'] += 1
            self._move_to_history(order)
        elif result.status == OrderStatus.OPEN.value:
            order.state = OrderState.OPEN
    
    def _move_to_history(self, order: TrackedOrder) -> None:
        """Move a completed order to history."""
        if order.order_id in self._orders:
            del self._orders[order.order_id]
            self._symbol_orders[order.symbol].discard(order.order_id)
            self._order_history.append(order)
    
    def _trigger_fill_callbacks(self, order: TrackedOrder) -> None:
        """Trigger fill callbacks."""
        for callback in self._on_fill_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
    
    def _trigger_partial_fill_callbacks(self, order: TrackedOrder, amount: float) -> None:
        """Trigger partial fill callbacks."""
        for callback in self._on_partial_fill_callbacks:
            try:
                callback(order, amount)
            except Exception as e:
                logger.error(f"Error in partial fill callback: {e}")
    
    def _trigger_cancel_callbacks(self, order: TrackedOrder) -> None:
        """Trigger cancel callbacks."""
        for callback in self._on_cancel_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in cancel callback: {e}")
    
    async def _update_loop(self) -> None:
        """Background task for periodic order updates."""
        while self._running:
            try:
                await self.update_orders()
                await asyncio.sleep(self.config.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.config.update_interval)
