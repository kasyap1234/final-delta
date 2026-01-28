"""
Order simulator module for backtesting.

This module provides realistic order simulation functionality for the backtesting system,
including post-only order rejection, retry logic with exponential backoff, partial fills,
market impact modeling, and realistic fill probability based on order book depth.
"""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging

from src.backtest.config import BacktestConfig, SlippageModel
from src.backtest.account_state import AccountState
from src.backtest.mock.price_impact_model import PriceImpactModel, ImpactConfig
from src.backtest.market.latency_model import (
    LatencyModel, LatencyConfig, LatencyType, LatencyEvent
)
from src.backtest.market.order_book import (
    SimulatedOrderBook, OrderBookConfig
)

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order status types matching live trading."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(str, Enum):
    """Order type enumeration."""
    LIMIT = "limit"
    MARKET = "market"
    POST_ONLY_LIMIT = "post_only_limit"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Time in force options."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


@dataclass
class Order:
    """Order data matching live trading structure."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'limit', 'market', 'post_only_limit'
    amount: float
    price: Optional[float] = None
    post_only: bool = False
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.PENDING
    filled: float = 0.0
    remaining: float = 0.0
    average_fill_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    fee: float = 0.0
    fee_currency: str = "USD"
    client_order_id: Optional[str] = None
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    reject_reason: Optional[str] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize remaining amount."""
        if self.remaining == 0.0 and self.filled == 0.0:
            self.remaining = self.amount
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching exchange format."""
        return {
            'id': self.id,
            'orderId': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'type': self.order_type,
            'amount': self.amount,
            'price': self.price,
            'postOnly': self.post_only,
            'timeInForce': self.time_in_force,
            'status': self.status.value,
            'filled': self.filled,
            'remaining': self.remaining,
            'average_fill_price': self.average_fill_price,
            'timestamp': int(self.timestamp.timestamp() * 1000),
            'datetime': self.timestamp.isoformat(),
            'fee': self.fee,
            'fee_currency': self.fee_currency,
            'clientOrderId': self.client_order_id,
            'fills': self.fills
        }
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED
        )
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.amount == 0:
            return 0.0
        return (self.filled / self.amount) * 100.0


@dataclass
class OHLCV:
    """OHLCV candle data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.high + self.low) / 2
    
    @property
    def spread_pct(self) -> float:
        """Calculate spread percentage."""
        if self.close == 0:
            return 0.0
        return (self.high - self.low) / self.close


@dataclass
class SimulatorConfig:
    """Configuration for order simulator."""
    # Retry configuration
    max_retries: int = 3
    retry_delay_base: float = 1.0  # seconds
    retry_delay_max: float = 10.0  # seconds
    enable_price_adjustment: bool = True
    price_adjustment_step: float = 0.005  # 0.005% per retry
    max_price_adjustment: float = 0.02  # Max 0.02% adjustment
    
    # Post-only configuration
    post_only_reject_probability: float = 0.15  # 15% chance of rejection
    post_only_retry_attempts: int = 3
    post_only_retry_delay: float = 0.5
    
    # Partial fill configuration
    enable_partial_fills: bool = True
    partial_fill_min_pct: float = 0.1  # Min 10% fill
    partial_fill_max_pct: float = 0.9  # Max 90% fill per tick
    
    # Market impact configuration
    enable_market_impact: bool = True
    impact_config: ImpactConfig = field(default_factory=ImpactConfig)
    
    # Fill probability configuration
    base_fill_probability: float = 0.85
    distance_penalty_factor: float = 0.5  # Penalty per 1% from mid
    size_penalty_factor: float = 0.3  # Penalty per 10% of volume
    
    # Market order slippage
    market_order_slippage_bps: float = 5.0  # 5 bps slippage
    
    # Order book simulation
    order_book_depth_levels: int = 10
    order_book_volume_concentration: float = 0.6  # 60% at best levels
    
    # Latency configuration
    enable_latency: bool = True
    latency_config: LatencyConfig = field(default_factory=LatencyConfig)
    
    # Order book configuration
    order_book_config: OrderBookConfig = field(default_factory=OrderBookConfig)
    
    # Time-in-force configuration
    enable_tif_expiration: bool = True
    tif_check_interval_ms: float = 100.0  # Check TIF every 100ms
    
    # Order queue simulation
    enable_order_queue: bool = True
    queue_position_factor: float = 0.1  # Impact of queue position on fill priority


class BacktestOrderSimulator:
    """
    Realistic order execution simulator for backtesting.
    
    This class simulates live trading behavior including:
    - Post-only order rejection simulation
    - Retry logic with exponential backoff
    - Partial fills based on order book depth
    - Market impact modeling (price moves against large orders)
    - Realistic fill probability based on order size and distance from mid
    - Multiple order types: limit, market, post-only
    - Order status transitions: pending -> open -> partially_filled -> filled/cancelled
    - Slippage simulation for market orders
    
    Example:
        ```python
        config = SimulatorConfig()
        simulator = BacktestOrderSimulator(config, account_state)
        
        # Create order
        order = simulator.create_order(
            symbol='BTC/USD',
            side='buy',
            order_type='post_only_limit',
            amount=0.1,
            price=50000
        )
        
        # Process against candle data
        fills = simulator.process_orders(candle)
        ```
    """
    
    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        account_state: Optional[AccountState] = None
    ):
        """
        Initialize the order simulator.
        
        Args:
            config: Simulator configuration
            account_state: Account state tracker
        """
        self.config = config or SimulatorConfig()
        self.account_state = account_state
        self.impact_model = PriceImpactModel(self.config.impact_config)
        
        # Latency model
        self.latency_model = LatencyModel(self.config.latency_config)
        
        # Order book simulation
        self._order_books: Dict[str, SimulatedOrderBook] = {}
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        
        # Pending order events (for latency simulation)
        self._pending_submissions: Dict[str, LatencyEvent] = {}
        self._pending_cancellations: Dict[str, LatencyEvent] = {}
        self._pending_fills: Dict[str, List[LatencyEvent]] = {}
        
        # Order queue position tracking
        self._order_queue_positions: Dict[str, int] = {}
        self._queue_counter = 0
        
        # Fill tracking
        self._fills: List[Dict[str, Any]] = []
        
        # Callbacks
        self._on_fill_callbacks: List[Callable[[Order], None]] = []
        self._on_partial_fill_callbacks: List[Callable[[Order, float], None]] = []
        
        # Statistics
        self._stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'partially_filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'retried_orders': 0,
            'total_fills': 0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'latency_ms_total': 0.0,
            'latency_count': 0,
        }
        
        # Current simulation time
        self._current_time: Optional[datetime] = None
        
        logger.info("BacktestOrderSimulator initialized with latency and order book support")
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        post_only: bool = False,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            order_type: Order type ('limit', 'market', 'post_only_limit')
            amount: Order amount
            price: Order price (required for limit orders)
            post_only: Whether order is post-only
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            client_order_id: Client order identifier
            params: Additional parameters
            
        Returns:
            Created order
        """
        self._order_counter += 1
        order_id = f"order_{self._order_counter}_{int(time.time() * 1000)}"
        
        # Validate order type
        if order_type == 'limit' and price is None:
            raise ValueError("Price is required for limit orders")
        
        if order_type == 'market':
            price = None  # Market orders don't have a price
        
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side.lower(),
            order_type=order_type,
            amount=amount,
            price=price,
            post_only=post_only or order_type == 'post_only_limit',
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            status=OrderStatus.PENDING
        )
        
        self._stats['total_orders'] += 1
        
        logger.debug(
            f"Order created: {order_id} {side} {amount} {symbol} @ {price}, "
            f"post_only={post_only}"
        )
        
        return order
    
    def submit_order(self, order: Order, current_time: Optional[datetime] = None) -> Order:
        """
        Submit an order to the simulator.
        
        Args:
            order: Order to submit
            current_time: Current simulation time for latency calculation
            
        Returns:
            Submitted order
        """
        order.status = OrderStatus.PENDING
        self._orders[order.id] = order
        
        # Update current time
        if current_time:
            self._current_time = current_time
        
        # Simulate latency for order submission
        if self.config.enable_latency and current_time:
            event = self.latency_model.schedule_event(
                LatencyType.ORDER_SUBMIT,
                current_time,
                metadata={'order_id': order.id}
            )
            self._pending_submissions[order.id] = event
            
            # Track queue position
            if self.config.enable_order_queue:
                self._queue_counter += 1
                self._order_queue_positions[order.id] = self._queue_counter
        else:
            # Immediate submission
            order.status = OrderStatus.OPEN
        
        logger.debug(f"Order submitted: {order.id} {order.side} {order.amount} @ {order.price}")
        return order
    
    def _process_pending_submissions(self, current_time: datetime) -> List[Order]:
        """Process orders whose submission latency has elapsed."""
        completed_events = self.latency_model.process_events(current_time)
        processed_orders = []
        
        for event in completed_events:
            if event.event_type == LatencyType.ORDER_SUBMIT:
                order_id = event.metadata.get('order_id')
                if order_id and order_id in self._orders:
                    order = self._orders[order_id]
                    if order.status == OrderStatus.PENDING:
                        order.status = OrderStatus.OPEN
                        order.timestamp = current_time
                        processed_orders.append(order)
                        
                        # Update stats
                        self._stats['latency_ms_total'] += event.latency_ms
                        self._stats['latency_count'] += 1
        
        # Clean up processed submissions
        for order_id in list(self._pending_submissions.keys()):
            if self._pending_submissions[order_id].is_completed:
                del self._pending_submissions[order_id]
        
        return processed_orders
    
    def cancel_order(
        self,
        order_id: str,
        current_time: Optional[datetime] = None
    ) -> Optional[Order]:
        """
        Cancel an order.
        
        Args:
            order_id: Order identifier
            current_time: Current simulation time for latency calculation
            
        Returns:
            Cancelled order if found, None otherwise
        """
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return None
        
        # Simulate cancellation latency
        if self.config.enable_latency and current_time:
            event = self.latency_model.schedule_event(
                LatencyType.ORDER_CANCEL,
                current_time,
                metadata={'order_id': order_id}
            )
            self._pending_cancellations[order_id] = event
            
            # Mark as pending cancellation
            order.status = OrderStatus.PENDING
            order.metadata = getattr(order, 'metadata', {})
            order.metadata['pending_cancel'] = True
        else:
            # Immediate cancellation
            order.status = OrderStatus.CANCELLED
            order.remaining = 0.0
            self._stats['cancelled_orders'] += 1
        
        logger.debug(f"Order cancellation requested: {order_id}")
        return order
    
    def _process_pending_cancellations(self, current_time: datetime) -> List[Order]:
        """Process cancellations whose latency has elapsed."""
        completed_events = self.latency_model.process_events(current_time)
        cancelled_orders = []
        
        for event in completed_events:
            if event.event_type == LatencyType.ORDER_CANCEL:
                order_id = event.metadata.get('order_id')
                if order_id and order_id in self._orders:
                    order = self._orders[order_id]
                    # Only cancel if not already filled
                    if order.status != OrderStatus.FILLED:
                        order.status = OrderStatus.CANCELLED
                        order.remaining = 0.0
                        cancelled_orders.append(order)
                        self._stats['cancelled_orders'] += 1
        
        # Clean up processed cancellations
        for order_id in list(self._pending_cancellations.keys()):
            if self._pending_cancellations[order_id].is_completed:
                del self._pending_cancellations[order_id]
        
        return cancelled_orders
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order if found, None otherwise
        """
        return self._orders.get(order_id)
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of orders
        """
        orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open orders
        """
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def process_orders(
        self,
        candle: OHLCV,
        current_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all open orders against a candle.
        
        Args:
            candle: OHLCV candle to process orders against
            current_time: Current simulation time
            
        Returns:
            List of fills that occurred
        """
        if current_time:
            self._current_time = current_time
        else:
            current_time = candle.timestamp
        
        # Update order book for this symbol
        self._update_order_book(candle)
        
        # Process pending submissions and cancellations
        self._process_pending_submissions(current_time)
        self._process_pending_cancellations(current_time)
        
        fills = []
        
        for order in list(self._orders.values()):
            if not order.is_active:
                continue
            
            if order.symbol != candle.symbol:
                continue
            
            # Check time-in-force expiration
            if self.config.enable_tif_expiration:
                if self._is_tif_expired(order, current_time):
                    order.status = OrderStatus.EXPIRED
                    order.remaining = 0.0
                    continue
            
            # Process order based on type
            if order.order_type == 'market':
                fill = self._process_market_order(order, candle, current_time)
                if fill:
                    fills.append(fill)
            elif order.order_type in ('limit', 'post_only_limit'):
                fill = self._process_limit_order(order, candle, current_time)
                if fill:
                    fills.append(fill)
        
        return fills
    
    def _update_order_book(self, candle: OHLCV) -> None:
        """Update order book for the symbol from candle data."""
        if candle.symbol not in self._order_books:
            self._order_books[candle.symbol] = SimulatedOrderBook(
                candle.symbol,
                self.config.order_book_config
            )
        
        # Calculate volatility from candle
        volatility = (candle.high - candle.low) / candle.close if candle.close > 0 else 0.02
        
        self._order_books[candle.symbol].update_from_candle(
            candle,
            volatility=volatility,
            timestamp=candle.timestamp
        )
    
    def _is_tif_expired(self, order: Order, current_time: datetime) -> bool:
        """Check if order's time-in-force has expired."""
        if order.time_in_force == 'GTC':
            return False
        
        if order.time_in_force == 'IOC':
            # IOC orders expire immediately if not filled
            # They get one chance to fill per process_orders call
            return order.status == OrderStatus.OPEN and order.filled == 0
        
        if order.time_in_force == 'FOK':
            # FOK orders must fill completely or not at all
            # They get one chance to fill
            return order.status == OrderStatus.OPEN
        
        return False
    
    def _get_order_book(self, symbol: str) -> Optional[SimulatedOrderBook]:
        """Get order book for a symbol."""
        return self._order_books.get(symbol)
    
    def _process_market_order(
        self,
        order: Order,
        candle: OHLCV,
        current_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Process a market order.
        
        Market orders fill immediately at the current price with slippage.
        Uses order book for realistic fill price calculation.
        
        Args:
            order: Market order to process
            candle: Current candle data
            current_time: Current simulation time
            
        Returns:
            Fill information if filled
        """
        # Get order book for fill price calculation
        order_book = self._get_order_book(order.symbol)
        
        if order_book:
            # Use order book for realistic fill price
            fill_price, filled_amount, is_complete = order_book.calculate_fill_price(
                order.side,
                order.remaining,
                allow_partial=True
            )
            
            if filled_amount <= 0:
                # No liquidity, order fails
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Insufficient liquidity"
                return None
        else:
            # Fallback to candle-based fill price
            fill_price = self._calculate_market_fill_price(order, candle)
            filled_amount = order.remaining
            is_complete = True
        
        # Simulate fill latency
        if self.config.enable_latency:
            event = self.latency_model.schedule_event(
                LatencyType.ORDER_FILL,
                current_time,
                metadata={
                    'order_id': order.id,
                    'fill_price': fill_price,
                    'fill_amount': filled_amount
                }
            )
            # For market orders, we process immediately but track the latency
            self._stats['latency_ms_total'] += event.latency_ms
            self._stats['latency_count'] += 1
        
        # Calculate fee (taker fee for market orders)
        fee = self._calculate_fee(order, fill_price, is_maker=False)
        
        # Update order
        order.filled = filled_amount
        order.remaining = order.amount - filled_amount
        order.average_fill_price = fill_price
        order.status = OrderStatus.FILLED if is_complete else OrderStatus.PARTIALLY_FILLED
        order.fee = fee
        
        # Record fill
        fill = self._create_fill_record(order, candle, fill_price, filled_amount, fee)
        self._fills.append(fill)
        
        # Update account state
        if self.account_state:
            self.account_state.process_fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                amount=filled_amount,
                price=fill_price,
                fee=fee,
                timestamp=current_time
            )
        
        self._stats['filled_orders'] += 1
        self._stats['total_fills'] += 1
        self._stats['total_fees'] += fee
        
        # Trigger callbacks
        self._trigger_fill_callbacks(order)
        
        logger.debug(
            f"Market order filled: {order.id} {order.side} {filled_amount} @ {fill_price:.2f}, "
            f"fee={fee:.2f}"
        )
        
        return fill
    
    def _process_limit_order(
        self,
        order: Order,
        candle: OHLCV,
        current_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Process a limit order with realistic fill simulation.
        Uses order book for fill price calculation and queue position for priority.
        
        Args:
            order: Limit order to process
            candle: Current candle data
            current_time: Current simulation time
            
        Returns:
            Fill information if filled
        """
        # Check if order should fill based on price
        if not self._should_fill_order(order, candle):
            # Order doesn't fill, transition from pending to open
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.OPEN
            return None
        
        # Check for post-only rejection
        if order.post_only and self._should_reject_post_only(order, candle):
            return self._handle_post_only_rejection(order)
        
        # Get order book for fill calculation
        order_book = self._get_order_book(order.symbol)
        
        # Calculate fill probability based on various factors
        fill_probability = self._calculate_fill_probability(order, candle)
        
        # Adjust probability by queue position (earlier = higher probability)
        if self.config.enable_order_queue and order.id in self._order_queue_positions:
            queue_pos = self._order_queue_positions[order.id]
            queue_factor = max(0.5, 1.0 - (queue_pos * self.config.queue_position_factor / 100))
            fill_probability *= queue_factor
        
        # Check if order fills based on probability
        if random.random() > fill_probability:
            # Order doesn't fill this tick
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.OPEN
            return None
        
        # Calculate fill amount (partial or complete)
        fill_amount = self._calculate_fill_amount(order, candle)
        
        if fill_amount <= 0:
            return None
        
        # Calculate fill price using order book if available
        if order_book:
            fill_price, actual_fill_amount, is_complete = order_book.calculate_fill_price(
                order.side,
                fill_amount,
                allow_partial=self.config.enable_partial_fills
            )
            fill_amount = actual_fill_amount
        else:
            # Fallback to impact model
            fill_price = self._calculate_limit_fill_price(order, candle, fill_amount)
            is_complete = fill_amount >= order.remaining
        
        # Simulate fill latency
        if self.config.enable_latency:
            event = self.latency_model.schedule_event(
                LatencyType.ORDER_FILL,
                current_time,
                metadata={
                    'order_id': order.id,
                    'fill_price': fill_price,
                    'fill_amount': fill_amount
                }
            )
            self._stats['latency_ms_total'] += event.latency_ms
            self._stats['latency_count'] += 1
        
        # Calculate fee (maker fee for limit orders that don't cross spread)
        is_maker = self._is_maker_fill(order, candle)
        fee = self._calculate_fee(order, fill_price, is_maker=is_maker)
        
        # Update order
        previous_filled = order.filled
        order.filled += fill_amount
        order.remaining = order.amount - order.filled
        
        # Update average fill price
        if order.average_fill_price == 0:
            order.average_fill_price = fill_price
        else:
            total_value = (order.average_fill_price * previous_filled) + (fill_price * fill_amount)
            order.average_fill_price = total_value / order.filled
        
        order.fee += fee
        
        # Determine new status
        if order.remaining <= 0.0001:  # Small threshold for floating point
            order.status = OrderStatus.FILLED
            order.remaining = 0.0
            self._stats['filled_orders'] += 1
            self._trigger_fill_callbacks(order)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            self._stats['partially_filled_orders'] += 1
            self._trigger_partial_fill_callbacks(order, fill_amount)
        
        # Record fill
        fill = self._create_fill_record(order, candle, fill_price, fill_amount, fee)
        fill['timestamp'] = current_time  # Use current time instead of candle time
        self._fills.append(fill)
        
        # Update account state
        if self.account_state:
            self.account_state.process_fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                amount=fill_amount,
                price=fill_price,
                fee=fee,
                timestamp=current_time
            )
        
        self._stats['total_fills'] += 1
        self._stats['total_fees'] += fee
        
        logger.debug(
            f"Limit order fill: {order.id} {order.side} {fill_amount}/{order.amount} "
            f"@ {fill_price:.2f}, status={order.status.value}"
        )
        
        return fill
    
    def _should_fill_order(self, order: Order, candle: OHLCV) -> bool:
        """
        Check if an order should fill based on candle data.
        
        Args:
            order: Order to check
            candle: OHLCV candle
            
        Returns:
            True if order should fill
        """
        if order.order_type == 'market':
            return True
        
        if order.price is None:
            return False
        
        # Buy limit order fills if price goes below order price
        if order.side == 'buy':
            return candle.low <= order.price
        
        # Sell limit order fills if price goes above order price
        else:  # side == 'sell'
            return candle.high >= order.price
    
    def _should_reject_post_only(self, order: Order, candle: OHLCV) -> bool:
        """
        Check if a post-only order should be rejected.
        
        Post-only orders are rejected if they would match immediately.
        
        Args:
            order: Order to check
            candle: OHLCV candle
            
        Returns:
            True if order should be rejected
        """
        if not order.post_only:
            return False
        
        if order.price is None:
            return False
        
        # Simulate bid/ask spread
        spread = candle.close * 0.0002  # 2 bps spread
        best_bid = candle.close - spread / 2
        best_ask = candle.close + spread / 2
        
        # Buy order would match if price >= best_ask
        if order.side == 'buy' and order.price >= best_ask:
            return True
        
        # Sell order would match if price <= best_bid
        if order.side == 'sell' and order.price <= best_bid:
            return True
        
        # Also apply probability-based rejection for aggressive post-only orders
        distance_from_mid = abs(order.price - candle.mid_price) / candle.close
        if distance_from_mid < 0.0005:  # Within 5 bps of mid
            rejection_prob = self.config.post_only_reject_probability
            if random.random() < rejection_prob:
                return True
        
        return False
    
    def _handle_post_only_rejection(self, order: Order) -> Optional[Dict[str, Any]]:
        """
        Handle post-only order rejection.
        
        Args:
            order: Rejected order
            
        Returns:
            None (rejected orders don't fill)
        """
        order.status = OrderStatus.REJECTED
        order.reject_reason = "Post-only order would match immediately"
        self._stats['rejected_orders'] += 1
        
        logger.debug(f"Post-only order rejected: {order.id}")
        return None
    
    def _calculate_fill_probability(self, order: Order, candle: OHLCV) -> float:
        """
        Calculate probability of order fill based on multiple factors.
        
        Args:
            order: Order to check
            candle: OHLCV candle
            
        Returns:
            Fill probability (0-1)
        """
        base_prob = self.config.base_fill_probability
        
        if order.price is None:
            return base_prob
        
        # Factor 1: Distance from mid price
        mid_price = candle.mid_price
        distance_pct = abs(order.price - mid_price) / mid_price
        
        # Closer to mid = higher fill probability
        if order.side == 'buy':
            # Buy orders below mid price
            if order.price < mid_price:
                distance_factor = 1.0 - (distance_pct * self.config.distance_penalty_factor * 100)
            else:
                # Above mid = lower probability
                distance_factor = 0.7 - (distance_pct * self.config.distance_penalty_factor * 200)
        else:  # sell
            # Sell orders above mid price
            if order.price > mid_price:
                distance_factor = 1.0 - (distance_pct * self.config.distance_penalty_factor * 100)
            else:
                # Below mid = lower probability
                distance_factor = 0.7 - (distance_pct * self.config.distance_penalty_factor * 200)
        
        # Factor 2: Order size relative to volume
        if candle.volume > 0:
            size_ratio = order.amount / candle.volume
            size_factor = 1.0 - (size_ratio * self.config.size_penalty_factor * 10)
        else:
            size_factor = 1.0
        
        # Factor 3: Time in force
        tif_factor = 1.0
        if order.time_in_force == 'IOC':
            tif_factor = 1.2  # IOC orders are more aggressive
        elif order.time_in_force == 'FOK':
            tif_factor = 0.8  # FOK orders are harder to fill
        
        # Combine factors
        probability = base_prob * distance_factor * size_factor * tif_factor
        
        # Clamp to valid range
        return max(0.05, min(0.99, probability))
    
    def _calculate_fill_amount(self, order: Order, candle: OHLCV) -> float:
        """
        Calculate how much of the order should fill.
        
        Args:
            order: Order to fill
            candle: OHLCV candle
            
        Returns:
            Fill amount
        """
        remaining = order.remaining
        
        if not self.config.enable_partial_fills:
            return remaining
        
        # Calculate available liquidity based on volume
        if candle.volume > 0:
            # Estimate available liquidity (portion of volume)
            available_liquidity = candle.volume * 0.3  # 30% of volume
            
            # Partial fill based on available liquidity
            max_fill_pct = random.uniform(
                self.config.partial_fill_min_pct,
                self.config.partial_fill_max_pct
            )
            
            max_fill = min(remaining, available_liquidity * max_fill_pct)
            
            # For IOC orders, either fill completely or cancel
            if order.time_in_force == 'IOC':
                if max_fill < remaining * 0.95:
                    return max_fill
                return remaining
            
            # For FOK orders, fill completely or not at all
            if order.time_in_force == 'FOK':
                if max_fill < remaining * 0.99:
                    return 0.0
                return remaining
            
            return max_fill
        
        return remaining
    
    def _calculate_market_fill_price(self, order: Order, candle: OHLCV) -> float:
        """
        Calculate fill price for market order with slippage.
        
        Args:
            order: Market order
            candle: OHLCV candle
            
        Returns:
            Fill price with slippage
        """
        # Base price is close price
        base_price = candle.close
        
        # Apply slippage
        slippage_pct = self.config.market_order_slippage_bps / 10000
        
        if order.side == 'buy':
            # Buy orders get worse price (higher)
            fill_price = base_price * (1 + slippage_pct)
            # Ensure within candle range
            fill_price = min(fill_price, candle.high)
        else:
            # Sell orders get worse price (lower)
            fill_price = base_price * (1 - slippage_pct)
            # Ensure within candle range
            fill_price = max(fill_price, candle.low)
        
        return fill_price
    
    def _calculate_limit_fill_price(self, order: Order, candle: OHLCV, fill_amount: float) -> float:
        """
        Calculate fill price for limit order with market impact.
        
        Args:
            order: Limit order
            candle: OHLCV candle
            fill_amount: Amount being filled
            
        Returns:
            Fill price
        """
        if order.price is None:
            return candle.close
        
        # Base fill price is the order price
        fill_price = order.price
        
        # Apply market impact for large orders
        if self.config.enable_market_impact and candle.volume > 0:
            impact = self.impact_model.calculate_impact(
                order_size=fill_amount,
                avg_volume=candle.volume,
                current_price=candle.close
            )
            
            # Apply temporary impact to fill price
            impact_pct = impact.temporary_impact / 10000
            
            if order.side == 'buy':
                fill_price *= (1 + impact_pct)
            else:
                fill_price *= (1 - impact_pct)
        
        # Ensure fill price is within candle range
        fill_price = max(candle.low, min(candle.high, fill_price))
        
        return fill_price
    
    def _is_maker_fill(self, order: Order, candle: OHLCV) -> bool:
        """
        Check if a limit order fill is a maker fill.
        
        Args:
            order: Order being filled
            candle: OHLCV candle
            
        Returns:
            True if maker fill
        """
        if order.price is None:
            return False
        
        # Simulate bid/ask spread
        spread = candle.close * 0.0002
        best_bid = candle.close - spread / 2
        best_ask = candle.close + spread / 2
        
        # Buy order is maker if price < best_ask
        if order.side == 'buy':
            return order.price < best_ask
        
        # Sell order is maker if price > best_bid
        return order.price > best_bid
    
    def _calculate_fee(self, order: Order, fill_price: float, is_maker: bool = True) -> float:
        """
        Calculate trading fee.
        
        Args:
            order: Order being filled
            fill_price: Fill price
            is_maker: Whether this is a maker fill
            
        Returns:
            Fee amount
        """
        trade_value = order.amount * fill_price
        
        if is_maker:
            fee_pct = 0.0002  # 0.02% maker fee
        else:
            fee_pct = 0.0005  # 0.05% taker fee
        
        return trade_value * fee_pct
    
    def _create_fill_record(
        self,
        order: Order,
        candle: OHLCV,
        fill_price: float,
        fill_amount: float,
        fee: float
    ) -> Dict[str, Any]:
        """
        Create a fill record.
        
        Args:
            order: Order being filled
            candle: OHLCV candle
            fill_price: Fill price
            fill_amount: Fill amount
            fee: Fee amount
            
        Returns:
            Fill record dictionary
        """
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'amount': fill_amount,
            'price': fill_price,
            'fee': fee,
            'timestamp': candle.timestamp,
            'candle': candle,
            'is_maker': self._is_maker_fill(order, candle),
            'fill_percentage': order.fill_percentage
        }
    
    def retry_order(self, order_id: str, new_price: Optional[float] = None) -> Optional[Order]:
        """
        Retry a rejected order with optional price adjustment.
        
        Args:
            order_id: Order ID to retry
            new_price: Optional new price for the order
            
        Returns:
            Updated order or None
        """
        order = self._orders.get(order_id)
        if not order:
            return None
        
        if order.status != OrderStatus.REJECTED:
            return None
        
        if order.retry_count >= self.config.max_retries:
            logger.warning(f"Max retries exceeded for order {order_id}")
            return None
        
        # Update order for retry
        order.retry_count += 1
        order.last_retry_at = datetime.now()
        order.status = OrderStatus.PENDING
        order.reject_reason = None
        
        if new_price is not None:
            order.price = new_price
        
        self._stats['retried_orders'] += 1
        
        logger.debug(f"Order retried: {order_id}, attempt {order.retry_count}")
        return order
    
    def adjust_price_for_retry(self, order: Order, attempt: int) -> float:
        """
        Adjust price for post-only retry.
        
        Args:
            order: Order to adjust
            attempt: Retry attempt number
            
        Returns:
            Adjusted price
        """
        if order.price is None:
            return 0.0
        
        if not self.config.enable_price_adjustment:
            return order.price
        
        adjustment = self.config.price_adjustment_step * (attempt + 1)
        adjustment = min(adjustment, self.config.max_price_adjustment)
        
        adjustment_multiplier = adjustment / 100.0
        
        if order.side == 'buy':
            # Move buy price lower
            new_price = order.price * (1 - adjustment_multiplier)
        else:
            # Move sell price higher
            new_price = order.price * (1 + adjustment_multiplier)
        
        return new_price
    
    def on_fill(self, callback: Callable[[Order], None]) -> None:
        """
        Register a callback for order fill events.
        
        Args:
            callback: Function to call when an order is filled
        """
        self._on_fill_callbacks.append(callback)
    
    def on_partial_fill(self, callback: Callable[[Order, float], None]) -> None:
        """
        Register a callback for partial fill events.
        
        Args:
            callback: Function to call when an order is partially filled
        """
        self._on_partial_fill_callbacks.append(callback)
    
    def _trigger_fill_callbacks(self, order: Order) -> None:
        """Trigger fill callbacks."""
        for callback in self._on_fill_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
    
    def _trigger_partial_fill_callbacks(self, order: Order, amount: float) -> None:
        """Trigger partial fill callbacks."""
        for callback in self._on_partial_fill_callbacks:
            try:
                callback(order, amount)
            except Exception as e:
                logger.error(f"Error in partial fill callback: {e}")
    
    def get_fills(self) -> List[Dict[str, Any]]:
        """
        Get all fills that have occurred.
        
        Returns:
            List of fills
        """
        return list(self._fills)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get simulator statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()
        
        # Add current order counts
        stats['open_orders'] = sum(1 for o in self._orders.values() if o.is_active)
        stats['pending_orders'] = sum(1 for o in self._orders.values() if o.status == OrderStatus.PENDING)
        stats['partially_filled'] = sum(1 for o in self._orders.values() if o.status == OrderStatus.PARTIALLY_FILLED)
        
        return stats
    
    def get_order_book_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current order book snapshot for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Order book depth summary or None
        """
        order_book = self._get_order_book(symbol)
        if order_book:
            return order_book.get_depth_summary()
        return None
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics."""
        return self.latency_model.get_stats()
    
    def reset(self) -> None:
        """Reset the order simulator."""
        self._orders.clear()
        self._fills.clear()
        self._order_counter = 0
        self._queue_counter = 0
        
        self._order_books.clear()
        self._pending_submissions.clear()
        self._pending_cancellations.clear()
        self._pending_fills.clear()
        self._order_queue_positions.clear()
        
        self.latency_model.reset()
        
        self._stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'partially_filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'retried_orders': 0,
            'total_fills': 0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'latency_ms_total': 0.0,
            'latency_count': 0,
        }
        
        self._current_time = None
        
        logger.info("BacktestOrderSimulator reset")
