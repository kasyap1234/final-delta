"""
Order simulator module for backtesting.

This module provides order simulation functionality for the backtesting system,
including limit order fills, slippage, and fees.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

from src.backtest.config import BacktestConfig, SlippageModel
from src.backtest.account_state import AccountState

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order status types."""
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'limit', 'market', etc.
    amount: float
    price: float
    status: OrderStatus = OrderStatus.OPEN
    filled: float = 0.0
    remaining: float = 0.0
    average_fill_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    fee: float = 0.0
    fee_currency: str = "USD"
    
    def __post_init__(self):
        """Initialize remaining amount."""
        self.remaining = self.amount - self.filled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'type': self.type,
            'amount': self.amount,
            'price': self.price,
            'status': self.status.value,
            'filled': self.filled,
            'remaining': self.remaining,
            'average_fill_price': self.average_fill_price,
            'timestamp': self.timestamp.isoformat(),
            'fee': self.fee,
            'fee_currency': self.fee_currency
        }


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


class BacktestOrderSimulator:
    """
    Simulate order execution during backtesting.
    
    This class handles:
    - Limit order fills based on candle high/low
    - Slippage application
    - Fee calculation
    - Order cancellation
    - Partial fills
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        account_state: AccountState
    ):
        """
        Initialize the order simulator.
        
        Args:
            config: Backtest configuration
            account_state: Account state tracker
        """
        self.config = config
        self.account_state = account_state
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        
        # Fill tracking
        self._fills: List[Dict[str, Any]] = []
        
        logger.info("BacktestOrderSimulator initialized")
    
    def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the simulator.
        
        Args:
            order: Order to submit
            
        Returns:
            Submitted order
        """
        self._orders[order.id] = order
        logger.debug(f"Order submitted: {order.id} {order.side} {order.amount} @ {order.price}")
        return order
    
    def cancel_order(self, order_id: str) -> Optional[Order]:
        """
        Cancel an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Cancelled order if found, None otherwise
        """
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return None
        
        order.status = OrderStatus.CANCELLED
        logger.debug(f"Order cancelled: {order_id}")
        return order
    
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
        orders = [o for o in self._orders.values() if o.status == OrderStatus.OPEN]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def process_orders(self, candle: OHLCV) -> List[Dict[str, Any]]:
        """
        Process all open orders against a candle.
        
        Args:
            candle: OHLCV candle to process orders against
            
        Returns:
            List of fills that occurred
        """
        fills = []
        
        for order in list(self._orders.values()):
            if order.status != OrderStatus.OPEN:
                continue
            
            if order.symbol != candle.symbol:
                continue
            
            # Check if order should fill
            if self._should_fill_order(order, candle):
                fill = self._fill_order(order, candle)
                if fill:
                    fills.append(fill)
        
        return fills
    
    def get_fills(self) -> List[Dict[str, Any]]:
        """
        Get all fills that have occurred.
        
        Returns:
            List of fills
        """
        return list(self._fills)
    
    def _should_fill_order(self, order: Order, candle: OHLCV) -> bool:
        """
        Check if an order should fill based on candle data.
        
        Args:
            order: Order to check
            candle: OHLCV candle
            
        Returns:
            True if order should fill
        """
        if order.type != 'limit':
            return False
        
        # Buy limit order fills if price goes below order price
        if order.side == 'buy':
            return candle.low <= order.price
        
        # Sell limit order fills if price goes above order price
        else:  # side == 'sell'
            return candle.high >= order.price
    
    def _fill_order(self, order: Order, candle: OHLCV) -> Optional[Dict[str, Any]]:
        """
        Fill an order.
        
        Args:
            order: Order to fill
            candle: OHLCV candle
            
        Returns:
            Fill information if filled, None otherwise
        """
        # Determine fill price with slippage
        fill_price = self._calculate_fill_price(order, candle)
        
        # Calculate fee
        fee = self._calculate_fee(order, fill_price)
        
        # Update order
        order.filled = order.amount
        order.remaining = 0.0
        order.average_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.fee = fee
        
        # Create fill record
        fill = {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'amount': order.amount,
            'price': fill_price,
            'fee': fee,
            'timestamp': candle.timestamp,
            'candle': candle
        }
        
        self._fills.append(fill)
        
        # Update account state
        self.account_state.process_fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            amount=order.amount,
            price=fill_price,
            fee=fee,
            timestamp=candle.timestamp
        )
        
        logger.debug(
            f"Order filled: {order.id} {order.side} {order.amount} @ {fill_price:.2f}, "
            f"fee={fee:.2f}"
        )
        
        return fill
    
    def _calculate_fill_price(self, order: Order, candle: OHLCV) -> float:
        """
        Calculate fill price with slippage.
        
        Args:
            order: Order being filled
            candle: OHLCV candle
            
        Returns:
            Fill price with slippage applied
        """
        # Base fill price is the order price
        fill_price = order.price
        
        # Apply slippage based on model
        if self.config.slippage_model == SlippageModel.PERCENTAGE:
            slippage = fill_price * (self.config.slippage_pct / 100)
            
            # Buy orders get worse price (higher)
            if order.side == 'buy':
                fill_price += slippage
            # Sell orders get worse price (lower)
            else:
                fill_price -= slippage
        
        elif self.config.slippage_model == SlippageModel.FIXED:
            # Fixed slippage in price units
            if order.side == 'buy':
                fill_price += self.config.slippage_pct
            else:
                fill_price -= self.config.slippage_pct
        
        # Ensure fill price is within candle range
        fill_price = max(candle.low, min(candle.high, fill_price))
        
        return fill_price
    
    def _calculate_fee(self, order: Order, fill_price: float) -> float:
        """
        Calculate trading fee.
        
        Args:
            order: Order being filled
            fill_price: Fill price
            
        Returns:
            Fee amount
        """
        # Calculate trade value
        trade_value = order.amount * fill_price
        
        # Use maker fee for limit orders (post-only)
        fee_pct = self.config.maker_fee_pct / 100
        
        return trade_value * fee_pct
    
    def _generate_order_id(self) -> str:
        """
        Generate a unique order ID.
        
        Returns:
            Unique order ID
        """
        self._order_counter += 1
        timestamp = int(datetime.now().timestamp())
        return f"order_{self._order_counter}_{timestamp}"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get simulator statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_orders = len(self._orders)
        filled_orders = sum(1 for o in self._orders.values() if o.status == OrderStatus.FILLED)
        cancelled_orders = sum(1 for o in self._orders.values() if o.status == OrderStatus.CANCELLED)
        open_orders = sum(1 for o in self._orders.values() if o.status == OrderStatus.OPEN)
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'open_orders': open_orders,
            'total_fills': len(self._fills),
            'total_fees': sum(f['fee'] for f in self._fills)
        }
    
    def reset(self) -> None:
        """Reset the order simulator."""
        self._orders.clear()
        self._fills.clear()
        self._order_counter = 0
        logger.info("BacktestOrderSimulator reset")
