"""State Manager module for backtesting.

This module provides the StateManager class for managing trading bot state
during backtesting, including open positions, active orders, and portfolio state.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from typing import Union
from enum import Enum

logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class PositionState:
    """Position state data."""
    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    status: PositionStatus
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'status': self.status.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'metadata': self.metadata
        }


@dataclass
class OrderState:
    """Order state data."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    amount: float
    price: Optional[float] = None
    filled: float = 0.0
    remaining: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, OrderSide) else self.side,
            'order_type': self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type,
            'status': self.status.value if isinstance(self.status, OrderStatus) else self.status,
            'amount': self.amount,
            'price': self.price,
            'filled': self.filled,
            'remaining': self.remaining,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }


@dataclass
class BotState:
    """Complete bot state for persistence."""
    
    # Bot metadata
    bot_id: str
    version: str = "1.0.0"
    last_saved: Optional[str] = None
    
    # Trading state
    open_positions: List[Dict[str, Any]] = None
    active_orders: List[Dict[str, Any]] = None
    position_groups: List[Dict[str, Any]] = None
    
    # Market state
    last_prices: Dict[str, float] = None
    
    # Risk state
    daily_pnl: float = 0.0
    daily_trades: int = 0
    
    # Session metrics
    session_start: Optional[str] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def __post_init__(self):
        if self.open_positions is None:
            self.open_positions = []
        if self.active_orders is None:
            self.active_orders = []
        if self.position_groups is None:
            self.position_groups = []
        if self.last_prices is None:
            self.last_prices = {}


class StateManager:
    """Manages bot state during backtesting.
    
    This class handles:
    - Position state transitions
    - Order state management
    - Portfolio state tracking
    - Event-driven state updates
    
    Attributes:
        state: Current BotState instance
        _positions: Dictionary of position states by ID
        _orders: Dictionary of order states by ID
        _symbol_positions: Dictionary mapping symbols to position IDs
        _current_time: Current backtest timestamp
    """
    
    def __init__(
        self,
        bot_id: Optional[str] = None,
        initial_balance: float = 10000.0
    ):
        """Initialize the StateManager.
        
        Args:
            bot_id: Unique identifier for this bot instance
            initial_balance: Initial account balance
        """
        self.state = BotState(
            bot_id=bot_id or f"backtest_bot_{int(datetime.now().timestamp())}",
            session_start=datetime.now().isoformat()
        )
        
        # Position tracking
        self._positions: Dict[str, PositionState] = {}
        self._orders: Dict[str, OrderState] = {}
        self._symbol_positions: Dict[str, str] = {}  # symbol -> position_id
        
        # Account tracking
        self._account_balance = initial_balance
        self._initial_balance = initial_balance
        self._peak_balance = initial_balance
        
        # Current time for backtest
        self._current_time: Optional[datetime] = None
        
        # Position and order counters
        self._position_counter = 0
        self._order_counter = 0
        
        logger.info(f"StateManager initialized for bot {self.state.bot_id}")
    
    def set_current_time(self, current_time: datetime) -> None:
        """Set current time for backtest.
        
        Args:
            current_time: Current backtest timestamp
        """
        self._current_time = current_time
    
    def create_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PositionState:
        """Create a new position.
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('long' or 'short')
            size: Position size
            entry_price: Entry price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            metadata: Optional position metadata
            
        Returns:
            Created PositionState
        """
        self._position_counter += 1
        timestamp = int(self._current_time.timestamp()) if self._current_time else int(datetime.now().timestamp())
        position_id = f"{symbol.replace('/', '_')}_{self._position_counter}_{timestamp}"
        
        position = PositionState(
            position_id=position_id,
            symbol=symbol,
            side=side.lower(),
            status=PositionStatus.OPEN,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=self._current_time if self._current_time else datetime.now(),
            metadata=metadata or {}
        )
        
        self._positions[position_id] = position
        self._symbol_positions[symbol] = position_id
        
        # Update state
        self.state.open_positions.append(position.to_dict())
        
        logger.debug(f"Created position {position_id} for {symbol}")
        return position
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float
    ) -> Optional[PositionState]:
        """Close a position.
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            realized_pnl: Realized P&L
            
        Returns:
            Closed PositionState if found, None otherwise
        """
        if position_id not in self._positions:
            return None
        
        position = self._positions[position_id]
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = self._current_time if self._current_time else datetime.now()
        position.realized_pnl = realized_pnl
        position.current_price = exit_price
        position.unrealized_pnl = 0.0
        
        # Remove from symbol mapping
        if position.symbol in self._symbol_positions:
            del self._symbol_positions[position.symbol]
        
        # Update state
        self.state.open_positions = [
            p for p in self.state.open_positions 
            if p.get('position_id') != position_id
        ]
        
        # Update metrics
        self.state.total_trades += 1
        if realized_pnl > 0:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1
        
        logger.debug(f"Closed position {position_id} with P&L: {realized_pnl:.2f}")
        return position
    
    def update_position_price(
        self,
        position_id: str,
        current_price: float
    ) -> Optional[PositionState]:
        """Update position price and unrealized P&L.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            Updated PositionState if found, None otherwise
        """
        if position_id not in self._positions:
            return None
        
        position = self._positions[position_id]
        position.current_price = current_price
        
        # Calculate unrealized P&L
        if position.side == 'long':
            price_diff = current_price - position.entry_price
        else:  # short
            price_diff = position.entry_price - current_price
        
        position.unrealized_pnl = price_diff * position.size
        
        return position
    
    def update_position(
        self,
        position_id: str,
        updates: Dict[str, Any]
    ) -> Optional[PositionState]:
        """Update position fields.
        
        Args:
            position_id: Position identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated PositionState if found, None otherwise
        """
        if position_id not in self._positions:
            return None
        
        position = self._positions[position_id]
        
        for key, value in updates.items():
            if hasattr(position, key):
                setattr(position, key, value)
        
        return position
    
    def get_position(self, position_id: str) -> Optional[PositionState]:
        """Get position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            PositionState if found, None otherwise
        """
        return self._positions.get(position_id)
    
    def get_position_by_symbol(self, symbol: str) -> Optional[PositionState]:
        """Get position by symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            PositionState if found, None otherwise
        """
        position_id = self._symbol_positions.get(symbol)
        if position_id:
            return self._positions.get(position_id)
        return None
    
    def get_open_positions(self) -> List[PositionState]:
        """Get all open positions.
        
        Returns:
            List of open PositionState objects
        """
        return [
            pos for pos in self._positions.values()
            if pos.status == PositionStatus.OPEN
        ]
    
    def get_open_positions_dict(self) -> List[Dict[str, Any]]:
        """Get all open positions as dictionaries.
        
        Returns:
            List of position dictionaries
        """
        return [pos.to_dict() for pos in self.get_open_positions()]
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if open position exists, False otherwise
        """
        position_id = self._symbol_positions.get(symbol)
        if position_id:
            position = self._positions.get(position_id)
            return position is not None and position.status == PositionStatus.OPEN
        return False
    
    def create_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        amount: float,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OrderState:
        """Create a new order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
            amount: Order amount
            price: Optional order price (for limit orders)
            metadata: Optional order metadata
            
        Returns:
            Created OrderState
        """
        self._order_counter += 1
        timestamp = int(self._current_time.timestamp()) if self._current_time else int(datetime.now().timestamp())
        order_id = f"order_{self._order_counter}_{timestamp}"
        
        # Convert enums if needed
        if isinstance(side, str):
            side = OrderSide(side.lower())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.lower())
        
        order = OrderState(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.PENDING,
            amount=amount,
            price=price,
            filled=0.0,
            remaining=amount,
            created_at=self._current_time if self._current_time else datetime.now(),
            metadata=metadata or {}
        )
        
        self._orders[order_id] = order
        self.state.active_orders.append(order.to_dict())
        
        logger.debug(f"Created order {order_id} for {symbol}")
        return order
    
    def fill_order(
        self,
        order_id: str,
        filled_amount: float,
        fill_price: float
    ) -> Optional[OrderState]:
        """Fill an order (partially or fully).
        
        Args:
            order_id: Order identifier
            filled_amount: Amount filled
            fill_price: Fill price
            
        Returns:
            Updated OrderState if found, None otherwise
        """
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        order.filled += filled_amount
        order.remaining = order.amount - order.filled
        order.updated_at = self._current_time if self._current_time else datetime.now()
        
        if order.remaining <= 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        logger.debug(f"Order {order_id} filled: {filled_amount} @ {fill_price}")
        return order
    
    def cancel_order(self, order_id: str) -> Optional[OrderState]:
        """Cancel an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Cancelled OrderState if found, None otherwise
        """
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.updated_at = self._current_time if self._current_time else datetime.now()
        
        # Update state
        self.state.active_orders = [
            o for o in self.state.active_orders
            if o.get('order_id') != order_id
        ]
        
        logger.debug(f"Order {order_id} cancelled")
        return order
    
    def get_order(self, order_id: str) -> Optional[OrderState]:
        """Get order by ID.
        
        Args:
            order_id: Order identifier
            
        Returns:
            OrderState if found, None otherwise
        """
        return self._orders.get(order_id)
    
    def get_active_orders(self) -> List[OrderState]:
        """Get all active (non-filled, non-cancelled) orders.
        
        Returns:
            List of active OrderState objects
        """
        return [
            order for order in self._orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]
    
    def get_active_orders_dict(self) -> List[Dict[str, Any]]:
        """Get all active orders as dictionaries.
        
        Returns:
            List of order dictionaries
        """
        return [order.to_dict() for order in self.get_active_orders()]
    
    def update_last_price(self, symbol: str, price: float) -> None:
        """Update last known price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        self.state.last_prices[symbol] = price
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Last known price or None if not available
        """
        return self.state.last_prices.get(symbol)
    
    def update_daily_metrics(self, pnl: float, trade_count: int = 1) -> None:
        """Update daily trading metrics.
        
        Args:
            pnl: Profit/loss amount to add
            trade_count: Number of trades to add
        """
        self.state.daily_pnl += pnl
        self.state.daily_trades += trade_count
        self.state.total_trades += trade_count
        
        if pnl > 0:
            self.state.winning_trades += 1
        elif pnl < 0:
            self.state.losing_trades += 1
    
    def set_account_balance(self, balance: float) -> None:
        """Update account balance.
        
        Args:
            balance: New account balance
        """
        self._account_balance = balance
        if balance > self._peak_balance:
            self._peak_balance = balance
    
    def get_account_balance(self) -> float:
        """Get current account balance.
        
        Returns:
            Account balance
        """
        return self._account_balance
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state.
        
        Returns:
            Dictionary with state summary
        """
        open_positions = self.get_open_positions()
        unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions)
        
        return {
            'bot_id': self.state.bot_id,
            'version': self.state.version,
            'last_saved': self.state.last_saved,
            'session_start': self.state.session_start,
            'open_positions_count': len(open_positions),
            'active_orders_count': len(self.get_active_orders()),
            'daily_pnl': self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'total_trades': self.state.total_trades,
            'win_rate': (self.state.winning_trades / self.state.total_trades * 100)
                if self.state.total_trades > 0 else 0.0,
            'account_balance': self._account_balance,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': self._account_balance + unrealized_pnl
        }
    
    def clear_state(self) -> None:
        """Clear all state data."""
        self.state = BotState(
            bot_id=self.state.bot_id,
            session_start=datetime.now().isoformat()
        )
        self._positions.clear()
        self._orders.clear()
        self._symbol_positions.clear()
        self._position_counter = 0
        self._order_counter = 0
        logger.info("State cleared")
    
    def save_state(self) -> Dict[str, Any]:
        """Save current state to dictionary.
        
        Returns:
            State dictionary
        """
        self.state.last_saved = datetime.now().isoformat()
        
        # Update state with current positions and orders
        self.state.open_positions = self.get_open_positions_dict()
        self.state.active_orders = self.get_active_orders_dict()
        
        return asdict(self.state)
    
    def load_state(self, state_dict: Dict[str, Any]) -> bool:
        """Load state from dictionary.
        
        Args:
            state_dict: State dictionary
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.state = BotState(**state_dict)
            
            # Restore positions
            self._positions.clear()
            for pos_dict in self.state.open_positions:
                position = PositionState(
                    position_id=pos_dict['position_id'],
                    symbol=pos_dict['symbol'],
                    side=pos_dict['side'],
                    status=PositionStatus(pos_dict['status']),
                    size=pos_dict['size'],
                    entry_price=pos_dict['entry_price'],
                    current_price=pos_dict['current_price'],
                    unrealized_pnl=pos_dict['unrealized_pnl'],
                    realized_pnl=pos_dict['realized_pnl'],
                    stop_loss=pos_dict.get('stop_loss'),
                    take_profit=pos_dict.get('take_profit'),
                    entry_time=datetime.fromisoformat(pos_dict['entry_time']) if pos_dict.get('entry_time') else datetime.now(),
                    exit_time=datetime.fromisoformat(pos_dict['exit_time']) if pos_dict.get('exit_time') else None,
                    exit_price=pos_dict.get('exit_price'),
                    metadata=pos_dict.get('metadata', {})
                )
                self._positions[position.position_id] = position
                if position.status == PositionStatus.OPEN:
                    self._symbol_positions[position.symbol] = position.position_id
            
            logger.info(f"State loaded: {len(self._positions)} positions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
