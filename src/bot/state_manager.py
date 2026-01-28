"""
State Manager module for the cryptocurrency trading bot.

This module provides the StateManager class for persisting and recovering
trading bot state, including open positions, active orders, and configuration.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from ..database import DatabaseManager, Position, PositionStatus, PositionSide, Order, OrderStatus
from ..database.models import generate_id

logger = logging.getLogger(__name__)


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
    """
    Manages bot state persistence and recovery.
    
    This class handles:
    - Saving bot state to database
    - Loading bot state from database
    - Recovering open positions after restart
    - State versioning and migration
    
    Attributes:
        db_manager: DatabaseManager for persistence
        state: Current BotState instance
        state_file: Optional file path for backup state storage
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        state_file: Optional[str] = None,
        bot_id: Optional[str] = None
    ):
        """
        Initialize the StateManager.
        
        Args:
            db_manager: DatabaseManager instance for database operations
            state_file: Optional file path for backup state storage
            bot_id: Unique identifier for this bot instance
        """
        self.db_manager = db_manager
        self.state_file = Path(state_file) if state_file else None
        self.state = BotState(
            bot_id=bot_id or generate_id("bot"),
            session_start=datetime.now().isoformat()
        )
        
        logger.info(f"StateManager initialized for bot {self.state.bot_id}")
    
    def save_state(self) -> bool:
        """
        Save current bot state to database and optional backup file.
        
        Returns:
            True if state was saved successfully, False otherwise
        """
        try:
            # Update timestamp
            self.state.last_saved = datetime.now().isoformat()
            
            # Save to database
            self._save_to_database()
            
            # Save to backup file if configured
            if self.state_file:
                self._save_to_file()
            
            logger.debug(f"State saved at {self.state.last_saved}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def _save_to_database(self) -> None:
        """Save state components to database."""
        # Save positions
        for position_data in self.state.open_positions:
            position = Position(
                id=position_data.get('id', generate_id("pos")),
                symbol=position_data['symbol'],
                side=PositionSide(position_data['side']),
                status=PositionStatus.OPEN,
                entry_price=position_data['entry_price'],
                current_price=position_data.get('current_price', position_data['entry_price']),
                size=position_data['size'],
                unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
                realized_pnl=0.0,
                stop_loss_price=position_data.get('stop_loss'),
                take_profit_price=position_data.get('take_profit'),
                entry_time=datetime.fromisoformat(position_data['entry_time']) 
                    if 'entry_time' in position_data else datetime.now(),
                metadata=json.dumps(position_data.get('metadata', {}))
            )
            self.db_manager.save_position(position)
        
        # Save orders
        for order_data in self.state.active_orders:
            order = Order(
                id=order_data.get('id', generate_id("ord")),
                symbol=order_data['symbol'],
                side=order_data['side'],
                order_type=order_data['order_type'],
                status=OrderStatus(order_data.get('status', 'open')),
                price=order_data.get('price'),
                amount=order_data['amount'],
                filled=order_data.get('filled', 0.0),
                remaining=order_data.get('remaining', order_data['amount']),
                created_at=datetime.fromisoformat(order_data['created_at']) 
                    if 'created_at' in order_data else datetime.now(),
                metadata=json.dumps(order_data.get('metadata', {}))
            )
            self.db_manager.save_order(order)
        
        # Save state metadata as system log
        self.db_manager.save_system_log(
            level="INFO",
            component="StateManager",
            message=f"State saved: {len(self.state.open_positions)} positions, "
                    f"{len(self.state.active_orders)} orders",
            metadata=json.dumps({
                'bot_id': self.state.bot_id,
                'version': self.state.version,
                'daily_pnl': self.state.daily_pnl,
                'daily_trades': self.state.daily_trades
            })
        )
    
    def _save_to_file(self) -> None:
        """Save state to backup file."""
        if not self.state_file:
            return
        
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert state to dict and save
        state_dict = asdict(self.state)
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    def load_state(self) -> Optional[BotState]:
        """
        Load bot state from database.
        
        Returns:
            BotState if state was loaded successfully, None otherwise
        """
        try:
            # Try to load from database first
            loaded_state = self._load_from_database()
            
            if loaded_state:
                self.state = loaded_state
                logger.info(f"State loaded from database: "
                           f"{len(self.state.open_positions)} positions, "
                           f"{len(self.state.active_orders)} orders")
                return self.state
            
            # Fall back to file if database is empty
            if self.state_file and self.state_file.exists():
                loaded_state = self._load_from_file()
                if loaded_state:
                    self.state = loaded_state
                    logger.info("State loaded from backup file")
                    return self.state
            
            logger.info("No previous state found, starting fresh")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def _load_from_database(self) -> Optional[BotState]:
        """Load state from database."""
        try:
            # Load open positions
            db_positions = self.db_manager.get_positions(status=PositionStatus.OPEN)
            open_positions = []
            for pos in db_positions:
                open_positions.append({
                    'id': pos.id,
                    'symbol': pos.symbol,
                    'side': pos.side.value,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'size': pos.size,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'stop_loss': pos.stop_loss_price,
                    'take_profit': pos.take_profit_price,
                    'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                    'metadata': json.loads(pos.metadata) if pos.metadata else {}
                })
            
            # Load active orders
            db_orders = self.db_manager.get_orders(status=OrderStatus.OPEN)
            active_orders = []
            for order in db_orders:
                active_orders.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side.value if hasattr(order.side, 'value') else order.side,
                    'order_type': order.order_type.value if hasattr(order.order_type, 'value') else order.order_type,
                    'status': order.status.value if hasattr(order.status, 'value') else order.status,
                    'price': order.price,
                    'amount': order.amount,
                    'filled': order.filled,
                    'remaining': order.remaining,
                    'created_at': order.created_at.isoformat() if order.created_at else None,
                    'metadata': json.loads(order.metadata) if order.metadata else {}
                })
            
            # Create state if we have any data
            if open_positions or active_orders:
                return BotState(
                    bot_id=self.state.bot_id,
                    open_positions=open_positions,
                    active_orders=active_orders,
                    last_saved=datetime.now().isoformat()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return None
    
    def _load_from_file(self) -> Optional[BotState]:
        """Load state from backup file."""
        try:
            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)
            
            return BotState(**state_dict)
            
        except Exception as e:
            logger.error(f"Error loading from file: {e}")
            return None
    
    def recover_positions(self) -> List[Dict[str, Any]]:
        """
        Recover open positions from saved state.
        
        This method should be called after initialization to restore
        any positions that were open when the bot last stopped.
        
        Returns:
            List of recovered position dictionaries
        """
        try:
            # Load state if not already loaded
            if not self.state.open_positions:
                self.load_state()
            
            recovered = []
            for position_data in self.state.open_positions:
                # Validate position data
                required_fields = ['id', 'symbol', 'side', 'entry_price', 'size']
                if all(field in position_data for field in required_fields):
                    recovered.append(position_data)
                    logger.info(f"Recovered position: {position_data['symbol']} "
                               f"({position_data['side']}) @ {position_data['entry_price']}")
                else:
                    logger.warning(f"Skipping invalid position data: {position_data}")
            
            if recovered:
                logger.info(f"Recovered {len(recovered)} positions from previous session")
            
            return recovered
            
        except Exception as e:
            logger.error(f"Error recovering positions: {e}")
            return []
    
    def add_position(self, position_data: Dict[str, Any]) -> None:
        """
        Add a position to the current state.
        
        Args:
            position_data: Dictionary containing position details
        """
        # Ensure position has an ID
        if 'id' not in position_data:
            position_data['id'] = generate_id("pos")
        
        # Add entry time if not present
        if 'entry_time' not in position_data:
            position_data['entry_time'] = datetime.now().isoformat()
        
        self.state.open_positions.append(position_data)
        logger.debug(f"Added position {position_data['id']} to state")
    
    def remove_position(self, position_id: str) -> bool:
        """
        Remove a position from the current state.
        
        Args:
            position_id: ID of the position to remove
            
        Returns:
            True if position was removed, False if not found
        """
        initial_count = len(self.state.open_positions)
        self.state.open_positions = [
            p for p in self.state.open_positions 
            if p.get('id') != position_id
        ]
        
        removed = len(self.state.open_positions) < initial_count
        if removed:
            logger.debug(f"Removed position {position_id} from state")
        
        return removed
    
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a position in the current state.
        
        Args:
            position_id: ID of the position to update
            updates: Dictionary of fields to update
            
        Returns:
            True if position was updated, False if not found
        """
        for position in self.state.open_positions:
            if position.get('id') == position_id:
                position.update(updates)
                logger.debug(f"Updated position {position_id}: {updates}")
                return True
        
        return False
    
    def add_order(self, order_data: Dict[str, Any]) -> None:
        """
        Add an order to the current state.
        
        Args:
            order_data: Dictionary containing order details
        """
        if 'id' not in order_data:
            order_data['id'] = generate_id("ord")
        
        if 'created_at' not in order_data:
            order_data['created_at'] = datetime.now().isoformat()
        
        self.state.active_orders.append(order_data)
        logger.debug(f"Added order {order_data['id']} to state")
    
    def remove_order(self, order_id: str) -> bool:
        """
        Remove an order from the current state.
        
        Args:
            order_id: ID of the order to remove
            
        Returns:
            True if order was removed, False if not found
        """
        initial_count = len(self.state.active_orders)
        self.state.active_orders = [
            o for o in self.state.active_orders 
            if o.get('id') != order_id
        ]
        
        removed = len(self.state.active_orders) < initial_count
        if removed:
            logger.debug(f"Removed order {order_id} from state")
        
        return removed
    
    def update_daily_metrics(self, pnl: float, trade_count: int = 1) -> None:
        """
        Update daily trading metrics.
        
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
        
        logger.debug(f"Updated daily metrics: PnL={self.state.daily_pnl}, "
                    f"Trades={self.state.daily_trades}")
    
    def update_last_price(self, symbol: str, price: float) -> None:
        """
        Update last known price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        self.state.last_prices[symbol] = price
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Get last known price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Last known price or None if not available
        """
        return self.state.last_prices.get(symbol)
    
    def clear_state(self) -> None:
        """Clear all state data."""
        self.state = BotState(
            bot_id=self.state.bot_id,
            session_start=datetime.now().isoformat()
        )
        logger.info("State cleared")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            'bot_id': self.state.bot_id,
            'version': self.state.version,
            'last_saved': self.state.last_saved,
            'session_start': self.state.session_start,
            'open_positions_count': len(self.state.open_positions),
            'active_orders_count': len(self.state.active_orders),
            'daily_pnl': self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'total_trades': self.state.total_trades,
            'win_rate': (self.state.winning_trades / self.state.total_trades * 100)
                if self.state.total_trades > 0 else 0.0
        }
