"""
Account state module for backtesting.

This module provides account state tracking for the backtesting system,
including balance, positions, and P&L tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
import logging

from src.risk.portfolio_tracker import Position, PositionStatus

logger = logging.getLogger(__name__)


@dataclass
class Balance:
    """Account balance information."""
    total: float
    free: float
    used: float
    currency: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total': self.total,
            'free': self.free,
            'used': self.used,
            'currency': self.currency
        }


class AccountState:
    """
    Track simulated account balance, positions, and P&L during backtesting.
    
    This class manages:
    - Account balance (total, free, used)
    - Open positions
    - Realized and unrealized P&L
    - Fee tracking
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        currency: str = "USD"
    ):
        """
        Initialize account state.
        
        Args:
            initial_balance: Starting account balance
            currency: Account currency
        """
        self._initial_balance = initial_balance
        self._currency = currency
        self._total_balance = initial_balance
        self._free_balance = initial_balance
        self._used_balance = 0.0
        
        # Position tracking
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._position_counter = 0
        
        # P&L tracking
        self._total_realized_pnl = 0.0
        self._total_fees_paid = 0.0
        
        logger.info(
            f"AccountState initialized: balance={initial_balance} {currency}"
        )
    
    def get_balance(self) -> Balance:
        """
        Get current account balance.
        
        Returns:
            Balance object with total, free, and used amounts
        """
        return Balance(
            total=self._total_balance,
            free=self._free_balance,
            used=self._used_balance,
            currency=self._currency
        )
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of positions as dictionaries
        """
        positions = []
        for pos in self._positions.values():
            if symbol is None or pos.symbol == symbol:
                positions.append(pos.to_dict())
        return positions
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get a specific position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position if found, None otherwise
        """
        return self._positions.get(position_id)
    
    def get_all_positions(self) -> List[Position]:
        """
        Get all open positions as Position objects.
        
        Returns:
            List of Position objects
        """
        return list(self._positions.values())
    
    def get_closed_positions(self) -> List[Position]:
        """
        Get all closed positions.
        
        Returns:
            List of closed Position objects
        """
        return list(self._closed_positions)
    
    def process_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        fee: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[Position]:
        """
        Process an order fill and update account state.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Amount filled
            price: Fill price
            fee: Fee paid
            timestamp: Fill timestamp (current time if None)
            
        Returns:
            Position if a position was opened/closed, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Deduct fee from balance
        self._total_balance -= fee
        self._free_balance -= fee
        self._total_fees_paid += fee
        
        # Check if this is opening or closing a position
        if side == 'buy':
            # Opening a long position
            position = self._open_position(
                symbol=symbol,
                side='long',
                size=amount,
                entry_price=price,
                timestamp=timestamp,
                fees=fee
            )
            return position
        else:  # side == 'sell'
            # Check if we have a long position to close
            position = self._find_position_to_close(symbol)
            if position:
                return self._close_position(
                    position_id=position.position_id,
                    exit_price=price,
                    timestamp=timestamp,
                    exit_fees=fee
                )
            else:
                # Opening a short position
                return self._open_position(
                    symbol=symbol,
                    side='short',
                    size=amount,
                    entry_price=price,
                    timestamp=timestamp,
                    fees=fee
                )
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]) -> None:
        """
        Update unrealized P&L for all open positions.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
        """
        for position in self._positions.values():
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                
                # Calculate unrealized P&L
                if position.side == 'long':
                    price_diff = current_price - position.entry_price
                else:  # short
                    price_diff = position.entry_price - current_price
                
                position.unrealized_pnl = price_diff * position.size
                position.unrealized_pnl_percent = (
                    price_diff / position.entry_price * 100
                    if position.entry_price != 0 else 0
                )
                position.current_price = current_price
    
    def get_equity(self) -> float:
        """
        Calculate total account equity (balance + unrealized P&L).
        
        Returns:
            Total equity
        """
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())
        return self._total_balance + unrealized_pnl
    
    def get_total_realized_pnl(self) -> float:
        """
        Get total realized P&L.
        
        Returns:
            Total realized P&L
        """
        return self._total_realized_pnl
    
    def get_total_fees_paid(self) -> float:
        """
        Get total fees paid.
        
        Returns:
            Total fees paid
        """
        return self._total_fees_paid
    
    def get_initial_balance(self) -> float:
        """
        Get initial balance.
        
        Returns:
            Initial balance
        """
        return self._initial_balance
    
    def _open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        timestamp: datetime,
        fees: float
    ) -> Position:
        """
        Open a new position.
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('long' or 'short')
            size: Position size
            entry_price: Entry price
            timestamp: Entry timestamp
            fees: Entry fees
            
        Returns:
            Created Position object
        """
        self._position_counter += 1
        position_id = f"{symbol.replace('/', '_')}_{self._position_counter}_{int(timestamp.timestamp())}"
        
        # Calculate position value
        position_value = size * entry_price
        
        # Update used balance
        self._used_balance += position_value
        self._free_balance -= position_value
        
        # Create position (stop_loss and take_profit will be set by risk manager)
        position = Position(
            position_id=position_id,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=timestamp,
            stop_loss=0.0,  # Will be set by risk manager
            take_profit=0.0,  # Will be set by risk manager
            risk_amount=0.0,  # Will be set by risk manager
            current_price=entry_price,
            fees_paid=fees
        )
        
        self._positions[position_id] = position
        
        logger.debug(
            f"Opened position: {position_id}, {symbol} {side} {size} @ {entry_price}"
        )
        
        return position
    
    def _close_position(
        self,
        position_id: str,
        exit_price: float,
        timestamp: datetime,
        exit_fees: float
    ) -> Optional[Position]:
        """
        Close an open position.
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            timestamp: Exit timestamp
            exit_fees: Exit fees
            
        Returns:
            Closed Position object if found, None otherwise
        """
        if position_id not in self._positions:
            return None
        
        position = self._positions[position_id]
        
        # Calculate realized P&L
        if position.side == 'long':
            price_diff = exit_price - position.entry_price
        else:  # short
            price_diff = position.entry_price - exit_price
        
        realized_pnl = price_diff * position.size
        
        # Update position
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = timestamp
        position.realized_pnl = realized_pnl
        position.fees_paid += exit_fees
        
        # Update balance
        self._total_balance += realized_pnl - exit_fees
        self._total_realized_pnl += realized_pnl
        
        # Update used/free balance
        position_value = position.size * position.entry_price
        self._used_balance -= position_value
        self._free_balance += position_value + realized_pnl - exit_fees
        
        # Move to closed positions
        del self._positions[position_id]
        self._closed_positions.append(position)
        
        logger.debug(
            f"Closed position: {position_id}, P&L={realized_pnl:.2f}, "
            f"fees={exit_fees:.2f}"
        )
        
        return position
    
    def open_position(self, symbol: str, side: str, size: float, entry_price: float) -> bool:
        """
        Open a position (simplified interface for strategy engine).
        
        Args:
            symbol: Trading pair symbol
            side: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            
        Returns:
            True if position opened successfully
        """
        position_value = size * entry_price
        
        # Check if we have enough free balance
        if position_value > self._free_balance:
            logger.warning(f"Insufficient balance to open position: {position_value:.2f} > {self._free_balance:.2f}")
            return False
        
        # Use existing _open_position method
        self._open_position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            timestamp=datetime.now(),
            fees=0.0
        )
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Position]:
        """
        Close a position by symbol (simplified interface for strategy engine).
        
        Args:
            symbol: Trading pair symbol
            exit_price: Exit price
            
        Returns:
            Closed Position if found, None otherwise
        """
        position = self._find_position_to_close(symbol)
        if position:
            return self._close_position(
                position_id=position.position_id,
                exit_price=exit_price,
                timestamp=datetime.now(),
                exit_fees=0.0
            )
        return None
    
    def _find_position_to_close(self, symbol: str) -> Optional[Position]:
        """
        Find a position to close for a given symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position if found, None otherwise
        """
        for position in self._positions.values():
            if position.symbol == symbol:
                return position
        return None
    
    def _generate_position_id(self, symbol: str) -> str:
        """
        Generate a unique position ID.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Unique position ID
        """
        self._position_counter += 1
        timestamp = int(datetime.now().timestamp())
        return f"{symbol.replace('/', '_')}_{self._position_counter}_{timestamp}"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get account statistics.
        
        Returns:
            Dictionary with account statistics
        """
        return {
            'initial_balance': self._initial_balance,
            'total_balance': self._total_balance,
            'free_balance': self._free_balance,
            'used_balance': self._used_balance,
            'equity': self.get_equity(),
            'total_realized_pnl': self._total_realized_pnl,
            'total_fees_paid': self._total_fees_paid,
            'num_open_positions': len(self._positions),
            'num_closed_positions': len(self._closed_positions),
            'currency': self._currency
        }
    
    def __repr__(self) -> str:
        """String representation of AccountState."""
        return (
            f"AccountState(balance={self._total_balance:.2f}, "
            f"equity={self.get_equity():.2f}, "
            f"positions={len(self._positions)}, "
            f"realized_pnl={self._total_realized_pnl:.2f})"
        )
