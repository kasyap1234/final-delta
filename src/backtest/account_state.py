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
from src.backtest.fees import FeeCalculator, FeeType, OrderType

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
    - Fee tracking with FeeCalculator integration
    - Funding rate deductions/additions
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        currency: str = "USD",
        fee_calculator: Optional[FeeCalculator] = None
    ):
        """
        Initialize account state.
        
        Args:
            initial_balance: Starting account balance
            currency: Account currency
            fee_calculator: FeeCalculator instance for fee calculations
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
        
        # Fee calculator integration
        self._fee_calculator = fee_calculator
        
        # Fee history for reporting
        self._fee_history: List[Dict[str, Any]] = []
        
        # Funding payment tracking
        self._total_funding_paid = 0.0
        self._total_funding_received = 0.0
        self._funding_history: List[Dict[str, Any]] = []
        
        # Last funding timestamp per symbol
        self._last_funding_time: Dict[str, datetime] = {}
        
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
    
    def set_fee_calculator(self, fee_calculator: FeeCalculator) -> None:
        """
        Set the fee calculator for this account.
        
        Args:
            fee_calculator: FeeCalculator instance
        """
        self._fee_calculator = fee_calculator
        logger.info("FeeCalculator set for AccountState")
    
    def get_fee_calculator(self) -> Optional[FeeCalculator]:
        """
        Get the fee calculator instance.
        
        Returns:
            FeeCalculator instance or None
        """
        return self._fee_calculator
    
    def process_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        fee: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        order_type: str = "limit",
        is_maker: bool = True
    ) -> Optional[Position]:
        """
        Process an order fill and update account state.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Amount filled
            price: Fill price
            fee: Fee paid (calculated if None using FeeCalculator)
            timestamp: Fill timestamp (current time if None)
            order_type: Order type ('limit', 'market', 'post_only')
            is_maker: Whether this is a maker order (for fee calculation)
            
        Returns:
            Position if a position was opened/closed, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate fee using FeeCalculator if available and fee not provided
        if fee is None and self._fee_calculator is not None:
            order_type_enum = self._parse_order_type(order_type, is_maker)
            fee_result = self._fee_calculator.calculate_trade_fee(
                symbol=symbol,
                amount=amount,
                price=price,
                order_type=order_type_enum,
                timestamp=timestamp,
                metadata={'order_id': order_id, 'side': side, 'is_maker': is_maker}
            )
            fee = fee_result['fee_paid']
            fee_type = fee_result['fee_type']
        else:
            fee = fee or 0.0
            fee_type = 'maker' if is_maker else 'taker'
        
        # Record fee in history
        self._record_fee({
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'order_id': order_id,
            'side': side,
            'amount': amount,
            'price': price,
            'fee': fee,
            'fee_type': fee_type,
            'notional_value': amount * price
        })
        
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
    
    def _parse_order_type(self, order_type: str, is_maker: bool) -> OrderType:
        """Parse order type string to OrderType enum."""
        order_type = order_type.lower()
        if order_type == 'market':
            return OrderType.MARKET
        elif order_type == 'post_only':
            return OrderType.POST_ONLY
        else:
            return OrderType.LIMIT
    
    def _record_fee(self, fee_record: Dict[str, Any]) -> None:
        """Record a fee transaction in history."""
        self._fee_history.append(fee_record)
    
    def process_funding(
        self,
        symbol: str,
        position_size: float,
        mark_price: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Process funding rate payment for a position.
        
        Args:
            symbol: Trading pair symbol
            position_size: Position size (positive for long, negative for short)
            mark_price: Current mark price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with funding payment details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check if we need to process funding (avoid duplicates)
        last_funding = self._last_funding_time.get(symbol)
        if last_funding is not None:
            time_diff = timestamp - last_funding
            if time_diff.total_seconds() < 3600:  # Minimum 1 hour between funding
                return {'fee_paid': 0.0, 'net_payment': 0.0, 'processed': False}
        
        # Calculate funding using FeeCalculator if available
        if self._fee_calculator is not None:
            funding_result = self._fee_calculator.calculate_funding_fee(
                symbol=symbol,
                position_size=position_size,
                mark_price=mark_price,
                timestamp=timestamp
            )
            net_payment = funding_result.get('net_payment', 0.0)
            fee_paid = funding_result.get('fee_paid', 0.0)
            funding_rate = funding_result.get('funding_rate_annual', 0.0)
        else:
            # Simple funding calculation without FeeCalculator
            net_payment = 0.0
            fee_paid = 0.0
            funding_rate = 0.0
        
        # Update balance based on funding payment
        if net_payment != 0:
            self._total_balance += net_payment
            self._free_balance += net_payment
            
            if net_payment < 0:
                self._total_funding_paid += abs(net_payment)
            else:
                self._total_funding_received += net_payment
            
            # Record funding transaction
            funding_record = {
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'position_size': position_size,
                'mark_price': mark_price,
                'funding_rate_annual': funding_rate,
                'net_payment': net_payment,
                'is_payment': net_payment < 0
            }
            self._funding_history.append(funding_record)
            self._last_funding_time[symbol] = timestamp
            
            logger.debug(
                f"Funding processed: {symbol} {'paid' if net_payment < 0 else 'received'} "
                f"{abs(net_payment):.4f}"
            )
        
        return {
            'fee_paid': fee_paid,
            'net_payment': net_payment,
            'funding_rate': funding_rate,
            'processed': True
        }
    
    def update_funding_for_all_positions(
        self,
        current_prices: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update funding for all open positions.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping symbols to funding results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        funding_results = {}
        
        for position in self._positions.values():
            if position.symbol in current_prices:
                # Determine position size with sign based on side
                position_size = position.size if position.side == 'long' else -position.size
                
                result = self.process_funding(
                    symbol=position.symbol,
                    position_size=position_size,
                    mark_price=current_prices[position.symbol],
                    timestamp=timestamp
                )
                funding_results[position.symbol] = result
        
        return funding_results
    
    def get_total_funding_paid(self) -> float:
        """
        Get total funding fees paid.
        
        Returns:
            Total funding fees paid
        """
        return self._total_funding_paid
    
    def get_total_funding_received(self) -> float:
        """
        Get total funding fees received.
        
        Returns:
            Total funding fees received
        """
        return self._total_funding_received
    
    def get_net_funding(self) -> float:
        """
        Get net funding (received - paid).
        
        Returns:
            Net funding amount
        """
        return self._total_funding_received - self._total_funding_paid
    
    def get_fee_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get fee history with optional time filtering.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of fee records
        """
        records = self._fee_history
        
        if start_time:
            records = [r for r in records if datetime.fromisoformat(r['timestamp']) >= start_time]
        
        if end_time:
            records = [r for r in records if datetime.fromisoformat(r['timestamp']) <= end_time]
        
        return records
    
    def get_funding_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get funding history with optional time filtering.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of funding records
        """
        records = self._funding_history
        
        if start_time:
            records = [r for r in records if datetime.fromisoformat(r['timestamp']) >= start_time]
        
        if end_time:
            records = [r for r in records if datetime.fromisoformat(r['timestamp']) <= end_time]
        
        return records
    
    def get_fee_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive fee summary.
        
        Returns:
            Dictionary with fee statistics
        """
        summary = {
            'trading_fees': self._total_fees_paid,
            'funding_paid': self._total_funding_paid,
            'funding_received': self._total_funding_received,
            'net_funding': self.get_net_funding(),
            'total_costs': self._total_fees_paid + self._total_funding_paid
        }
        
        # Add FeeCalculator stats if available
        if self._fee_calculator is not None:
            summary['fee_calculator_stats'] = self._fee_calculator.get_cumulative_fees()
            summary['volume_stats'] = self._fee_calculator.get_volume_stats()
        
        return summary
    
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
    
    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        timestamp: Optional[datetime] = None,
        order_type: str = "limit",
        is_maker: bool = True
    ) -> bool:
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
        
        if timestamp is None:
            timestamp = datetime.now()

        fee = 0.0
        fee_type = 'maker' if is_maker else 'taker'
        if self._fee_calculator is not None:
            order_type_enum = self._parse_order_type(order_type, is_maker)
            fee_result = self._fee_calculator.calculate_trade_fee(
                symbol=symbol,
                amount=size,
                price=entry_price,
                order_type=order_type_enum,
                timestamp=timestamp,
                metadata={'side': side, 'is_maker': is_maker}
            )
            fee = fee_result['fee_paid']
            fee_type = fee_result['fee_type']

        # Record fee in history
        if fee > 0:
            self._record_fee({
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'order_id': f"entry_{symbol}",
                'side': side,
                'amount': size,
                'price': entry_price,
                'fee': fee,
                'fee_type': fee_type,
                'notional_value': size * entry_price
            })

            # Deduct fee from balance
            self._total_balance -= fee
            self._free_balance -= fee
            self._total_fees_paid += fee

        # Use existing _open_position method
        self._open_position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            timestamp=timestamp,
            fees=fee
        )
        return True
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: Optional[datetime] = None,
        order_type: str = "limit",
        is_maker: bool = True
    ) -> Optional[Position]:
        """
        Close a position by symbol (simplified interface for strategy engine).
        
        Args:
            symbol: Trading pair symbol
            exit_price: Exit price
            
        Returns:
            Closed Position if found, None otherwise
        """
        position = self._find_position_to_close(symbol)
        if not position:
            return None

        if timestamp is None:
            timestamp = datetime.now()

        fee = 0.0
        fee_type = 'maker' if is_maker else 'taker'
        if self._fee_calculator is not None:
            order_type_enum = self._parse_order_type(order_type, is_maker)
            fee_result = self._fee_calculator.calculate_trade_fee(
                symbol=symbol,
                amount=position.size,
                price=exit_price,
                order_type=order_type_enum,
                timestamp=timestamp,
                metadata={'side': 'sell' if position.side == 'long' else 'buy', 'is_maker': is_maker}
            )
            fee = fee_result['fee_paid']
            fee_type = fee_result['fee_type']

        # Record fee in history
        if fee > 0:
            self._record_fee({
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'order_id': f"exit_{symbol}",
                'side': 'sell' if position.side == 'long' else 'buy',
                'amount': position.size,
                'price': exit_price,
                'fee': fee,
                'fee_type': fee_type,
                'notional_value': position.size * exit_price
            })

            self._total_balance -= fee
            self._free_balance -= fee
            self._total_fees_paid += fee

        return self._close_position(
            position_id=position.position_id,
            exit_price=exit_price,
            timestamp=timestamp,
            exit_fees=fee
        )
    
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
        stats = {
            'initial_balance': self._initial_balance,
            'total_balance': self._total_balance,
            'free_balance': self._free_balance,
            'used_balance': self._used_balance,
            'equity': self.get_equity(),
            'total_realized_pnl': self._total_realized_pnl,
            'total_fees_paid': self._total_fees_paid,
            'total_funding_paid': self._total_funding_paid,
            'total_funding_received': self._total_funding_received,
            'net_funding': self.get_net_funding(),
            'num_open_positions': len(self._positions),
            'num_closed_positions': len(self._closed_positions),
            'num_fee_transactions': len(self._fee_history),
            'num_funding_transactions': len(self._funding_history),
            'currency': self._currency
        }
        
        # Add fee calculator stats if available
        if self._fee_calculator is not None:
            stats['fee_rates'] = self._fee_calculator.get_current_fee_rates()
            stats['cumulative_fees'] = self._fee_calculator.get_cumulative_fees()
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of AccountState."""
        return (
            f"AccountState(balance={self._total_balance:.2f}, "
            f"equity={self.get_equity():.2f}, "
            f"positions={len(self._positions)}, "
            f"realized_pnl={self._total_realized_pnl:.2f}, "
            f"fees={self._total_fees_paid:.2f}, "
            f"net_funding={self.get_net_funding():.2f})"
        )
