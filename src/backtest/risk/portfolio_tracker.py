"""Portfolio Tracker module for backtesting.

This module provides portfolio tracking, P&L calculation, and risk reporting for backtesting.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class PositionStatus(str, Enum):
    """Position status types."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position:
    """Trading position data."""
    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    risk_amount: float
    status: PositionStatus = PositionStatus.OPEN
    
    # Mutable fields
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    fees_paid: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_amount': self.risk_amount,
            'status': self.status.value,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_percent': self.unrealized_pnl_percent,
            'realized_pnl': self.realized_pnl,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'fees_paid': self.fees_paid
        }


@dataclass
class TradeRecord:
    """Completed trade record."""
    trade_id: str
    position_id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    realized_pnl_percent: float
    fees_paid: float
    risk_reward_ratio: float
    risk_amount: float


@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot."""
    timestamp: datetime
    account_balance: float
    equity: float
    total_exposure: float
    total_risk: float
    unrealized_pnl: float
    realized_pnl_today: float
    num_open_positions: int
    num_closed_positions_today: int
    drawdown_percent: float


@dataclass
class RiskReport:
    """Comprehensive risk report."""
    generated_at: datetime
    account_balance: float
    total_equity: float
    total_exposure: float
    total_exposure_percent: float
    total_risk: float
    total_risk_percent: float
    unrealized_pnl: float
    realized_pnl_today: float
    realized_pnl_week: float
    realized_pnl_month: float
    num_open_positions: int
    num_trades_today: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: Optional[float] = None
    max_drawdown_percent: float = 0.0
    current_drawdown_percent: float = 0.0


class PortfolioTracker:
    """Portfolio tracker for managing positions and calculating P&L for backtesting.
    
    This class handles:
    - Tracking all open positions
    - Calculating unrealized and realized P&L
    - Managing account balance
    - Generating risk reports
    - Historical trade tracking
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """Initialize the PortfolioTracker.
        
        Args:
            initial_balance: Starting account balance
        """
        self._account_balance = initial_balance
        self._initial_balance = initial_balance
        self._peak_equity = initial_balance
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        
        # Position tracking
        self._open_positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._trade_history: List[TradeRecord] = []
        
        # Daily tracking
        self._daily_realized_pnl: Dict[str, float] = defaultdict(float)
        self._daily_trades: Dict[str, int] = defaultdict(int)
        
        # Snapshots for historical analysis
        self._snapshots: List[PortfolioSnapshot] = []
        
        # Position ID counter
        self._position_counter = 0
        
        # Current time for backtest
        self._current_time: Optional[datetime] = None
    
    def set_current_time(self, current_time: datetime) -> None:
        """Set current time for backtest.
        
        Args:
            current_time: Current backtest timestamp
        """
        self._current_time = current_time
    
    def add_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_amount: float,
        position_id: Optional[str] = None,
        entry_time: Optional[datetime] = None,
        fees: float = 0.0
    ) -> Position:
        """Add a new position to the portfolio.
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('long' or 'short')
            size: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            risk_amount: Dollar amount at risk
            position_id: Optional position ID (auto-generated if None)
            entry_time: Optional entry time (current time if None)
            fees: Entry fees paid
            
        Returns:
            Created Position object
        """
        if position_id is None:
            self._position_counter += 1
            timestamp = int(self._current_time.timestamp()) if self._current_time else int(datetime.now().timestamp())
            position_id = f"{symbol.replace('/', '_')}_{self._position_counter}_{timestamp}"
        
        if entry_time is None:
            entry_time = self._current_time if self._current_time else datetime.now()
        
        position = Position(
            position_id=position_id,
            symbol=symbol,
            side=side.lower(),
            size=size,
            entry_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            current_price=entry_price,
            fees_paid=fees
        )
        
        self._open_positions[position_id] = position
        
        # Deduct fees from balance
        self._account_balance -= fees
        
        return position
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        exit_fees: float = 0.0
    ) -> Optional[TradeRecord]:
        """Close an open position.
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            exit_time: Optional exit time (current time if None)
            exit_fees: Exit fees paid
            
        Returns:
            TradeRecord if position was found and closed, None otherwise
        """
        if position_id not in self._open_positions:
            return None
        
        position = self._open_positions[position_id]
        
        if exit_time is None:
            exit_time = self._current_time if self._current_time else datetime.now()
        
        # Calculate realized P&L
        if position.side == 'long':
            price_diff = exit_price - position.entry_price
        else:  # short
            price_diff = position.entry_price - exit_price
        
        realized_pnl = price_diff * position.size
        realized_pnl_percent = (price_diff / position.entry_price * 100) if position.entry_price != 0 else 0
        
        # Update position
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = exit_time
        position.realized_pnl = realized_pnl
        position.fees_paid += exit_fees
        
        # Calculate risk:reward ratio achieved
        risk_reward_ratio = 0.0
        if position.risk_amount > 0:
            risk_reward_ratio = realized_pnl / position.risk_amount
        
        # Move to closed positions
        del self._open_positions[position_id]
        self._closed_positions.append(position)
        
        # Create trade record
        trade = TradeRecord(
            trade_id=f"trade_{position_id}",
            position_id=position_id,
            symbol=position.symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            realized_pnl=realized_pnl,
            realized_pnl_percent=realized_pnl_percent,
            fees_paid=position.fees_paid,
            risk_reward_ratio=risk_reward_ratio,
            risk_amount=position.risk_amount
        )
        
        self._trade_history.append(trade)
        
        # Update balance and daily tracking
        self._account_balance += realized_pnl - exit_fees
        
        today = exit_time.strftime('%Y-%m-%d')
        self._daily_realized_pnl[today] += realized_pnl
        self._daily_trades[today] += 1
        
        # Update drawdown tracking
        self._update_drawdown()
        
        return trade
    
    def update_position_price(self, position_id: str, current_price: float) -> Optional[Position]:
        """Update current price for a position and recalculate unrealized P&L.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            Updated Position if found, None otherwise
        """
        if position_id not in self._open_positions:
            return None
        
        position = self._open_positions[position_id]
        position.current_price = current_price
        
        # Calculate unrealized P&L
        if position.side == 'long':
            price_diff = current_price - position.entry_price
        else:  # short
            price_diff = position.entry_price - current_price
        
        position.unrealized_pnl = price_diff * position.size
        position.unrealized_pnl_percent = (price_diff / position.entry_price * 100) if position.entry_price != 0 else 0
        
        return position
    
    def update_all_positions(self, price_updates: Dict[str, float]) -> None:
        """Update prices for all positions.
        
        Args:
            price_updates: Dictionary mapping symbols to current prices
        """
        for position in self._open_positions.values():
            if position.symbol in price_updates:
                self.update_position_price(position.position_id, price_updates[position.symbol])
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position if found, None otherwise
        """
        return self._open_positions.get(position_id)
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get position by symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position if found, None otherwise
        """
        for position in self._open_positions.values():
            if position.symbol == symbol:
                return position
        return None
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open positions, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of positions as dictionaries
        """
        positions = []
        for pos in self._open_positions.values():
            if symbol is None or pos.symbol == symbol:
                positions.append(pos.to_dict())
        return positions
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions as Position objects.
        
        Returns:
            List of Position objects
        """
        return list(self._open_positions.values())
    
    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if position exists, False otherwise
        """
        for pos in self._open_positions.values():
            if pos.symbol == symbol:
                return True
        return False
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L for all open positions.
        
        Returns:
            Total unrealized P&L
        """
        return sum(pos.unrealized_pnl for pos in self._open_positions.values())
    
    def calculate_realized_pnl(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> float:
        """Calculate realized P&L for a time period.
        
        Args:
            start_time: Start of period (inclusive)
            end_time: End of period (inclusive)
            
        Returns:
            Total realized P&L
        """
        total = 0.0
        for trade in self._trade_history:
            if start_time and trade.exit_time < start_time:
                continue
            if end_time and trade.exit_time > end_time:
                continue
            total += trade.realized_pnl
        return total
    
    def get_equity(self) -> float:
        """Calculate total account equity (balance + unrealized P&L).
        
        Returns:
            Total equity
        """
        return self._account_balance + self.calculate_unrealized_pnl()
    
    def get_account_balance(self) -> float:
        """Get current account balance.
        
        Returns:
            Account balance
        """
        return self._account_balance
    
    def set_account_balance(self, balance: float) -> None:
        """Set account balance.
        
        Args:
            balance: New account balance
        """
        self._account_balance = balance
        self._update_drawdown()
    
    def get_total_exposure(self) -> float:
        """Calculate total portfolio exposure.
        
        Returns:
            Total exposure (sum of position values)
        """
        return sum(
            pos.size * pos.current_price 
            for pos in self._open_positions.values()
        )
    
    def get_total_risk(self) -> float:
        """Calculate total risk across all positions.
        
        Returns:
            Total risk amount
        """
        return sum(
            pos.risk_amount 
            for pos in self._open_positions.values()
        )
    
    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Get realized P&L for a specific date.
        
        Args:
            date: Date to get P&L for (defaults to today)
            
        Returns:
            Realized P&L for the date
        """
        if date is None:
            date = self._current_time if self._current_time else datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        return self._daily_realized_pnl.get(date_str, 0.0)
    
    def get_weekly_pnl(self) -> float:
        """Get realized P&L for the current week.
        
        Returns:
            Realized P&L for current week
        """
        today = self._current_time if self._current_time else datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        
        return self.calculate_realized_pnl(start_time=start_of_week)
    
    def get_monthly_pnl(self) -> float:
        """Get realized P&L for the current month.
        
        Returns:
            Realized P&L for current month
        """
        today = self._current_time if self._current_time else datetime.now()
        start_of_month = today.replace(day=1)
        
        return self.calculate_realized_pnl(start_time=start_of_month)
    
    def generate_risk_report(self) -> RiskReport:
        """Generate comprehensive risk report.
        
        Returns:
            RiskReport with portfolio statistics
        """
        equity = self.get_equity()
        balance = self._account_balance
        unrealized_pnl = self.calculate_unrealized_pnl()
        total_exposure = self.get_total_exposure()
        total_risk = self.get_total_risk()
        
        # Calculate percentages
        exposure_percent = (total_exposure / balance * 100) if balance > 0 else 0
        risk_percent = (total_risk / balance * 100) if balance > 0 else 0
        
        # Get P&L for different periods
        realized_today = self.get_daily_pnl()
        realized_week = self.get_weekly_pnl()
        realized_month = self.get_monthly_pnl()
        
        # Calculate win/loss statistics
        win_count = sum(1 for t in self._trade_history if t.realized_pnl > 0)
        loss_count = sum(1 for t in self._trade_history if t.realized_pnl <= 0)
        total_trades = win_count + loss_count
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        wins = [t.realized_pnl for t in self._trade_history if t.realized_pnl > 0]
        losses = [abs(t.realized_pnl) for t in self._trade_history if t.realized_pnl <= 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        total_wins = sum(wins)
        total_losses = sum(losses)
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = None
        if len(self._trade_history) > 1:
            returns = [t.realized_pnl_percent for t in self._trade_history]
            if len(returns) > 1:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                std_return = variance ** 0.5
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)  # Annualized
        
        # Today's trade count
        today = self._current_time if self._current_time else datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        trades_today = self._daily_trades.get(today_str, 0)
        
        # Drawdown
        current_dd = self._get_current_drawdown_percent()
        max_dd = self._max_drawdown
        
        return RiskReport(
            generated_at=today,
            account_balance=balance,
            total_equity=equity,
            total_exposure=total_exposure,
            total_exposure_percent=exposure_percent,
            total_risk=total_risk,
            total_risk_percent=risk_percent,
            unrealized_pnl=unrealized_pnl,
            realized_pnl_today=realized_today,
            realized_pnl_week=realized_week,
            realized_pnl_month=realized_month,
            num_open_positions=len(self._open_positions),
            num_trades_today=trades_today,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_percent=max_dd,
            current_drawdown_percent=current_dd
        )
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state.
        
        Returns:
            PortfolioSnapshot
        """
        equity = self.get_equity()
        today = self._current_time if self._current_time else datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        
        snapshot = PortfolioSnapshot(
            timestamp=today,
            account_balance=self._account_balance,
            equity=equity,
            total_exposure=self.get_total_exposure(),
            total_risk=self.get_total_risk(),
            unrealized_pnl=self.calculate_unrealized_pnl(),
            realized_pnl_today=self._daily_realized_pnl.get(today_str, 0.0),
            num_open_positions=len(self._open_positions),
            num_closed_positions_today=self._daily_trades.get(today_str, 0),
            drawdown_percent=self._get_current_drawdown_percent()
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TradeRecord]:
        """Get trade history, optionally filtered.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of trades to return
            
        Returns:
            List of TradeRecord objects
        """
        trades = self._trade_history
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Sort by exit time descending
        trades = sorted(trades, key=lambda t: t.exit_time, reverse=True)
        
        if limit:
            trades = trades[:limit]
        
        return trades
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        total_return = ((self._account_balance - self._initial_balance) / self._initial_balance * 100) if self._initial_balance > 0 else 0
        
        win_count = sum(1 for t in self._trade_history if t.realized_pnl > 0)
        loss_count = len(self._trade_history) - win_count
        
        wins = [t.realized_pnl for t in self._trade_history if t.realized_pnl > 0]
        losses = [abs(t.realized_pnl) for t in self._trade_history if t.realized_pnl <= 0]
        
        return {
            'initial_balance': self._initial_balance,
            'current_balance': self._account_balance,
            'total_return_percent': total_return,
            'total_trades': len(self._trade_history),
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': (win_count / len(self._trade_history) * 100) if self._trade_history else 0,
            'avg_win': (sum(wins) / len(wins)) if wins else 0,
            'avg_loss': (sum(losses) / len(losses)) if losses else 0,
            'largest_win': max(wins) if wins else 0,
            'largest_loss': max(losses) if losses else 0,
            'max_drawdown_percent': self._max_drawdown,
            'current_drawdown_percent': self._get_current_drawdown_percent(),
            'open_positions': len(self._open_positions),
            'total_unrealized_pnl': self.calculate_unrealized_pnl()
        }
    
    def _update_drawdown(self) -> None:
        """Update drawdown calculations."""
        equity = self.get_equity()
        
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._current_drawdown = 0.0
        else:
            self._current_drawdown = self._peak_equity - equity
            current_dd_percent = (self._current_drawdown / self._peak_equity * 100) if self._peak_equity > 0 else 0
            self._max_drawdown = max(self._max_drawdown, current_dd_percent)
    
    def _get_current_drawdown_percent(self) -> float:
        """Get current drawdown as percentage."""
        if self._peak_equity > 0:
            return (self._current_drawdown / self._peak_equity * 100)
        return 0.0
    
    def reset(self) -> None:
        """Reset portfolio tracker to initial state."""
        self._account_balance = self._initial_balance
        self._peak_equity = self._initial_balance
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._open_positions.clear()
        self._closed_positions.clear()
        self._trade_history.clear()
        self._daily_realized_pnl.clear()
        self._daily_trades.clear()
        self._snapshots.clear()
        self._position_counter = 0
