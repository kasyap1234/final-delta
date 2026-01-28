"""
Trade history module for backtesting.

This module provides trade history tracking for the backtesting system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
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
    holding_period_minutes: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'realized_pnl': self.realized_pnl,
            'realized_pnl_percent': self.realized_pnl_percent,
            'fees_paid': self.fees_paid,
            'risk_reward_ratio': self.risk_reward_ratio,
            'risk_amount': self.risk_amount,
            'holding_period_minutes': self.holding_period_minutes
        }


class TradeHistory:
    """
    Track all trades during backtesting.
    
    This class handles:
    - Recording all completed trades
    - Calculating trade statistics
    - Filtering and querying trades
    """
    
    def __init__(self):
        """Initialize trade history."""
        self._trades: List[TradeRecord] = []
        self._trade_counter = 0
        
        logger.info("TradeHistory initialized")
    
    def add_trade(self, trade: TradeRecord) -> None:
        """
        Add a trade to history.
        
        Args:
            trade: Trade record to add
        """
        # Calculate holding period
        holding_period = (trade.exit_time - trade.entry_time).total_seconds() / 60
        trade.holding_period_minutes = holding_period
        
        self._trades.append(trade)
        logger.debug(f"Trade added: {trade.trade_id}, P&L={trade.realized_pnl:.2f}")
    
    def get_all_trades(self) -> List[TradeRecord]:
        """
        Get all trades.
        
        Returns:
            List of all trades
        """
        return list(self._trades)
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """
        Get trades for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of trades for the symbol
        """
        return [t for t in self._trades if t.symbol == symbol]
    
    def get_trades_by_side(self, side: str) -> List[TradeRecord]:
        """
        Get trades by side (long/short).
        
        Args:
            side: Position side ('long' or 'short')
            
        Returns:
            List of trades with the specified side
        """
        return [t for t in self._trades if t.side == side]
    
    def get_winning_trades(self) -> List[TradeRecord]:
        """
        Get all winning trades.
        
        Returns:
            List of trades with positive P&L
        """
        return [t for t in self._trades if t.realized_pnl > 0]
    
    def get_losing_trades(self) -> List[TradeRecord]:
        """
        Get all losing trades.
        
        Returns:
            List of trades with negative P&L
        """
        return [t for t in self._trades if t.realized_pnl < 0]
    
    def get_total_trades(self) -> int:
        """
        Get total number of trades.
        
        Returns:
            Total number of trades
        """
        return len(self._trades)
    
    def get_total_pnl(self) -> float:
        """
        Get total realized P&L.
        
        Returns:
            Total realized P&L
        """
        return sum(t.realized_pnl for t in self._trades)
    
    def get_total_fees(self) -> float:
        """
        Get total fees paid.
        
        Returns:
            Total fees paid
        """
        return sum(t.fees_paid for t in self._trades)
    
    def get_win_rate(self) -> float:
        """
        Calculate win rate.
        
        Returns:
            Win rate as percentage (0.0 to 1.0)
        """
        if not self._trades:
            return 0.0
        
        winning_trades = len(self.get_winning_trades())
        return winning_trades / len(self._trades)
    
    def get_avg_win(self) -> float:
        """
        Calculate average winning trade.
        
        Returns:
            Average profit from winning trades
        """
        winning_trades = self.get_winning_trades()
        if not winning_trades:
            return 0.0
        
        return sum(t.realized_pnl for t in winning_trades) / len(winning_trades)
    
    def get_avg_loss(self) -> float:
        """
        Calculate average losing trade.
        
        Returns:
            Average loss from losing trades
        """
        losing_trades = self.get_losing_trades()
        if not losing_trades:
            return 0.0
        
        return sum(t.realized_pnl for t in losing_trades) / len(losing_trades)
    
    def get_profit_factor(self) -> float:
        """
        Calculate profit factor.
        
        Returns:
            Profit factor (gross profit / gross loss)
        """
        gross_profit = sum(t.realized_pnl for t in self.get_winning_trades())
        gross_loss = abs(sum(t.realized_pnl for t in self.get_losing_trades()))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_avg_holding_period(self) -> float:
        """
        Calculate average holding period.
        
        Returns:
            Average holding period in minutes
        """
        if not self._trades:
            return 0.0
        
        return sum(t.holding_period_minutes for t in self._trades) / len(self._trades)
    
    def get_best_trade(self) -> Optional[TradeRecord]:
        """
        Get the best (most profitable) trade.
        
        Returns:
            Best trade or None
        """
        if not self._trades:
            return None
        
        return max(self._trades, key=lambda t: t.realized_pnl)
    
    def get_worst_trade(self) -> Optional[TradeRecord]:
        """
        Get the worst (least profitable) trade.
        
        Returns:
            Worst trade or None
        """
        if not self._trades:
            return None
        
        return min(self._trades, key=lambda t: t.realized_pnl)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get trade statistics.
        
        Returns:
            Dictionary with trade statistics
        """
        winning_trades = self.get_winning_trades()
        losing_trades = self.get_losing_trades()
        
        return {
            'total_trades': self.get_total_trades(),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': self.get_win_rate(),
            'total_pnl': self.get_total_pnl(),
            'total_fees': self.get_total_fees(),
            'avg_win': self.get_avg_win(),
            'avg_loss': self.get_avg_loss(),
            'profit_factor': self.get_profit_factor(),
            'avg_holding_period_minutes': self.get_avg_holding_period(),
            'best_trade': self.get_best_trade().to_dict() if self.get_best_trade() else None,
            'worst_trade': self.get_worst_trade().to_dict() if self.get_worst_trade() else None
        }
    
    def clear(self) -> None:
        """Clear all trades."""
        self._trades.clear()
        self._trade_counter = 0
        logger.info("TradeHistory cleared")
    
    def _generate_trade_id(self) -> str:
        """
        Generate a unique trade ID.
        
        Returns:
            Unique trade ID
        """
        self._trade_counter += 1
        timestamp = int(datetime.now().timestamp())
        return f"trade_{self._trade_counter}_{timestamp}"
