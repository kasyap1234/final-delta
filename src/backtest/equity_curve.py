"""
Equity curve module for backtesting.

This module provides equity curve tracking for the backtesting system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class EquityPoint:
    """Single point on the equity curve."""
    timestamp: datetime
    equity: float
    balance: float
    free_balance: float
    used_balance: float
    unrealized_pnl: float
    realized_pnl: float
    num_positions: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'equity': self.equity,
            'balance': self.balance,
            'free_balance': self.free_balance,
            'used_balance': self.used_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'num_positions': self.num_positions
        }


class EquityCurve:
    """
    Track account equity over time during backtesting.
    
    This class handles:
    - Recording equity points at each time step
    - Calculating drawdowns
    - Analyzing equity curve statistics
    """
    
    def __init__(self, initial_balance: float):
        """
        Initialize equity curve.
        
        Args:
            initial_balance: Starting account balance
        """
        self._initial_balance = initial_balance
        self._points: List[EquityPoint] = []
        self._peak_equity = initial_balance
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0
        
        logger.info(f"EquityCurve initialized with initial balance: {initial_balance}")
    
    def add_point(
        self,
        timestamp: datetime,
        equity: float,
        balance: float,
        free_balance: float,
        used_balance: float,
        unrealized_pnl: float,
        realized_pnl: float,
        num_positions: int
    ) -> None:
        """
        Add an equity point to the curve.
        
        Args:
            timestamp: Point timestamp
            equity: Total account equity
            balance: Account balance
            free_balance: Free balance
            used_balance: Used balance
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            num_positions: Number of open positions
        """
        point = EquityPoint(
            timestamp=timestamp,
            equity=equity,
            balance=balance,
            free_balance=free_balance,
            used_balance=used_balance,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            num_positions=num_positions
        )
        
        self._points.append(point)
        
        # Update peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._current_drawdown = 0.0
        else:
            self._current_drawdown = (self._peak_equity - equity) / self._peak_equity
            if self._current_drawdown > self._max_drawdown:
                self._max_drawdown = self._current_drawdown
        
        logger.debug(f"Equity point added: {equity:.2f} at {timestamp}")
    
    def get_points(self) -> List[EquityPoint]:
        """
        Get all equity points.
        
        Returns:
            List of equity points
        """
        return list(self._points)
    
    def get_equity_values(self) -> List[float]:
        """
        Get equity values only.
        
        Returns:
            List of equity values
        """
        return [p.equity for p in self._points]
    
    def get_timestamps(self) -> List[datetime]:
        """
        Get timestamps only.
        
        Returns:
            List of timestamps
        """
        return [p.timestamp for p in self._points]
    
    def get_initial_balance(self) -> float:
        """
        Get initial balance.
        
        Returns:
            Initial balance
        """
        return self._initial_balance
    
    def get_final_equity(self) -> float:
        """
        Get final equity.
        
        Returns:
            Final equity value
        """
        if not self._points:
            return self._initial_balance
        
        return self._points[-1].equity
    
    def get_total_return(self) -> float:
        """
        Calculate total return.
        
        Returns:
            Total return as percentage
        """
        final_equity = self.get_final_equity()
        return (final_equity - self._initial_balance) / self._initial_balance
    
    def get_peak_equity(self) -> float:
        """
        Get peak equity.
        
        Returns:
            Peak equity value
        """
        return self._peak_equity
    
    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown.
        
        Returns:
            Maximum drawdown as percentage (0.0 to 1.0)
        """
        return self._max_drawdown
    
    def get_current_drawdown(self) -> float:
        """
        Get current drawdown.
        
        Returns:
            Current drawdown as percentage (0.0 to 1.0)
        """
        return self._current_drawdown
    
    def get_num_points(self) -> int:
        """
        Get number of equity points.
        
        Returns:
            Number of points
        """
        return len(self._points)
    
    def get_equity_at_time(self, timestamp: datetime) -> Optional[float]:
        """
        Get equity at a specific timestamp.
        
        Args:
            timestamp: Timestamp to query
            
        Returns:
            Equity value at timestamp, or None if not found
        """
        for point in self._points:
            if point.timestamp == timestamp:
                return point.equity
        
        return None
    
    def get_drawdown_curve(self) -> List[float]:
        """
        Get drawdown curve.
        
        Returns:
            List of drawdown values at each point
        """
        drawdowns = []
        peak = self._initial_balance
        
        for point in self._points:
            if point.equity > peak:
                peak = point.equity
            
            drawdown = (peak - point.equity) / peak if peak > 0 else 0.0
            drawdowns.append(drawdown)
        
        return drawdowns
    
    def get_returns_curve(self) -> List[float]:
        """
        Get returns curve (percentage changes).
        
        Returns:
            List of return values
        """
        if len(self._points) < 2:
            return []
        
        returns = []
        for i in range(1, len(self._points)):
            prev_equity = self._points[i-1].equity
            curr_equity = self._points[i].equity
            
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return returns
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get equity curve statistics.
        
        Returns:
            Dictionary with statistics
        """
        equity_values = self.get_equity_values()
        
        if not equity_values:
            return {
                'initial_balance': self._initial_balance,
                'final_equity': self._initial_balance,
                'total_return': 0.0,
                'peak_equity': self._initial_balance,
                'max_drawdown': 0.0,
                'num_points': 0
            }
        
        import numpy as np
        
        returns = self.get_returns_curve()
        
        stats = {
            'initial_balance': self._initial_balance,
            'final_equity': self.get_final_equity(),
            'total_return': self.get_total_return(),
            'peak_equity': self._peak_equity,
            'max_drawdown': self._max_drawdown,
            'num_points': len(self._points)
        }
        
        if returns:
            stats['avg_return'] = np.mean(returns)
            stats['std_return'] = np.std(returns)
            stats['min_return'] = np.min(returns)
            stats['max_return'] = np.max(returns)
        
        return stats
    
    def clear(self) -> None:
        """Clear all equity points."""
        self._points.clear()
        self._peak_equity = self._initial_balance
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0
        logger.info("EquityCurve cleared")
