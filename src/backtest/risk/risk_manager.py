"""Risk Manager module for backtesting.

This module provides portfolio-level risk management including exposure tracking,
risk limits, and position validation for backtesting.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class RiskStatus(str, Enum):
    """Risk check status."""
    ALLOWED = "allowed"
    DENIED = "denied"
    WARNING = "warning"


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    can_trade: bool
    status: RiskStatus
    reason: Optional[str] = None
    current_exposure: float = 0.0
    max_exposure: float = 0.0
    current_risk: float = 0.0
    max_risk: float = 0.0
    remaining_capacity: float = 0.0


@dataclass
class PositionRisk:
    """Risk information for a position."""
    position_id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss_price: float
    risk_amount: float
    risk_percent: float
    unrealized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)


@dataclass
class DailyRiskMetrics:
    """Daily risk tracking metrics."""
    date: datetime
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    max_drawdown: float = 0.0
    total_risk_taken: float = 0.0


class RiskManager:
    """Portfolio-level risk manager for backtesting.
    
    This class handles:
    - Position validation before opening
    - Total exposure tracking
    - Risk limit enforcement
    - Daily/weekly loss limits
    - Correlation-adjusted position sizing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RiskManager.
        
        Args:
            config: Configuration dictionary with risk management settings.
                   Expected keys:
                   - max_total_exposure_percent: Max portfolio exposure (default: 80.0)
                   - max_total_risk_percent: Max total risk (default: 5.0)
                   - max_positions: Maximum number of positions (default: 10)
                   - daily_loss_limit_percent: Daily loss limit (default: 3.0)
                   - weekly_loss_limit_percent: Weekly loss limit (default: 10.0)
                   - max_correlated_exposure: Max exposure to correlated assets (default: 15.0)
                   - correlation_threshold: Correlation threshold for grouping (default: 0.7)
        """
        self.config = config or {}
        self.max_total_exposure_percent = self.config.get('max_total_exposure_percent', 80.0)
        self.max_total_risk_percent = self.config.get('max_total_risk_percent', 5.0)
        self.max_positions = self.config.get('max_positions', 10)
        self.daily_loss_limit_percent = self.config.get('daily_loss_limit_percent', 3.0)
        self.weekly_loss_limit_percent = self.config.get('weekly_loss_limit_percent', 10.0)
        self.max_correlated_exposure = self.config.get('max_correlated_exposure', 15.0)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        # Track position risks
        self.position_risks: Dict[str, PositionRisk] = {}
        
        # Track daily metrics
        self.daily_metrics: Dict[str, DailyRiskMetrics] = {}
        
        # Track correlation groups
        self.correlation_groups: Dict[str, List[str]] = {}
        
        # Account balance tracking
        self._account_balance: float = 0.0
        self._peak_balance: float = 0.0
        self._current_drawdown: float = 0.0
        
        # Current time for backtest
        self._current_time: Optional[datetime] = None
    
    def set_current_time(self, current_time: datetime) -> None:
        """Set current time for backtest.
        
        Args:
            current_time: Current backtest timestamp
        """
        self._current_time = current_time
    
    def can_open_position(
        self,
        symbol: str,
        position_size: float,
        stop_loss_price: float,
        entry_price: float,
        current_positions: List[Dict[str, Any]],
        account_balance: Optional[float] = None
    ) -> RiskCheckResult:
        """Check if a new position can be opened based on risk rules.
        
        Args:
            symbol: Trading pair symbol
            position_size: Proposed position size
            stop_loss_price: Stop loss price
            entry_price: Entry price
            current_positions: List of current open positions
            account_balance: Current account balance (uses tracked balance if None)
            
        Returns:
            RiskCheckResult with trading permission and details
        """
        balance = account_balance if account_balance is not None else self._account_balance
        
        if balance <= 0:
            return RiskCheckResult(
                can_trade=False,
                status=RiskStatus.DENIED,
                reason="Account balance is zero or negative"
            )
        
        # Check position count limit
        if len(current_positions) >= self.max_positions:
            return RiskCheckResult(
                can_trade=False,
                status=RiskStatus.DENIED,
                reason=f"Maximum number of positions ({self.max_positions}) reached"
            )
        
        # Calculate current exposure
        current_exposure = self.calculate_total_exposure(current_positions)
        position_value = position_size * entry_price
        new_exposure = current_exposure + position_value
        max_exposure = balance * (self.max_total_exposure_percent / 100)
        
        # Check total exposure limit
        if new_exposure > max_exposure:
            return RiskCheckResult(
                can_trade=False,
                status=RiskStatus.DENIED,
                reason=f"Position would exceed max exposure limit ({self.max_total_exposure_percent}%)",
                current_exposure=current_exposure,
                max_exposure=max_exposure,
                remaining_capacity=max(0, max_exposure - current_exposure)
            )
        
        # Calculate current risk
        current_risk = self.calculate_total_risk(current_positions)
        stop_loss_distance = abs(entry_price - stop_loss_price)
        new_position_risk = position_size * stop_loss_distance
        new_total_risk = current_risk + new_position_risk
        max_risk = balance * (self.max_total_risk_percent / 100)
        
        # Check total risk limit
        if new_total_risk > max_risk:
            return RiskCheckResult(
                can_trade=False,
                status=RiskStatus.DENIED,
                reason=f"Position would exceed max risk limit ({self.max_total_risk_percent}%)",
                current_exposure=current_exposure,
                max_exposure=max_exposure,
                current_risk=current_risk,
                max_risk=max_risk,
                remaining_capacity=max(0, max_risk - current_risk)
            )
        
        # Check daily loss limit
        daily_pnl = self.get_daily_pnl()
        daily_limit = balance * (self.daily_loss_limit_percent / 100)
        if daily_pnl < -daily_limit:
            return RiskCheckResult(
                can_trade=False,
                status=RiskStatus.DENIED,
                reason=f"Daily loss limit ({self.daily_loss_limit_percent}%) exceeded",
                current_exposure=current_exposure,
                max_exposure=max_exposure,
                current_risk=current_risk,
                max_risk=max_risk,
                remaining_capacity=0.0
            )
        
        # Check correlation-adjusted exposure
        correlation_check = self._check_correlation_exposure(
            symbol, position_value, current_positions, balance
        )
        if not correlation_check['allowed']:
            return RiskCheckResult(
                can_trade=False,
                status=RiskStatus.DENIED,
                reason=correlation_check['reason'],
                current_exposure=current_exposure,
                max_exposure=max_exposure,
                current_risk=current_risk,
                max_risk=max_risk,
                remaining_capacity=max(0, max_risk - current_risk)
            )
        
        # All checks passed
        return RiskCheckResult(
            can_trade=True,
            status=RiskStatus.ALLOWED,
            current_exposure=current_exposure,
            max_exposure=max_exposure,
            current_risk=current_risk,
            max_risk=max_risk,
            remaining_capacity=max_risk - new_total_risk
        )
    
    def calculate_total_exposure(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate total portfolio exposure.
        
        Args:
            positions: List of open positions
            
        Returns:
            Total exposure in base currency (sum of absolute position values)
        """
        total = 0.0
        for pos in positions:
            size = abs(pos.get('size', 0))
            price = pos.get('current_price', pos.get('entry_price', 0))
            total += size * price
        return total
    
    def calculate_total_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate total risk across all positions.
        
        Args:
            positions: List of open positions
            
        Returns:
            Total risk amount (sum of individual position risks)
        """
        total_risk = 0.0
        for pos in positions:
            size = abs(pos.get('size', 0))
            entry = pos.get('entry_price', 0)
            stop = pos.get('stop_loss', entry)
            
            if stop != entry:
                stop_distance = abs(entry - stop)
                total_risk += size * stop_distance
        
        return total_risk
    
    def check_daily_loss_limit(
        self,
        daily_pnl: float,
        account_balance: float,
        limit_percent: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if daily loss limit has been exceeded.
        
        Args:
            daily_pnl: Current daily P&L
            account_balance: Current account balance
            limit_percent: Optional override for daily loss limit percent
            
        Returns:
            Dictionary with limit check results
        """
        limit = limit_percent if limit_percent is not None else self.daily_loss_limit_percent
        limit_amount = account_balance * (limit / 100)
        
        limit_breached = daily_pnl < -limit_amount
        percent_used = (abs(daily_pnl) / limit_amount * 100) if limit_amount > 0 else 0
        
        return {
            'limit_breached': limit_breached,
            'daily_pnl': daily_pnl,
            'limit_amount': limit_amount,
            'limit_percent': limit,
            'percent_used': min(percent_used, 100.0),
            'remaining': max(0, limit_amount - abs(daily_pnl))
        }
    
    def check_weekly_loss_limit(
        self,
        weekly_pnl: float,
        account_balance: float,
        limit_percent: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if weekly loss limit has been exceeded.
        
        Args:
            weekly_pnl: Current weekly P&L
            account_balance: Current account balance
            limit_percent: Optional override for weekly loss limit percent
            
        Returns:
            Dictionary with limit check results
        """
        limit = limit_percent if limit_percent is not None else self.weekly_loss_limit_percent
        limit_amount = account_balance * (limit / 100)
        
        limit_breached = weekly_pnl < -limit_amount
        percent_used = (abs(weekly_pnl) / limit_amount * 100) if limit_amount > 0 else 0
        
        return {
            'limit_breached': limit_breached,
            'weekly_pnl': weekly_pnl,
            'limit_amount': limit_amount,
            'limit_percent': limit,
            'percent_used': min(percent_used, 100.0),
            'remaining': max(0, limit_amount - abs(weekly_pnl))
        }
    
    def get_remaining_risk_capacity(
        self,
        account_balance: float,
        current_risk: Optional[float] = None,
        current_positions: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Get remaining risk capacity before hitting limits.
        
        Args:
            account_balance: Current account balance
            current_risk: Current total risk (calculated from positions if None)
            current_positions: List of current positions (used if current_risk is None)
            
        Returns:
            Remaining risk capacity in base currency
        """
        if current_risk is None:
            if current_positions is None:
                current_positions = []
            current_risk = self.calculate_total_risk(current_positions)
        
        max_risk = account_balance * (self.max_total_risk_percent / 100)
        return max(0, max_risk - current_risk)
    
    def update_position_risk(
        self,
        position_id: str,
        risk_amount: float,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        size: Optional[float] = None,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        risk_percent: Optional[float] = None
    ) -> None:
        """Track position risk.
        
        Args:
            position_id: Unique position identifier
            risk_amount: Dollar amount at risk
            symbol: Trading pair symbol
            side: Position side ('long' or 'short')
            size: Position size
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_percent: Risk as percentage of account
        """
        self.position_risks[position_id] = PositionRisk(
            position_id=position_id,
            symbol=symbol or '',
            side=side or '',
            size=size or 0.0,
            entry_price=entry_price or 0.0,
            stop_loss_price=stop_loss_price or 0.0,
            risk_amount=risk_amount,
            risk_percent=risk_percent or 0.0
        )
    
    def remove_position_risk(self, position_id: str) -> None:
        """Remove position from risk tracking.
        
        Args:
            position_id: Position identifier to remove
        """
        if position_id in self.position_risks:
            del self.position_risks[position_id]
    
    def update_position_pnl(
        self,
        position_id: str,
        unrealized_pnl: float
    ) -> None:
        """Update unrealized P&L for a position.
        
        Args:
            position_id: Position identifier
            unrealized_pnl: Current unrealized P&L
        """
        if position_id in self.position_risks:
            self.position_risks[position_id].unrealized_pnl = unrealized_pnl
    
    def record_trade_result(
        self,
        symbol: str,
        realized_pnl: float,
        risk_amount: float
    ) -> None:
        """Record a completed trade result.
        
        Args:
            symbol: Trading pair symbol
            realized_pnl: Realized P&L from the trade
            risk_amount: Amount risked on the trade
        """
        today = self._current_time.strftime('%Y-%m-%d') if self._current_time else datetime.now().strftime('%Y-%m-%d')
        
        if today not in self.daily_metrics:
            today_date = self._current_time if self._current_time else datetime.now()
            self.daily_metrics[today] = DailyRiskMetrics(date=today_date)
        
        metrics = self.daily_metrics[today]
        metrics.total_pnl += realized_pnl
        metrics.realized_pnl += realized_pnl
        metrics.num_trades += 1
        metrics.total_risk_taken += risk_amount
        
        if realized_pnl > 0:
            metrics.num_wins += 1
        else:
            metrics.num_losses += 1
        
        # Update drawdown
        if realized_pnl < 0:
            self._current_drawdown += abs(realized_pnl)
            metrics.max_drawdown = max(metrics.max_drawdown, self._current_drawdown)
        else:
            self._current_drawdown = 0
    
    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Get P&L for a specific date.
        
        Args:
            date: Date to get P&L for (defaults to today)
            
        Returns:
            Total P&L for the date
        """
        if date is None:
            date = self._current_time if self._current_time else datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        if date_str in self.daily_metrics:
            return self.daily_metrics[date_str].total_pnl
        return 0.0
    
    def get_weekly_pnl(self) -> float:
        """Get P&L for the current week.
        
        Returns:
            Total P&L for the current week
        """
        today = self._current_time if self._current_time else datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        
        total = 0.0
        for date_str, metrics in self.daily_metrics.items():
            if metrics.date >= start_of_week:
                total += metrics.total_pnl
        
        return total
    
    def set_correlation_groups(self, groups: Dict[str, List[str]]) -> None:
        """Set correlation groups for risk management.
        
        Args:
            groups: Dictionary mapping group names to lists of symbols
        """
        self.correlation_groups = groups
    
    def _check_correlation_exposure(
        self,
        symbol: str,
        position_value: float,
        current_positions: List[Dict[str, Any]],
        account_balance: float
    ) -> Dict[str, Any]:
        """Check if position would exceed correlated exposure limits.
        
        Args:
            symbol: Trading pair symbol
            position_value: Value of the new position
            current_positions: Current open positions
            account_balance: Account balance
            
        Returns:
            Dictionary with check results
        """
        # Find which correlation group the symbol belongs to
        symbol_group = None
        for group_name, symbols in self.correlation_groups.items():
            if symbol in symbols:
                symbol_group = group_name
                break
        
        # If symbol not in any group, no correlation limit applies
        if symbol_group is None:
            return {'allowed': True}
        
        # Calculate current exposure in the same group
        group_exposure = 0.0
        for pos in current_positions:
            pos_symbol = pos.get('symbol', '')
            if pos_symbol in self.correlation_groups[symbol_group]:
                size = abs(pos.get('size', 0))
                price = pos.get('current_price', pos.get('entry_price', 0))
                group_exposure += size * price
        
        # Add new position
        total_group_exposure = group_exposure + position_value
        max_group_exposure = account_balance * (self.max_correlated_exposure / 100)
        
        if total_group_exposure > max_group_exposure:
            return {
                'allowed': False,
                'reason': f"Position would exceed max correlated exposure for group {symbol_group} ({self.max_correlated_exposure}%)"
            }
        
        return {'allowed': True}
    
    def get_risk_summary(self, account_balance: float) -> Dict[str, Any]:
        """Get comprehensive risk summary.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Dictionary with risk summary
        """
        positions = list(self.position_risks.values())
        
        total_risk = sum(p.risk_amount for p in positions)
        total_risk_percent = (total_risk / account_balance * 100) if account_balance > 0 else 0
        
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        daily_pnl = self.get_daily_pnl()
        weekly_pnl = self.get_weekly_pnl()
        
        return {
            'account_balance': account_balance,
            'total_positions': len(positions),
            'total_risk_amount': total_risk,
            'total_risk_percent': total_risk_percent,
            'max_risk_percent': self.max_total_risk_percent,
            'remaining_risk_percent': max(0, self.max_total_risk_percent - total_risk_percent),
            'total_unrealized_pnl': total_unrealized_pnl,
            'daily_pnl': daily_pnl,
            'weekly_pnl': weekly_pnl,
            'daily_loss_limit_percent': self.daily_loss_limit_percent,
            'weekly_loss_limit_percent': self.weekly_loss_limit_percent,
            'current_drawdown': self._current_drawdown
        }
    
    def set_account_balance(self, balance: float) -> None:
        """Update tracked account balance.
        
        Args:
            balance: Current account balance
        """
        self._account_balance = balance
        if balance > self._peak_balance:
            self._peak_balance = balance
            self._current_drawdown = 0
    
    def get_win_loss_stats(self) -> Dict[str, Any]:
        """Get win/loss statistics.
        
        Returns:
            Dictionary with win/loss statistics
        """
        total_wins = sum(m.num_wins for m in self.daily_metrics.values())
        total_losses = sum(m.num_losses for m in self.daily_metrics.values())
        total_trades = total_wins + total_losses
        
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'loss_rate': (total_losses / total_trades * 100) if total_trades > 0 else 0
        }
