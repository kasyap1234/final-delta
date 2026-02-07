"""Shared risk dataclasses used by both live and backtest modules."""

from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


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


class PositionType(str, Enum):
    """Position direction types."""
    LONG = "long"
    SHORT = "short"


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    position_size: float
    risk_amount: float
    stop_loss_distance: float
    is_valid: bool
    error_message: Optional[str] = None
    adjusted_size: Optional[float] = None


@dataclass
class StopLossResult:
    """Result of stop loss calculation."""
    stop_loss_price: float
    atr_value: float
    atr_multiplier: float
    stop_loss_distance: float
    stop_loss_percent: float


@dataclass
class TakeProfitResult:
    """Result of take profit calculation."""
    take_profit_price: float
    risk_reward_ratio: float
    potential_profit: float
    potential_profit_percent: float
    stop_loss_distance: float
