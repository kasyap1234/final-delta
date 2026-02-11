"""Shared hedge dataclasses used by both live and backtest modules."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class HedgeStatus(Enum):
    """Status of the hedge group."""
    NO_HEDGE = "no_hedge"
    HEDGE_ACTIVE = "hedge_active"
    HEDGE_PROFIT = "hedge_profit"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class HedgePosition:
    """Represents a single hedge position."""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    status: str = "open"  # 'open', 'closed', 'partially_closed'
    chunk_orders: List[str] = field(default_factory=list)

    def update_pnl(self, current_price: float) -> float:
        """Update unrealized P&L based on current price."""
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        return self.unrealized_pnl

    @property
    def is_long(self) -> bool:
        return self.side == 'long'

    @property
    def is_short(self) -> bool:
        return self.side == 'short'

    @property
    def is_open(self) -> bool:
        return self.status == 'open'

    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0


@dataclass
class OriginalPosition:
    """Represents the original position that may need hedging."""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "open"

    # Hedge tracking
    hedge_count: int = 0
    last_hedge_trigger_price: Optional[float] = None
    last_hedge_trigger_loss: float = 0.0

    def update_pnl(self, current_price: float) -> float:
        """Update unrealized P&L based on current price."""
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        return self.unrealized_pnl

    def get_stop_loss_distance(self) -> float:
        """Calculate the stop-loss distance from entry."""
        return abs(self.entry_price - self.stop_loss)

    def get_loss_percentage_of_sl(self) -> float:
        """Calculate current loss as percentage of stop-loss distance."""
        sl_distance = self.get_stop_loss_distance()
        if sl_distance == 0:
            return 0.0
        if self.side == 'long':
            price_distance = self.entry_price - self.current_price
        else:
            price_distance = self.current_price - self.entry_price
        return abs(price_distance) / sl_distance

    @property
    def is_long(self) -> bool:
        return self.side == 'long'

    @property
    def is_short(self) -> bool:
        return self.side == 'short'

    @property
    def is_open(self) -> bool:
        return self.status == 'open'


# Priority assets for hedge selection (in order of preference)
PRIORITY_ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD", "BTC/USDT", "ETH/USDT", "SOL/USDT"]


@dataclass
class HedgeRequest:
    """Request to open a hedge position."""
    original_symbol: str
    original_side: str  # 'long' or 'short'
    original_size: float
    original_entry_price: float
    original_stop_loss: float
    current_price: float
    hedge_symbol: Optional[str] = None
    hedge_size: Optional[float] = None
    num_chunks: int = 3
    priority_assets: List[str] = field(default_factory=lambda: PRIORITY_ASSETS.copy())


@dataclass
class HedgeChunk:
    """Represents a single chunk of a hedge position."""
    chunk_id: str
    symbol: str
    side: str
    size: float
    target_price: float
    order_id: Optional[str] = None
    status: str = "pending"  # pending, open, filled, failed
    filled_amount: float = 0.0
    filled_price: Optional[float] = None
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class HedgeExecutionResult:
    """Result of hedge execution."""
    success: bool
    hedge_id: str
    symbol: str
    side: str
    total_size: float
    filled_size: float
    average_price: Optional[float] = None
    chunks: List[HedgeChunk] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class HedgeExecutorConfig:
    """Configuration for HedgeExecutor."""
    num_chunks: int = 3
    chunk_delay_seconds: float = 2.0
    hedge_size_ratio: float = 0.5  # 50% of original position
    min_correlation: float = 0.5
    priority_assets: List[str] = field(default_factory=lambda: PRIORITY_ASSETS.copy())
    post_only: bool = True
    max_retries_per_chunk: int = 3
    price_adjustment_step: float = 0.001  # 0.1%


@dataclass
class HedgeManagerConfig:
    """Configuration for HedgeManager."""
    hedge_trigger_threshold: float = 0.5  # 50% of SL distance
    profit_target_ratio: float = 2.0  # 2:1 R:R
    enable_rehedging: bool = True
    max_hedges_per_position: int = 5
    auto_close_on_breakeven: bool = True
    hedge_executor_config: Optional[HedgeExecutorConfig] = None


@dataclass
class HedgeTriggerResult:
    """Result of hedge trigger check."""
    should_hedge: bool
    trigger_level: int
    current_loss_pct: float
    loss_amount: float
    trigger_threshold: float
    message: str


@dataclass
class HedgeCloseResult:
    """Result of closing a hedge position."""
    success: bool
    hedge_id: str
    realized_pnl: float
    close_price: Optional[float] = None
    error_message: Optional[str] = None
