"""Position group module for backtest hedge management.

This module provides the PositionGroup class for managing a group of related
positions - the original position and all its hedge positions - with combined
P&L tracking and status management for backtesting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HedgeStatus(Enum):
    """Status of the hedge group."""
    NO_HEDGE = "no_hedge"
    HEDGE_ACTIVE = "hedge_active"
    HEDGE_PROFIT = "hedge_profit"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class HedgePosition:
    """Represents a single hedge position in backtest."""
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    status: str = "open"
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
        """Check if position is long."""
        return self.side == 'long'

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == 'short'

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.status == 'open'

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl > 0


@dataclass
class OriginalPosition:
    """Represents the original position that may need hedging."""
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "open"

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
        """Check if position is long."""
        return self.side == 'long'

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == 'short'

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.status == 'open'


class PositionGroup:
    """Manages a group of related positions - original and hedges.

    This class tracks the original position and all its hedge positions,
    calculates combined P&L, and determines when positions should be closed.

    Attributes:
        group_id: Unique identifier for the position group
        original: The original position
        hedges: Dictionary of hedge positions (hedge_id -> HedgePosition)
        status: Current status of the group
        created_at: When the group was created
        updated_at: When the group was last updated
        target_profit_ratio: Risk:reward ratio for taking hedge profit (default 2:1)
    """

    def __init__(
        self,
        original_position: OriginalPosition,
        target_profit_ratio: float = 2.0,
        hedge_trigger_threshold: float = 0.5
    ):
        """Initialize a position group.

        Args:
            original_position: The original position to track
            target_profit_ratio: R:R ratio for hedge profit taking (default 2:1)
            hedge_trigger_threshold: Loss % of SL to trigger hedge (default 0.5 = 50%)
        """
        self.group_id = original_position.id
        self.original = original_position
        self.hedges: Dict[str, HedgePosition] = {}
        self.status = HedgeStatus.NO_HEDGE
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.target_profit_ratio = target_profit_ratio
        self.hedge_trigger_threshold = hedge_trigger_threshold

        self.next_hedge_trigger_level = hedge_trigger_threshold

        logger.info(f"PositionGroup created for {original_position.symbol} position {original_position.id}")

    def add_hedge(self, hedge: HedgePosition) -> None:
        """Add a hedge position to the group."""
        self.hedges[hedge.id] = hedge
        self.original.hedge_count += 1
        self.status = HedgeStatus.HEDGE_ACTIVE
        self.updated_at = datetime.utcnow()

        self.next_hedge_trigger_level += self.hedge_trigger_threshold

        logger.info(
            f"Added hedge {hedge.id} ({hedge.symbol}) to group {self.group_id}. "
            f"Total hedges: {self.original.hedge_count}"
        )

    def remove_hedge(self, hedge_id: str) -> Optional[HedgePosition]:
        """Remove a hedge position from the group."""
        if hedge_id in self.hedges:
            hedge = self.hedges.pop(hedge_id)
            self.updated_at = datetime.utcnow()

            if not self.hedges:
                self.status = HedgeStatus.NO_HEDGE

            logger.info(f"Removed hedge {hedge_id} from group {self.group_id}")
            return hedge
        return None

    def update_prices(self, symbol_prices: Dict[str, float]) -> Dict[str, float]:
        """Update all position prices and recalculate P&L."""
        pnls = {}

        if self.original.symbol in symbol_prices:
            pnl = self.original.update_pnl(symbol_prices[self.original.symbol])
            pnls[self.original.id] = pnl

        for hedge_id, hedge in self.hedges.items():
            if hedge.symbol in symbol_prices:
                pnl = hedge.update_pnl(symbol_prices[hedge.symbol])
                pnls[hedge_id] = pnl

        self.updated_at = datetime.utcnow()
        return pnls

    def calculate_total_pnl(self) -> float:
        """Calculate total P&L across all positions in the group."""
        total = self.original.realized_pnl + self.original.unrealized_pnl

        for hedge in self.hedges.values():
            total += hedge.realized_pnl + hedge.unrealized_pnl

        return total

    def calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        total = self.original.unrealized_pnl

        for hedge in self.hedges.values():
            total += hedge.unrealized_pnl

        return total

    def calculate_realized_pnl(self) -> float:
        """Calculate total realized P&L."""
        total = self.original.realized_pnl

        for hedge in self.hedges.values():
            total += hedge.realized_pnl

        return total

    def should_trigger_hedge(self) -> bool:
        """Check if a new hedge should be triggered."""
        if not self.original.is_open:
            return False

        loss_percentage = self.original.get_loss_percentage_of_sl()

        return loss_percentage >= self.next_hedge_trigger_level

    def get_hedge_trigger_level(self) -> int:
        """Get the current hedge trigger level number."""
        return self.original.hedge_count + 1

    def get_hedge_trigger_loss_amount(self) -> float:
        """Calculate the loss amount that triggers the next hedge."""
        sl_distance = self.original.get_stop_loss_distance()
        trigger_price_distance = sl_distance * self.next_hedge_trigger_level

        return trigger_price_distance * self.original.size

    def check_hedge_profit_targets(self) -> List[HedgePosition]:
        """Check which hedge positions have hit their profit targets."""
        profitable_hedges = []

        for hedge in self.hedges.values():
            if not hedge.is_open:
                continue

            if hedge.stop_loss:
                risk_amount = abs(hedge.entry_price - hedge.stop_loss) * hedge.size
            else:
                risk_amount = self.original.get_stop_loss_distance() * hedge.size

            if hedge.unrealized_pnl >= risk_amount * self.target_profit_ratio:
                profitable_hedges.append(hedge)
                logger.info(
                    f"Hedge {hedge.id} hit profit target: "
                    f"P&L=${hedge.unrealized_pnl:.2f}, Risk=${risk_amount:.2f}, "
                    f"R:R={self.target_profit_ratio}:1"
                )

        return profitable_hedges

    def should_close_all(self) -> bool:
        """Determine if all positions should be closed."""
        total_pnl = self.calculate_total_pnl()
        return total_pnl >= 0

    def get_hedge_direction(self) -> str:
        """Get the direction for a new hedge position."""
        return 'short' if self.original.is_long else 'long'

    def get_open_hedges(self) -> List[HedgePosition]:
        """Get all open hedge positions."""
        return [h for h in self.hedges.values() if h.is_open]

    def get_all_positions(self) -> List[Any]:
        """Get all positions including original and hedges."""
        positions = [self.original]
        positions.extend(self.hedges.values())
        return positions

    def close_group(self) -> Dict[str, Any]:
        """Mark the entire group as closing/closed."""
        self.status = HedgeStatus.CLOSED
        self.updated_at = datetime.utcnow()

        summary = {
            'group_id': self.group_id,
            'original_symbol': self.original.symbol,
            'total_pnl': self.calculate_total_pnl(),
            'realized_pnl': self.calculate_realized_pnl(),
            'unrealized_pnl': self.calculate_unrealized_pnl(),
            'hedge_count': self.original.hedge_count,
            'duration_seconds': (datetime.utcnow() - self.created_at).total_seconds()
        }

        logger.info(f"PositionGroup {self.group_id} closed. Final P&L: ${summary['total_pnl']:.2f}")
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert group to dictionary representation."""
        return {
            'group_id': self.group_id,
            'status': self.status.value,
            'original': {
                'id': self.original.id,
                'symbol': self.original.symbol,
                'side': self.original.side,
                'size': self.original.size,
                'entry_price': self.original.entry_price,
                'current_price': self.original.current_price,
                'unrealized_pnl': self.original.unrealized_pnl,
                'realized_pnl': self.original.realized_pnl,
                'hedge_count': self.original.hedge_count,
                'status': self.original.status
            },
            'hedges': {
                hid: {
                    'id': h.id,
                    'symbol': h.symbol,
                    'side': h.side,
                    'size': h.size,
                    'entry_price': h.entry_price,
                    'current_price': h.current_price,
                    'unrealized_pnl': h.unrealized_pnl,
                    'status': h.status
                }
                for hid, h in self.hedges.items()
            },
            'total_pnl': self.calculate_total_pnl(),
            'next_hedge_trigger': self.next_hedge_trigger_level,
            'should_close_all': self.should_close_all(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
