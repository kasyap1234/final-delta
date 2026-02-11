"""Position group module for backtest hedge management.

Re-exports from the live PositionGroup since backtest uses identical logic.
"""

from src.hedge.position_group import PositionGroup
from src.shared.hedge_types import HedgeStatus, HedgePosition, OriginalPosition

__all__ = ['PositionGroup', 'HedgeStatus', 'HedgePosition', 'OriginalPosition']
