"""State management module for backtesting.

This module provides state management components for backtesting including:
- StateManager: Position state transitions, order state management, portfolio state tracking
"""

from .state_manager import (
    StateManager,
    BotState,
    PositionState,
    OrderState,
    PositionStatus,
    OrderStatus,
    OrderSide,
    OrderType
)

__all__ = [
    'StateManager',
    'BotState',
    'PositionState',
    'OrderState',
    'PositionStatus',
    'OrderStatus',
    'OrderSide',
    'OrderType'
]
