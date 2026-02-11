"""Hedge management module for cryptocurrency trading bot."""

from .position_group import PositionGroup
from .hedge_executor import HedgeExecutor
from .hedge_manager import HedgeManager

from src.shared.hedge_types import (
    OriginalPosition,
    HedgePosition,
    HedgeStatus,
    HedgeRequest,
    HedgeChunk,
    HedgeExecutionResult,
    HedgeExecutorConfig,
    PRIORITY_ASSETS,
    HedgeManagerConfig,
    HedgeTriggerResult,
    HedgeCloseResult,
)

__all__ = [
    # Position Group
    'PositionGroup',
    'OriginalPosition',
    'HedgePosition',
    'HedgeStatus',
    # Hedge Executor
    'HedgeExecutor',
    'HedgeRequest',
    'HedgeChunk',
    'HedgeExecutionResult',
    'HedgeExecutorConfig',
    'PRIORITY_ASSETS',
    # Hedge Manager
    'HedgeManager',
    'HedgeManagerConfig',
    'HedgeTriggerResult',
    'HedgeCloseResult',
]

__version__ = '1.0.0'
