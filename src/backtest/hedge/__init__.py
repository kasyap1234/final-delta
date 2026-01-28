"""Backtest hedge management module.

This module provides backtest versions of hedge management classes
that work with the mock exchange and account state.
"""

from .position_group import (
    HedgeStatus,
    HedgePosition,
    OriginalPosition,
    PositionGroup
)
from .hedge_executor import (
    HedgeRequest,
    HedgeChunk,
    HedgeExecutionResult,
    HedgeExecutorConfig,
    BacktestHedgeExecutor
)
from .hedge_manager import (
    HedgeManagerConfig,
    HedgeTriggerResult,
    HedgeCloseResult,
    BacktestHedgeManager
)

__all__ = [
    'HedgeStatus',
    'HedgePosition',
    'OriginalPosition',
    'PositionGroup',
    'HedgeRequest',
    'HedgeChunk',
    'HedgeExecutionResult',
    'HedgeExecutorConfig',
    'BacktestHedgeExecutor',
    'HedgeManagerConfig',
    'HedgeTriggerResult',
    'HedgeCloseResult',
    'BacktestHedgeManager'
]
