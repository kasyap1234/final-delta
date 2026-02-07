"""Backtest hedge management module."""

from .position_group import PositionGroup
from .hedge_executor import BacktestHedgeExecutor
from .hedge_manager import BacktestHedgeManager

from src.shared.hedge_types import (
    HedgeStatus,
    HedgePosition,
    OriginalPosition,
    HedgeRequest,
    HedgeChunk,
    HedgeExecutionResult,
    HedgeExecutorConfig,
    HedgeManagerConfig,
    HedgeTriggerResult,
    HedgeCloseResult,
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
    'BacktestHedgeManager',
]
