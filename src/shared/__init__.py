"""Shared types and dataclasses used by both live and backtest modules."""

from .risk_types import (
    PositionStatus,
    Position,
    TradeRecord,
    PortfolioSnapshot,
    RiskReport,
    RiskStatus,
    RiskCheckResult,
    PositionRisk,
    DailyRiskMetrics,
    PositionType,
    PositionSizeResult,
    StopLossResult,
    TakeProfitResult,
)

from .hedge_types import (
    HedgeStatus,
    HedgePosition,
    OriginalPosition,
    PRIORITY_ASSETS,
    HedgeRequest,
    HedgeChunk,
    HedgeExecutionResult,
    HedgeExecutorConfig,
    HedgeManagerConfig,
    HedgeTriggerResult,
    HedgeCloseResult,
)

__all__ = [
    # Risk types
    'PositionStatus', 'Position', 'TradeRecord', 'PortfolioSnapshot', 'RiskReport',
    'RiskStatus', 'RiskCheckResult', 'PositionRisk', 'DailyRiskMetrics',
    'PositionType', 'PositionSizeResult', 'StopLossResult', 'TakeProfitResult',
    # Hedge types
    'HedgeStatus', 'HedgePosition', 'OriginalPosition', 'PRIORITY_ASSETS',
    'HedgeRequest', 'HedgeChunk', 'HedgeExecutionResult', 'HedgeExecutorConfig',
    'HedgeManagerConfig', 'HedgeTriggerResult', 'HedgeCloseResult',
]
