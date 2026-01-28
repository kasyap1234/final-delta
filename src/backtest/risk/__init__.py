"""Risk management module for backtesting.

This module provides risk management components for backtesting including:
- RiskManager: Portfolio-level risk management with correlation-adjusted exposure
- PortfolioTracker: Real-time P&L tracking and drawdown monitoring
- PositionSizer: Risk-based position sizing with ATR-based stops
"""

from .risk_manager import RiskManager, RiskCheckResult, RiskStatus, PositionRisk, DailyRiskMetrics
from .portfolio_tracker import PortfolioTracker, Position, TradeRecord, PositionStatus, PortfolioSnapshot, RiskReport
from .position_sizer import PositionSizer, PositionSizeResult, StopLossResult, TakeProfitResult, PositionType

__all__ = [
    'RiskManager',
    'RiskCheckResult',
    'RiskStatus',
    'PositionRisk',
    'DailyRiskMetrics',
    'PortfolioTracker',
    'Position',
    'TradeRecord',
    'PositionStatus',
    'PortfolioSnapshot',
    'RiskReport',
    'PositionSizer',
    'PositionSizeResult',
    'StopLossResult',
    'TakeProfitResult',
    'PositionType'
]
