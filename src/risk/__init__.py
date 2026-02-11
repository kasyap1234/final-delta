"""Risk management module for cryptocurrency trading bot."""

from .position_sizer import PositionSizer
from .risk_manager import RiskManager
from .portfolio_tracker import PortfolioTracker

from src.shared.risk_types import (
    PositionSizeResult,
    StopLossResult,
    TakeProfitResult,
    PositionType,
    RiskCheckResult,
    PositionRisk,
    DailyRiskMetrics,
    RiskStatus,
    Position,
    TradeRecord,
    PortfolioSnapshot,
    RiskReport,
    PositionStatus,
)

__all__ = [
    # Position Sizer
    'PositionSizer',
    'PositionSizeResult',
    'StopLossResult',
    'TakeProfitResult',
    'PositionType',
    # Risk Manager
    'RiskManager',
    'RiskCheckResult',
    'PositionRisk',
    'DailyRiskMetrics',
    'RiskStatus',
    # Portfolio Tracker
    'PortfolioTracker',
    'Position',
    'TradeRecord',
    'PortfolioSnapshot',
    'RiskReport',
    'PositionStatus',
]

__version__ = '1.0.0'
