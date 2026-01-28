"""Risk management module for cryptocurrency trading bot.

This module provides comprehensive risk management capabilities including:
- Position sizing based on risk parameters
- Stop-loss and take-profit calculations
- Portfolio risk tracking and limits
- Risk metrics and reporting

Example Usage:
    from risk import PositionSizer, RiskManager, PortfolioTracker
    
    # Initialize components
    sizer = PositionSizer(config={
        'default_risk_percent': 1.0,
        'default_atr_multiplier': 2.0,
        'default_risk_reward_ratio': 2.0
    })
    
    risk_manager = RiskManager(config={
        'max_total_exposure_percent': 80.0,
        'max_total_risk_percent': 5.0,
        'daily_loss_limit_percent': 3.0
    })
    
    portfolio = PortfolioTracker(initial_balance=10000.0)
    
    # Calculate position
    account_balance = 10000
    entry_price = 50000
    atr = 500
    
    stop_loss = sizer.calculate_stop_loss(entry_price, atr, 2.0, 'long')
    take_profit = sizer.calculate_take_profit(entry_price, stop_loss.stop_loss_price, 2.0, 'long')
    
    position_result = sizer.calculate_position_size(
        account_balance=account_balance,
        risk_percent=1.0,
        entry_price=entry_price,
        stop_loss_price=stop_loss.stop_loss_price,
        symbol='BTC/USD'
    )
    
    # Check if we can open position
    can_trade = risk_manager.can_open_position(
        symbol='BTC/USD',
        position_size=position_result.position_size,
        stop_loss_price=stop_loss.stop_loss_price,
        entry_price=entry_price,
        current_positions=portfolio.get_positions()
    )
    
    if can_trade.can_trade:
        portfolio.add_position(
            symbol='BTC/USD',
            side='long',
            size=position_result.position_size,
            entry_price=entry_price,
            stop_loss=stop_loss.stop_loss_price,
            take_profit=take_profit.take_profit_price,
            risk_amount=position_result.risk_amount
        )
"""

from .position_sizer import (
    PositionSizer,
    PositionSizeResult,
    StopLossResult,
    TakeProfitResult,
    PositionType
)

from .risk_manager import (
    RiskManager,
    RiskCheckResult,
    PositionRisk,
    DailyRiskMetrics,
    RiskStatus
)

from .portfolio_tracker import (
    PortfolioTracker,
    Position,
    TradeRecord,
    PortfolioSnapshot,
    RiskReport,
    PositionStatus
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