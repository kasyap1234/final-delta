"""
Database module for the cryptocurrency trading bot.

This module provides SQLite database functionality for trade journaling,
market data storage, and performance tracking.

Example Usage:
    from database import DatabaseManager, Trade, Signal
    
    # Initialize database
    db = DatabaseManager('data/trading_bot.db')
    db.initialize_database()
    
    # Save a trade
    trade = Trade(
        symbol='BTC/USD',
        side=TradeSide.BUY,
        entry_price=50000,
        quantity=0.1,
        entry_time=datetime.now()
    )
    db.save_trade(trade)
    
    # Query trades
    trades = db.get_trades(symbol='BTC/USD', status=TradeStatus.CLOSED)
    
    # Get performance summary
    performance = db.get_performance_summary()
"""

from .db_manager import DatabaseManager, DatabaseError
from .models import (
    # Models
    Trade,
    Signal,
    Position,
    Order,
    Hedge,
    MarketData,
    Correlation,
    Performance,
    BalanceHistory,
    TradeJournal,
    SystemLog,
    # Enums
    TradeSide,
    TradeStatus,
    SignalType,
    PositionSide,
    PositionStatus,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    HedgeStatus,
    CorrelationType,
    MetricType,
    LogLevel,
    # Utilities
    generate_id,
    to_json,
    from_json,
)

__all__ = [
    # Database Manager
    'DatabaseManager',
    'DatabaseError',
    # Models
    'Trade',
    'Signal',
    'Position',
    'Order',
    'Hedge',
    'MarketData',
    'Correlation',
    'Performance',
    'BalanceHistory',
    'TradeJournal',
    'SystemLog',
    # Enums
    'TradeSide',
    'TradeStatus',
    'SignalType',
    'PositionSide',
    'PositionStatus',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    'HedgeStatus',
    'CorrelationType',
    'MetricType',
    'LogLevel',
    # Utilities
    'generate_id',
    'to_json',
    'from_json',
]

__version__ = '1.0.0'