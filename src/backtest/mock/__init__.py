"""
Backtest mock module for simulating exchange and order behavior.

This module provides realistic order simulation for backtesting,
including post-only order rejection, retry logic, partial fills,
market impact modeling, and order status polling.
"""

from src.backtest.mock.order_simulator import (
    BacktestOrderSimulator,
    Order,
    OHLCV,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    SimulatorConfig,
)
from src.backtest.mock.price_impact_model import (
    PriceImpactModel,
    PriceImpact,
    ImpactConfig,
)

# Import optional modules only if dependencies are available
__all__ = [
    'BacktestOrderSimulator',
    'Order',
    'OHLCV',
    'OrderStatus',
    'OrderType',
    'OrderSide',
    'TimeInForce',
    'SimulatorConfig',
    'PriceImpactModel',
    'PriceImpact',
    'ImpactConfig',
]

# Import exchange_client only if data module dependencies are available
try:
    from src.backtest.mock.exchange_client import BacktestExchangeClient
    __all__.append('BacktestExchangeClient')
except ImportError:
    pass

# Import data_cache and stream_manager if available
try:
    from src.backtest.mock.data_cache import BacktestDataCache
    __all__.append('BacktestDataCache')
except ImportError:
    pass

try:
    from src.backtest.mock.stream_manager import BacktestStreamManager
    __all__.append('BacktestStreamManager')
except ImportError:
    pass
