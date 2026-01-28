"""
Market simulation module for backtesting.

This module provides realistic market simulation components including:
- Latency modeling for network and exchange delays
- Order book depth simulation
- Market condition modeling
"""

from src.backtest.market.latency_model import (
    LatencyModel,
    VariableLatencyModel,
    LatencyConfig,
    LatencyEvent,
    LatencyType,
    LatencyDistribution,
    ExchangeQueueState,
)

from src.backtest.market.order_book import (
    SimulatedOrderBook,
    MultiSymbolOrderBook,
    OrderBookConfig,
    OrderBookSnapshot,
    OrderBookSide,
    PriceLevel,
)

__all__ = [
    # Latency model
    'LatencyModel',
    'VariableLatencyModel',
    'LatencyConfig',
    'LatencyEvent',
    'LatencyType',
    'LatencyDistribution',
    'ExchangeQueueState',
    # Order book
    'SimulatedOrderBook',
    'MultiSymbolOrderBook',
    'OrderBookConfig',
    'OrderBookSnapshot',
    'OrderBookSide',
    'PriceLevel',
]
