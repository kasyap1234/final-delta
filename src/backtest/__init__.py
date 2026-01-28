"""
Backtesting module for the trading bot.

This module provides a complete backtesting framework that uses
exactly the same trading logic as the live bot, with mocked
data sources and order execution.
"""

from .config import BacktestConfig
from .time_controller import TimeController
from .account_state import AccountState

__all__ = [
    'BacktestConfig',
    'TimeController',
    'AccountState',
]
