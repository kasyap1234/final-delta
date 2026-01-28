"""
Bot package for the cryptocurrency trading bot.

This package provides the main orchestrator components:
- TradingBot: Main trading bot that coordinates all modules
- StateManager: Manages bot state persistence and recovery
- CircuitBreaker: Circuit breaker for handling repeated failures
"""

from .state_manager import StateManager, BotState
from .trading_bot import TradingBot, CircuitBreaker, TradingBotConfig

__all__ = [
    'TradingBot',
    'StateManager',
    'BotState',
    'CircuitBreaker',
    'TradingBotConfig',
]
