"""Correlation module for cryptocurrency trading bot.

This module provides correlation calculation capabilities for identifying
hedge assets and analyzing relationships between cryptocurrency pairs.
"""

from .price_history import PriceHistory, PricePoint
from .correlation_calculator import CorrelationCalculator, CorrelationResult

__all__ = [
    'PriceHistory',
    'PricePoint',
    'CorrelationCalculator',
    'CorrelationResult',
]

__version__ = '1.0.0'
