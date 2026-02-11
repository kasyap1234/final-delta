"""
Indicators Package

This package provides technical indicators and signal detection
for the cryptocurrency trading bot.
"""

from .technical_indicators import (
    calculate_ema,
    calculate_ema_crossover,
    calculate_rsi,
    calculate_atr,
    calculate_pivot_points,
    calculate_pivot_points_from_ohlcv,
    is_near_resistance,
    is_near_support,
    detect_rsi_divergence,
    get_trend_direction,
    calculate_all_emas,
    CrossoverType,
)

from .indicator_manager import (
    IndicatorManager,
    IndicatorValues,
)

from .signal_detector import (
    SignalDetector,
    Signal,
    SignalType,
)

from .enhanced_signal_detector import (
    EnhancedSignalDetector,
)

from .market_regime import (
    MarketRegimeDetector,
    MarketRegime,
    RegimeMetrics,
)

__all__ = [
    # Technical Indicators
    "calculate_ema",
    "calculate_ema_crossover",
    "calculate_rsi",
    "calculate_atr",
    "calculate_pivot_points",
    "calculate_pivot_points_from_ohlcv",
    "is_near_resistance",
    "is_near_support",
    "detect_rsi_divergence",
    "get_trend_direction",
    "calculate_all_emas",
    "CrossoverType",
    # Indicator Manager
    "IndicatorManager",
    "IndicatorValues",
    # Signal Detectors
    "SignalDetector",
    "EnhancedSignalDetector",
    "Signal",
    "SignalType",
    # Market Regime
    "MarketRegimeDetector",
    "MarketRegime",
    "RegimeMetrics",
]
