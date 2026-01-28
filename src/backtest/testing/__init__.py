"""
Backtest testing module for differential testing and validation.

This module provides tools for comparing live trading results with
backtest results to ensure backtest fidelity.
"""

from .comparison_report import (
    ComparisonReport,
    SignalComparison,
    OrderComparison,
    PositionComparison,
    PnLComparison,
    RiskComparison,
    TimingComparison,
    StateTransitionComparison,
    DivergenceSeverity,
    ComparisonCategory,
    DivergenceMetric,
    ToleranceConfig
)

from .mock_live_runner import (
    LiveDataCapture,
    MockLiveRunner,
    CapturedSignal,
    CapturedOrder,
    CapturedPosition,
    CapturedRiskCheck,
    CapturedState,
    CapturedHedge
)

from .differential_tester import (
    DifferentialTester
)

__all__ = [
    # Comparison report classes
    'ComparisonReport',
    'SignalComparison',
    'OrderComparison',
    'PositionComparison',
    'PnLComparison',
    'RiskComparison',
    'TimingComparison',
    'StateTransitionComparison',
    'DivergenceSeverity',
    'ComparisonCategory',
    'DivergenceMetric',
    'ToleranceConfig',
    
    # Mock live runner classes
    'LiveDataCapture',
    'MockLiveRunner',
    'CapturedSignal',
    'CapturedOrder',
    'CapturedPosition',
    'CapturedRiskCheck',
    'CapturedState',
    'CapturedHedge',
    
    # Differential tester
    'DifferentialTester'
]
