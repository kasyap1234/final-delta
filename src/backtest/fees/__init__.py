"""
Fee calculation module for backtesting.

This module provides realistic fee structures with maker/taker tiers,
funding rates, and exchange-specific fee schedules.
"""

from .fee_calculator import (
    FeeCalculator,
    FeeSchedule,
    FeeRecord,
    FeeType,
    OrderType,
    VolumeTier,
    FundingRateConfig,
    FundingRateModel,
    FeeSchedulePresets
)

__all__ = [
    'FeeCalculator',
    'FeeSchedule',
    'FeeRecord',
    'FeeType',
    'OrderType',
    'VolumeTier',
    'FundingRateConfig',
    'FundingRateModel',
    'FeeSchedulePresets'
]
