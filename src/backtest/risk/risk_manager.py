"""Risk Manager module for backtesting.

Thin subclass of the live RiskManager that overrides time to use
simulated backtest time instead of datetime.now().
"""

from typing import Optional
from datetime import datetime

from src.risk.risk_manager import RiskManager as _BaseRiskManager
from src.shared.risk_types import (
    RiskStatus, RiskCheckResult, PositionRisk, DailyRiskMetrics
)


class RiskManager(_BaseRiskManager):
    """Backtest risk manager with simulated time support."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self._current_time: Optional[datetime] = None

    def set_current_time(self, current_time: datetime) -> None:
        """Set current time for backtest."""
        self._current_time = current_time

    def _get_current_time(self) -> datetime:
        """Return simulated backtest time instead of wall-clock time."""
        return self._current_time if self._current_time else datetime.now()
