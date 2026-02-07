"""Portfolio Tracker module for backtesting.

Thin subclass of the live PortfolioTracker that overrides time to use
simulated backtest time instead of datetime.now().
"""

from typing import Optional
from datetime import datetime

from src.risk.portfolio_tracker import PortfolioTracker as _BasePortfolioTracker
from src.shared.risk_types import (
    PositionStatus, Position, TradeRecord, PortfolioSnapshot, RiskReport
)


class PortfolioTracker(_BasePortfolioTracker):
    """Backtest portfolio tracker with simulated time support."""

    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance=initial_balance)
        self._current_time: Optional[datetime] = None

    def set_current_time(self, current_time: datetime) -> None:
        """Set current time for backtest."""
        self._current_time = current_time

    def _get_current_time(self) -> datetime:
        """Return simulated backtest time instead of wall-clock time."""
        return self._current_time if self._current_time else datetime.now()
