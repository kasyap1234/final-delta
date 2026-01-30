"""
All-Weather Exit Manager

Manages position exits using multiple factors:
1. Technical exits (trend reversal, RSI extremes, crossovers)
2. Time-based exits (regime-specific time limits)
3. Trailing stops (profit protection)
4. Regime-change exits (exit when regime becomes unfavorable)
5. Profit protection scaling (scale out at key levels)
"""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExitType(Enum):
    """Types of exit signals."""

    TECHNICAL = "technical"
    TIME = "time"
    TRAILING_STOP = "trailing_stop"
    REGIME_CHANGE = "regime_change"
    PROFIT_PROTECTION = "profit_protection"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class ExitSignal:
    """Exit signal with metadata."""

    symbol: str
    position_id: str
    exit_type: ExitType
    reason: str
    priority: int  # Higher = more urgent
    timestamp: datetime
    price: float
    scale_percent: Optional[float] = None  # For partial exits


class AllWeatherExitManager:
    """
    Multi-factor exit manager for all-weather trading.

    Combines multiple exit mechanisms to protect capital and maximize profits
    across different market regimes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the exit manager."""
        self.config = config or {}

        # Track position state
        self.position_entry_times: Dict[str, datetime] = {}
        self.position_highest_prices: Dict[str, float] = {}  # For longs
        self.position_lowest_prices: Dict[str, float] = {}  # For shorts
        self.position_entry_prices: Dict[str, float] = {}
        self.position_sides: Dict[str, str] = {}

        # Time limits by regime (hours)
        self.time_limits = {
            "trending_up": 48,
            "trending_down": 48,
            "ranging": 12,
            "volatile": 6,
            "quiet": 24,
            "unknown": 24,
        }

        # Trailing stop configuration
        self.trailing_atr_multiplier = self.config.get("trailing_atr_multiplier", 2.0)
        self.profit_retracement_threshold = self.config.get(
            "profit_retracement_threshold", 0.5
        )

        logger.info("AllWeatherExitManager initialized")

    def register_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: datetime,
    ):
        """Register a new position for exit tracking."""
        self.position_entry_times[position_id] = entry_time
        self.position_entry_prices[position_id] = entry_price
        self.position_sides[position_id] = side

        if side == "long":
            self.position_highest_prices[position_id] = entry_price
        else:
            self.position_lowest_prices[position_id] = entry_price

        logger.debug(f"Registered position {position_id} for exit tracking")

    def update_position_price(self, position_id: str, current_price: float):
        """Update tracked price for a position."""
        side = self.position_sides.get(position_id)
        if side == "long":
            if position_id in self.position_highest_prices:
                self.position_highest_prices[position_id] = max(
                    self.position_highest_prices[position_id], current_price
                )
        elif side == "short":
            if position_id in self.position_lowest_prices:
                self.position_lowest_prices[position_id] = min(
                    self.position_lowest_prices[position_id], current_price
                )

    def check_all_exits(
        self,
        position: Dict[str, Any],
        indicators: Any,
        regime_metrics: Any,
        current_time: datetime,
    ) -> Optional[ExitSignal]:
        """
        Check all exit conditions and return the highest priority exit signal.

        Args:
            position: Position dict with id, symbol, side, entry_price, etc.
            indicators: IndicatorValues with technical data
            regime_metrics: RegimeMetrics with regime and confidence
            current_time: Current timestamp

        Returns:
            ExitSignal if an exit should occur, None otherwise
        """
        position_id = position.get("id")
        symbol = position.get("symbol")
        side = position.get("side")
        current_price = position.get("current_price", 0)

        if not position_id or not symbol:
            return None

        # Update tracked prices
        self.update_position_price(position_id, current_price)

        exits: List[ExitSignal] = []

        # Check time-based exit
        time_exit = self._check_time_exit(
            position_id, symbol, side, current_price, current_time, regime_metrics
        )
        if time_exit:
            exits.append(time_exit)

        # Check trailing stop
        trailing_exit = self._check_trailing_stop(
            position_id, symbol, side, current_price, indicators
        )
        if trailing_exit:
            exits.append(trailing_exit)

        # Check regime change exit
        regime_exit = self._check_regime_exit(
            position_id, symbol, side, current_price, regime_metrics
        )
        if regime_exit:
            exits.append(regime_exit)

        # Check profit protection
        profit_exit = self._check_profit_protection(
            position_id, symbol, side, current_price, position.get("take_profit")
        )
        if profit_exit:
            exits.append(profit_exit)

        # Return highest priority exit if any found
        if exits:
            highest_priority_exit = max(exits, key=lambda x: x.priority)
            logger.info(
                f"Exit signal for {symbol}: {highest_priority_exit.exit_type.value} "
                f"(priority={highest_priority_exit.priority}, reason={highest_priority_exit.reason})"
            )
            return highest_priority_exit

        return None

    def _check_time_exit(
        self,
        position_id: str,
        symbol: str,
        side: str,
        current_price: float,
        current_time: datetime,
        regime_metrics: Any,
    ) -> Optional[ExitSignal]:
        """Check if position should exit based on time held."""
        entry_time = self.position_entry_times.get(position_id)
        if not entry_time:
            return None

        hours_held = (current_time - entry_time).total_seconds() / 3600

        # Get regime-specific time limit
        regime_value = "unknown"
        if regime_metrics is not None:
            regime = getattr(regime_metrics, "regime", None)
            if regime is not None:
                regime_value = regime.value if hasattr(regime, "value") else str(regime)

        time_limit = self.time_limits.get(regime_value, 24)

        if hours_held > time_limit:
            # Check if trade is profitable enough
            entry_price = self.position_entry_prices.get(position_id, current_price)

            if side == "long":
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl_pct = (entry_price - current_price) / entry_price

            # Exit if not at least 50% toward target (assuming 2:1 R:R, target is ~2x stop distance)
            # 50% toward target = ~1x stop distance = ~1% profit (assuming 1% risk)
            if unrealized_pnl_pct < 0.01:  # Less than 1% profit
                return ExitSignal(
                    symbol=symbol,
                    position_id=position_id,
                    exit_type=ExitType.TIME,
                    reason=f"Time limit exceeded ({hours_held:.1f}h, {unrealized_pnl_pct:.2%} PnL)",
                    priority=3,
                    timestamp=current_time,
                    price=current_price,
                )

        return None

    def _check_trailing_stop(
        self,
        position_id: str,
        symbol: str,
        side: str,
        current_price: float,
        indicators: Any,
    ) -> Optional[ExitSignal]:
        """Check trailing stop conditions."""
        entry_price = self.position_entry_prices.get(position_id)
        if not entry_price:
            return None

        atr = getattr(indicators, "atr", None)
        if atr is None or atr <= 0:
            atr = entry_price * 0.02  # Default 2% if no ATR

        trailing_distance = atr * self.trailing_atr_multiplier

        if side == "long":
            highest_price = self.position_highest_prices.get(position_id, entry_price)
            peak_profit = highest_price - entry_price
            current_profit = current_price - entry_price

            # Check profit retracement
            if peak_profit > 0:
                retracement = peak_profit - current_profit
                retracement_pct = retracement / peak_profit

                if retracement_pct > self.profit_retracement_threshold:
                    return ExitSignal(
                        symbol=symbol,
                        position_id=position_id,
                        exit_type=ExitType.TRAILING_STOP,
                        reason=f"Profit retracement ({retracement_pct:.1%} of {peak_profit:.2f} peak)",
                        priority=2,
                        timestamp=datetime.now(),
                        price=current_price,
                    )

            # Check trailing stop hit
            trailing_stop_price = highest_price - trailing_distance
            if (
                current_price < trailing_stop_price
                and highest_price > entry_price * 1.01
            ):
                return ExitSignal(
                    symbol=symbol,
                    position_id=position_id,
                    exit_type=ExitType.TRAILING_STOP,
                    reason=f"Trailing stop hit ({trailing_distance:.2f} from peak)",
                    priority=1,
                    timestamp=datetime.now(),
                    price=current_price,
                )

        else:  # short
            lowest_price = self.position_lowest_prices.get(position_id, entry_price)
            peak_profit = entry_price - lowest_price
            current_profit = entry_price - current_price

            # Check profit retracement
            if peak_profit > 0:
                retracement = peak_profit - current_profit
                retracement_pct = retracement / peak_profit

                if retracement_pct > self.profit_retracement_threshold:
                    return ExitSignal(
                        symbol=symbol,
                        position_id=position_id,
                        exit_type=ExitType.TRAILING_STOP,
                        reason=f"Profit retracement ({retracement_pct:.1%} of {peak_profit:.2f} peak)",
                        priority=2,
                        timestamp=datetime.now(),
                        price=current_price,
                    )

            # Check trailing stop hit
            trailing_stop_price = lowest_price + trailing_distance
            if (
                current_price > trailing_stop_price
                and lowest_price < entry_price * 0.99
            ):
                return ExitSignal(
                    symbol=symbol,
                    position_id=position_id,
                    exit_type=ExitType.TRAILING_STOP,
                    reason=f"Trailing stop hit ({trailing_distance:.2f} from low)",
                    priority=1,
                    timestamp=datetime.now(),
                    price=current_price,
                )

        return None

    def _check_regime_exit(
        self,
        position_id: str,
        symbol: str,
        side: str,
        current_price: float,
        regime_metrics: Any,
    ) -> Optional[ExitSignal]:
        """Check if regime change warrants an exit."""
        if regime_metrics is None:
            return None

        regime = getattr(regime_metrics, "regime", None)
        confidence = getattr(regime_metrics, "confidence", 0.0)

        if regime is None or confidence < 0.7:
            return None

        regime_value = regime.value if hasattr(regime, "value") else str(regime)

        # Define unfavorable regimes by position direction
        unfavorable_regimes = {
            "long": ["trending_down", "volatile"],
            "short": ["trending_up", "volatile"],
        }

        if regime_value in unfavorable_regimes.get(side, []):
            return ExitSignal(
                symbol=symbol,
                position_id=position_id,
                exit_type=ExitType.REGIME_CHANGE,
                reason=f"Regime change to {regime_value} (confidence={confidence:.2f})",
                priority=4,
                timestamp=datetime.now(),
                price=current_price,
            )

        return None

    def _check_profit_protection(
        self,
        position_id: str,
        symbol: str,
        side: str,
        current_price: float,
        take_profit_price: Optional[float],
    ) -> Optional[ExitSignal]:
        """Check if we should scale out to protect profits."""
        if not take_profit_price:
            return None

        entry_price = self.position_entry_prices.get(position_id)
        if not entry_price:
            return None

        # Calculate progress toward target
        if side == "long":
            total_distance = take_profit_price - entry_price
            current_distance = current_price - entry_price
        else:
            total_distance = entry_price - take_profit_price
            current_distance = entry_price - current_price

        if total_distance <= 0:
            return None

        progress = current_distance / total_distance

        # Scale out at 50% and 75% of target
        if progress > 0.75:
            return ExitSignal(
                symbol=symbol,
                position_id=position_id,
                exit_type=ExitType.PROFIT_PROTECTION,
                reason="75% target reached - scale out",
                priority=5,
                timestamp=datetime.now(),
                price=current_price,
                scale_percent=50,  # Close 50% of position
            )
        elif progress > 0.50:
            return ExitSignal(
                symbol=symbol,
                position_id=position_id,
                exit_type=ExitType.PROFIT_PROTECTION,
                reason="50% target reached - scale out",
                priority=5,
                timestamp=datetime.now(),
                price=current_price,
                scale_percent=25,  # Close 25% of position
            )

        return None

    def close_position(self, position_id: str):
        """Remove position from tracking when closed."""
        self.position_entry_times.pop(position_id, None)
        self.position_highest_prices.pop(position_id, None)
        self.position_lowest_prices.pop(position_id, None)
        self.position_entry_prices.pop(position_id, None)
        self.position_sides.pop(position_id, None)

    def get_position_hold_time(self, position_id: str, current_time: datetime) -> float:
        """Get the number of hours a position has been held."""
        entry_time = self.position_entry_times.get(position_id)
        if not entry_time:
            return 0.0
        return (current_time - entry_time).total_seconds() / 3600
