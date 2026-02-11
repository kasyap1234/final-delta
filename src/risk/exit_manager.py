"""
All-Weather Exit Manager

Manages position exits using multiple factors:
1. Technical exits (trend reversal, RSI extremes, crossovers)
2. Time-based exits (regime-specific time limits)
3. Trailing stops (profit protection)
4. Regime-change exits (exit when regime becomes unfavorable)
5. Profit protection scaling (scale out at key levels)
6. Take profit ladders (partial exits at predefined R levels)
"""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from src.indicators.market_regime import get_regime_profile
from .regime_risk_router import (
    get_risk_parameters,
    calculate_take_profit_prices,
    calculate_trailing_stop_price,
    check_profit_retracement_exit,
    calculate_r_per_r,
)

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
    TAKE_PROFIT_PARTIAL = "take_profit_partial"
    # Phase 3: Invalidation exits
    INVALIDATION = "invalidation"


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

        # Track TP levels for partial exits
        self.position_tp_levels: Dict[str, List[Tuple[float, float]]] = {}  # position_id -> [(price, scale), ...]
        self.position_tp_hit: Dict[str, List[int]] = {}  # position_id -> [level indices hit]
        self.position_stop_losses: Dict[str, float] = {}  # position_id -> stop loss price
        self.position_atr: Dict[str, float] = {}  # position_id -> ATR at entry

        # Track minimum hold time (new from profitability improvement plan)
        self.position_entry_candle_count: Dict[str, int] = {}  # position_id -> candle count at entry
        self.current_candle_count: int = 0  # Global candle counter

        # V2 Fix B2: Track breakeven progression after TP1
        self.position_breakeven_set: Dict[str, bool] = {}  # position_id -> whether breakeven has been set
        self.position_regime_value: Dict[str, str] = {}  # position_id -> regime value for breakeven logic

        # V3 Pivot: Track V3-specific payoff mechanics
        self.position_v3_runner_retained: Dict[str, bool] = {}  # position_id -> whether runner is retained
        self.position_v3_lockin_set: Dict[str, bool] = {}  # position_id -> whether lock-in has been set
        self.position_v3_hybrid_stop_price: Dict[str, float] = {}  # position_id -> hybrid stop price

        logger.info("AllWeatherExitManager initialized (using REGIME_PROFILES with TP ladders, min-hold logic, breakeven progression, and V3 payoff mechanics)")

    def _get_time_limit(self, regime_value: str) -> float:
        """Get time limit in hours from REGIME_PROFILES."""
        profile = get_regime_profile(regime_value)
        return float(profile["time_limit_hours"])

    def _get_trailing_params(self, regime_value: str) -> Tuple[float, float]:
        """Get trailing distance ATR multiplier and profit retracement threshold.

        Returns:
            (trailing_distance_atr_multiplier, profit_retracement_pct)
        """
        profile = get_regime_profile(regime_value)
        return (
            float(profile["trailing_distance_atr"]),
            float(profile["profit_retracement_pct"]),
        )

    def register_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: datetime,
        stop_loss_price: Optional[float] = None,
        atr: Optional[float] = None,
        regime_value: str = "unknown",
    ):
        """Register a new position for exit tracking."""
        self.position_entry_times[position_id] = entry_time
        self.position_entry_prices[position_id] = entry_price
        self.position_sides[position_id] = side

        # Store ATR and stop loss for TP calculations
        if atr is not None:
            self.position_atr[position_id] = atr
        if stop_loss_price is not None:
            self.position_stop_losses[position_id] = stop_loss_price

        # Record entry candle count for minimum hold time tracking
        self.position_entry_candle_count[position_id] = self.current_candle_count

        # V2 Fix B2: Initialize breakeven tracking
        self.position_breakeven_set[position_id] = False
        self.position_regime_value[position_id] = regime_value

        # V3 Pivot: Initialize V3-specific payoff tracking
        self.position_v3_runner_retained[position_id] = False
        self.position_v3_lockin_set[position_id] = False
        self.position_v3_hybrid_stop_price[position_id] = stop_loss_price if stop_loss_price else entry_price

        if side == "long":
            self.position_highest_prices[position_id] = entry_price
        else:
            self.position_lowest_prices[position_id] = entry_price

        # Calculate and store TP levels from regime profile
        if stop_loss_price is not None and atr is not None:
            from src.indicators.market_regime import MarketRegime
            regime_enum = MarketRegime(regime_value) if regime_value in [r.value for r in MarketRegime] else MarketRegime.UNKNOWN
            risk_params = get_risk_parameters(regime_enum)

            if risk_params.tp_levels:
                tp_prices = calculate_take_profit_prices(
                    entry_price, stop_loss_price, side, risk_params.tp_levels
                )
                self.position_tp_levels[position_id] = tp_prices
                self.position_tp_hit[position_id] = []
                logger.debug(
                    f"Registered TP levels for {position_id}: "
                    f"{[(f'{p:.2f}@{s:.0%}') for p, s in tp_prices]}"
                )

        logger.debug(f"Registered position {position_id} for exit tracking (min_hold_candles={risk_params.min_hold_candles})")

    def increment_candle_count(self):
        """Increment the global candle counter for minimum hold time tracking."""
        self.current_candle_count += 1

    def _check_minimum_hold_time(
        self,
        position_id: str,
        exit_type: ExitType,
        regime_value: str = "unknown",
    ) -> Tuple[bool, str]:
        """
        Check if minimum hold time has been satisfied.

        During minimum hold, block low-priority exits such as weak technical reversals.
        Always allow fail-safe exits:
        - hard stop loss (STOP_LOSS)
        - severe regime flip with high confidence and low suitability (REGIME_CHANGE)
        - risk breach safety exit

        Args:
            position_id: Position identifier
            exit_type: Type of exit being considered
            regime_value: Current regime value

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Get minimum hold candles from regime profile
        min_hold_candles = get_regime_profile(regime_value).get("min_hold_candles", 0)

        # If no minimum hold time, always allow
        if min_hold_candles <= 0:
            return True, "No minimum hold time requirement"

        # Check if minimum hold time has been satisfied
        entry_candle = self.position_entry_candle_count.get(position_id, 0)
        candles_held = self.current_candle_count - entry_candle

        if candles_held >= min_hold_candles:
            return True, f"Minimum hold time satisfied ({candles_held} >= {min_hold_candles} candles)"

        # Minimum hold time not yet satisfied - check if this is a fail-safe exit
        fail_safe_exit_types = {
            ExitType.STOP_LOSS,  # Hard stop loss - always allowed
            ExitType.TAKE_PROFIT,  # Take profit - always allowed
            ExitType.TAKE_PROFIT_PARTIAL,  # Partial TP - always allowed
        }

        if exit_type in fail_safe_exit_types:
            return True, f"Fail-safe exit type {exit_type.value} allowed during min-hold period"

        # For regime change exit, check if it's severe (high confidence, low suitability)
        if exit_type == ExitType.REGIME_CHANGE:
            # This is checked in _check_regime_exit with confidence >= 0.7
            # Consider it a fail-safe if confidence is very high
            return True, "Regime change exit allowed as fail-safe during min-hold period"

        # All other exit types are blocked during minimum hold time
        return False, f"Minimum hold time not satisfied ({candles_held} < {min_hold_candles} candles), blocking {exit_type.value} exit"

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

        # Extract regime value once for all sub-checks
        regime_value = "unknown"
        if regime_metrics is not None:
            regime = getattr(regime_metrics, "regime", None)
            if regime is not None:
                regime_value = regime.value if hasattr(regime, "value") else str(regime)

        exits: List[ExitSignal] = []
        blocked_exits: List[str] = []  # Track exits blocked by minimum hold time

        # Check time-based exit
        time_exit = self._check_time_exit(
            position_id, symbol, side, current_price, current_time, regime_metrics
        )
        if time_exit:
            # Check minimum hold time
            allowed, reason = self._check_minimum_hold_time(
                position_id, time_exit.exit_type, regime_value
            )
            if allowed:
                exits.append(time_exit)
            else:
                blocked_exits.append(f"{time_exit.exit_type.value}: {reason}")

        # Check TP levels (partial exits)
        tp_exit = self._check_take_profit_levels(
            position_id, symbol, side, current_price
        )
        if tp_exit:
            # Check minimum hold time (TP is always allowed as fail-safe)
            allowed, reason = self._check_minimum_hold_time(
                position_id, tp_exit.exit_type, regime_value
            )
            if allowed:
                exits.append(tp_exit)
            else:
                blocked_exits.append(f"{tp_exit.exit_type.value}: {reason}")

        # Check trailing stop (pass regime for profile-specific params)
        trailing_exit = self._check_trailing_stop(
            position_id, symbol, side, current_price, indicators, regime_value
        )
        if trailing_exit:
            # Check minimum hold time
            allowed, reason = self._check_minimum_hold_time(
                position_id, trailing_exit.exit_type, regime_value
            )
            if allowed:
                exits.append(trailing_exit)
            else:
                blocked_exits.append(f"{trailing_exit.exit_type.value}: {reason}")

        # Check regime change exit
        regime_exit = self._check_regime_exit(
            position_id, symbol, side, current_price, regime_metrics
        )
        if regime_exit:
            # Check minimum hold time (regime change is allowed as fail-safe)
            allowed, reason = self._check_minimum_hold_time(
                position_id, regime_exit.exit_type, regime_value
            )
            if allowed:
                exits.append(regime_exit)
            else:
                blocked_exits.append(f"{regime_exit.exit_type.value}: {reason}")

        # Check profit protection
        profit_exit = self._check_profit_protection(
            position_id, symbol, side, current_price, position.get("take_profit")
        )
        if profit_exit:
            # Check minimum hold time
            allowed, reason = self._check_minimum_hold_time(
                position_id, profit_exit.exit_type, regime_value
            )
            if allowed:
                exits.append(profit_exit)
            else:
                blocked_exits.append(f"{profit_exit.exit_type.value}: {reason}")

        # Check invalidation exit (Phase 3: payoff asymmetry redesign)
        invalidation_exit = self._check_invalidation_exit(
            position_id, symbol, side, current_price, indicators, regime_value
        )
        if invalidation_exit:
            # Invalidation exits are always allowed (fail-safe)
            exits.append(invalidation_exit)

        # Return highest priority exit if any found
        if exits:
            highest_priority_exit = max(exits, key=lambda x: x.priority)
            logger.info(
                f"Exit signal for {symbol}: {highest_priority_exit.exit_type.value} "
                f"(priority={highest_priority_exit.priority}, reason={highest_priority_exit.reason})"
            )
            return highest_priority_exit

        return None

    def _check_take_profit_levels(
        self,
        position_id: str,
        symbol: str,
        side: str,
        current_price: float,
    ) -> Optional[ExitSignal]:
        """Check if any TP levels have been hit for partial exit.
        
        V2 Fix B2: Add breakeven progression after TP1 is hit for trend regimes.
        
        Args:
            position_id: Position identifier
            symbol: Trading symbol
            side: 'long' or 'short'
            current_price: Current market price
        
        Returns:
            ExitSignal if a TP level is hit, None otherwise
        """
        if position_id not in self.position_tp_levels:
            return None
        
        tp_levels = self.position_tp_levels[position_id]
        tp_hit = self.position_tp_hit.get(position_id, [])
        
        for i, (tp_price, scale_percent) in enumerate(tp_levels):
            if i in tp_hit:
                continue  # Already hit this level
            
            # Check if TP level is hit
            hit = False
            if side == "long" and current_price >= tp_price:
                hit = True
            elif side == "short" and current_price <= tp_price:
                hit = True
            
            if hit:
                tp_hit.append(i)
                self.position_tp_hit[position_id] = tp_hit
                
                # V2 Fix B2: Breakeven progression after TP1 for trend regimes
                if i == 0 and not self.position_breakeven_set.get(position_id, False):
                    regime_value = self.position_regime_value.get(position_id, "unknown")
                    profile = get_regime_profile(regime_value)
                    
                    # Check if breakeven progression is enabled for this regime
                    if profile.get("breakeven_after_tp1", False):
                        entry_price = self.position_entry_prices.get(position_id)
                        atr = self.position_atr.get(position_id)
                        
                        if entry_price and atr:
                            # Calculate breakeven price with buffer
                            breakeven_buffer_atr = profile.get("breakeven_buffer_atr", 0.3)
                            breakeven_buffer = atr * breakeven_buffer_atr
                            
                            if side == "long":
                                breakeven_price = entry_price + breakeven_buffer
                            else:
                                breakeven_price = entry_price - breakeven_buffer
                            
                            # Update stop loss to breakeven
                            self.position_stop_losses[position_id] = breakeven_price
                            self.position_breakeven_set[position_id] = True
                            
                            logger.info(
                                f"V2 Fix B2: Breakeven set for {position_id} after TP1: "
                                f"stop moved to {breakeven_price:.2f} (buffer {breakeven_buffer_atr} ATR)"
                            )
                
                # V3 Pivot: V3-specific payoff mechanics after TP1
                if i == 0:
                    regime_value = self.position_regime_value.get(position_id, "unknown")
                    profile = get_regime_profile(regime_value)
                    
                    # V3: Check if V3 payoff mechanics are enabled for this regime
                    if profile.get("v3_payoff_enabled", False):
                        entry_price = self.position_entry_prices.get(position_id)
                        atr = self.position_atr.get(position_id)
                        
                        if entry_price and atr:
                            # V3: Hybrid stop with structure anchor and bounded floor/cap
                            # Calculate hybrid stop price using structure anchor (entry price) and ATR
                            v3_hybrid_stop_atr_mult = profile.get("v3_hybrid_stop_atr_mult", 1.5)
                            v3_stop_pct_floor = profile.get("v3_stop_pct_floor", 0.01)
                            v3_stop_pct_cap = profile.get("v3_stop_pct_cap", 0.05)
                            
                            # Calculate hybrid stop distance
                            hybrid_stop_distance = atr * v3_hybrid_stop_atr_mult
                            
                            # Calculate hybrid stop price
                            if side == "long":
                                hybrid_stop_price = entry_price - hybrid_stop_distance
                                # Apply floor/cap bounds
                                hybrid_stop_price = max(
                                    entry_price * (1 - v3_stop_pct_cap),
                                    min(entry_price * (1 - v3_stop_pct_floor), hybrid_stop_price)
                                )
                            else:
                                hybrid_stop_price = entry_price + hybrid_stop_distance
                                # Apply floor/cap bounds
                                hybrid_stop_price = min(
                                    entry_price * (1 + v3_stop_pct_cap),
                                    max(entry_price * (1 + v3_stop_pct_floor), hybrid_stop_price)
                                )
                            
                            # Update hybrid stop price
                            self.position_v3_hybrid_stop_price[position_id] = hybrid_stop_price
                            
                            logger.info(
                                f"V3 Pivot: Hybrid stop set for {position_id} after TP1: "
                                f"stop moved to {hybrid_stop_price:.2f} (ATR mult {v3_hybrid_stop_atr_mult}, "
                                f"floor {v3_stop_pct_floor:.1%}, cap {v3_stop_pct_cap:.1%})"
                            )
                            
                            # V3: Lock-in progression after TP1
                            if not self.position_v3_lockin_set.get(position_id, False):
                                v3_lockin_buffer_atr = profile.get("v3_lockin_buffer_atr", 0.5)
                                lockin_buffer = atr * v3_lockin_buffer_atr
                                
                                if side == "long":
                                    lockin_price = entry_price + lockin_buffer
                                else:
                                    lockin_price = entry_price - lockin_buffer
                                
                                # Update stop loss to lock-in price
                                self.position_stop_losses[position_id] = lockin_price
                                self.position_v3_lockin_set[position_id] = True
                                
                                logger.info(
                                    f"V3 Pivot: Lock-in set for {position_id} after TP1: "
                                    f"stop moved to {lockin_price:.2f} (buffer {v3_lockin_buffer_atr} ATR)"
                                )
                
                # V3 Pivot: Runner retention after TP2
                if i == 1:
                    regime_value = self.position_regime_value.get(position_id, "unknown")
                    profile = get_regime_profile(regime_value)
                    
                    # V3: Check if runner retention is enabled for this regime
                    if profile.get("v3_runner_retention_enabled", False):
                        # Mark runner as retained
                        self.position_v3_runner_retained[position_id] = True
                        
                        logger.info(
                            f"V3 Pivot: Runner retained for {position_id} after TP2"
                        )
                
                return ExitSignal(
                    symbol=symbol,
                    position_id=position_id,
                    exit_type=ExitType.TAKE_PROFIT_PARTIAL,
                    reason=f"TP{i+1} hit at {tp_price:.2f} (scale {scale_percent:.0%})",
                    priority=2,
                    timestamp=datetime.now(),
                    price=current_price,
                    scale_percent=scale_percent,
                )
        
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

        # Get regime-specific time limit from REGIME_PROFILES
        regime_value = "unknown"
        if regime_metrics is not None:
            regime = getattr(regime_metrics, "regime", None)
            if regime is not None:
                regime_value = regime.value if hasattr(regime, "value") else str(regime)

        time_limit = self._get_time_limit(regime_value)

        if hours_held > time_limit:
            # Check if trade has made minimum progress
            entry_price = self.position_entry_prices.get(position_id, current_price)
            stop_loss = self.position_stop_losses.get(position_id)
            
            # Calculate R progress if we have stop loss
            if stop_loss is not None:
                current_r = calculate_r_per_r(current_price, entry_price, stop_loss, side)
                min_progress_r = get_regime_profile(regime_value).get("min_progress_r", 0.5)
                
                if current_r < min_progress_r:
                    return ExitSignal(
                        symbol=symbol,
                        position_id=position_id,
                        exit_type=ExitType.TIME,
                        reason=f"Time limit exceeded ({hours_held:.1f}h, {current_r:.2f}R < {min_progress_r:.1f}R min progress)",
                        priority=3,
                        timestamp=current_time,
                        price=current_price,
                    )
            else:
                # Fallback to PnL percentage check
                if side == "long":
                    unrealized_pnl_pct = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl_pct = (entry_price - current_price) / entry_price

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
        regime_value: str = "unknown",
    ) -> Optional[ExitSignal]:
        """Check trailing stop conditions using regime-specific parameters."""
        entry_price = self.position_entry_prices.get(position_id)
        if not entry_price:
            return None

        atr = getattr(indicators, "atr", None)
        if atr is None or atr <= 0:
            atr = entry_price * 0.02  # Default 2% if no ATR

        # Get regime-specific trailing params from REGIME_PROFILES
        trailing_atr_mult, profit_retracement_threshold = self._get_trailing_params(regime_value)
        trailing_distance = atr * trailing_atr_mult

        if side == "long":
            highest_price = self.position_highest_prices.get(position_id, entry_price)
            
            # Check profit retracement using regime risk router
            if check_profit_retracement_exit(
                current_price, entry_price, highest_price, entry_price,
                profit_retracement_threshold, side
            ):
                peak_profit = highest_price - entry_price
                current_profit = current_price - entry_price
                retracement = peak_profit - current_profit
                retracement_pct = retracement / peak_profit if peak_profit > 0 else 0
                
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
            
            # Check profit retracement using regime risk router
            if check_profit_retracement_exit(
                current_price, entry_price, entry_price, lowest_price,
                profit_retracement_threshold, side
            ):
                peak_profit = entry_price - lowest_price
                current_profit = entry_price - current_price
                retracement = peak_profit - current_profit
                retracement_pct = retracement / peak_profit if peak_profit > 0 else 0
                
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
        """Check if we should scale out to protect profits.

        DISABLED: Let winners run to full take-profit or trailing stop.
        Premature scaling was cutting 75% of position before target.
        """
        return None

    def _check_invalidation_exit(
        self,
        position_id: str,
        symbol: str,
        side: str,
        current_price: float,
        indicators: Any,
        regime_value: str = "unknown",
    ) -> Optional[ExitSignal]:
        """
        Check invalidation exit conditions (Phase 3: payoff asymmetry redesign).
        
        Invalidation exits:
        - Trend invalidation: structure break and momentum decay
        - Range invalidation: confirmed range break beyond tolerance
        - Breakout invalidation: failed breakout with return to prior value zone
        
        Args:
            position_id: Position identifier
            symbol: Trading symbol
            side: 'long' or 'short'
            current_price: Current market price
            indicators: IndicatorValues with technical indicators
            regime_value: Current regime value
        
        Returns:
            ExitSignal if invalidation condition met, None otherwise
        """
        entry_price = self.position_entry_prices.get(position_id)
        if not entry_price:
            return None
        
        # Get regime-specific invalidation rules
        profile = get_regime_profile(regime_value)
        
        # Trend invalidation: structure break and momentum decay
        if regime_value in ("trending_up", "trending_down"):
            # Check for trend structure break
            if indicators.ema_9 and indicators.ema_50:
                ema_crossover = False
                
                if side == "long" and regime_value == "trending_up":
                    # Long in uptrend: invalidation if 9-EMA crosses below 50-EMA
                    if indicators.ema_9 < indicators.ema_50:
                        ema_crossover = True
                elif side == "short" and regime_value == "trending_down":
                    # Short in downtrend: invalidation if 9-EMA crosses above 50-EMA
                    if indicators.ema_9 > indicators.ema_50:
                        ema_crossover = True
                
                if ema_crossover:
                    # Check for momentum decay (ADX dropping)
                    momentum_decay = False
                    if indicators.adx is not None and indicators.adx < 20.0:
                        momentum_decay = True
                    
                    if momentum_decay:
                        return ExitSignal(
                            symbol=symbol,
                            position_id=position_id,
                            exit_type=ExitType.INVALIDATION,
                            reason=f"Trend invalidation: EMA crossover and momentum decay (ADX={indicators.adx:.1f})",
                            priority=1,  # High priority - exit immediately
                            timestamp=datetime.now(),
                            price=current_price,
                        )
        
        # Range invalidation: confirmed range break beyond tolerance
        elif regime_value == "ranging":
            # Check if price has broken out of range
            tolerance = 0.01  # 1% tolerance for range break
            
            if side == "long":
                # Long mean reversion: invalidation if price breaks above resistance
                if indicators.r1 is not None and current_price > indicators.r1 * (1 + tolerance):
                    return ExitSignal(
                        symbol=symbol,
                        position_id=position_id,
                        exit_type=ExitType.INVALIDATION,
                        reason=f"Range invalidation: price broke above resistance {indicators.r1:.2f}",
                        priority=1,
                        timestamp=datetime.now(),
                        price=current_price,
                    )
            else:  # short
                # Short mean reversion: invalidation if price breaks below support
                if indicators.s1 is not None and current_price < indicators.s1 * (1 - tolerance):
                    return ExitSignal(
                        symbol=symbol,
                        position_id=position_id,
                        exit_type=ExitType.INVALIDATION,
                        reason=f"Range invalidation: price broke below support {indicators.s1:.2f}",
                        priority=1,
                        timestamp=datetime.now(),
                        price=current_price,
                    )
        
        # Breakout invalidation: failed breakout with return to prior value zone
        elif regime_value == "volatile":
            # Check if breakout has failed (price returned to prior range)
            prior_range_pct = 0.02  # 2% prior value zone
            
            if side == "long":
                # Long breakout: invalidation if price returns below entry by prior_range_pct
                if current_price < entry_price * (1 - prior_range_pct):
                    return ExitSignal(
                        symbol=symbol,
                        position_id=position_id,
                        exit_type=ExitType.INVALIDATION,
                        reason=f"Breakout invalidation: failed breakout, price returned to prior zone",
                        priority=1,
                        timestamp=datetime.now(),
                        price=current_price,
                    )
            else:  # short
                # Short breakout: invalidation if price returns above entry by prior_range_pct
                if current_price > entry_price * (1 + prior_range_pct):
                    return ExitSignal(
                        symbol=symbol,
                        position_id=position_id,
                        exit_type=ExitType.INVALIDATION,
                        reason=f"Breakout invalidation: failed breakout, price returned to prior zone",
                        priority=1,
                        timestamp=datetime.now(),
                        price=current_price,
                    )
        
        return None

    def close_position(self, position_id: str):
        """Remove position from tracking when closed."""
        self.position_entry_times.pop(position_id, None)
        self.position_highest_prices.pop(position_id, None)
        self.position_lowest_prices.pop(position_id, None)
        self.position_entry_prices.pop(position_id, None)
        self.position_sides.pop(position_id, None)
        self.position_tp_levels.pop(position_id, None)
        self.position_tp_hit.pop(position_id, None)
        self.position_stop_losses.pop(position_id, None)
        self.position_atr.pop(position_id, None)
        self.position_entry_candle_count.pop(position_id, None)
        # V2 Fix B2: Clean up breakeven tracking
        self.position_breakeven_set.pop(position_id, None)
        self.position_regime_value.pop(position_id, None)

    def get_position_hold_time(self, position_id: str, current_time: datetime) -> float:
        """Get the number of hours a position has been held."""
        entry_time = self.position_entry_times.get(position_id)
        if not entry_time:
            return 0.0
        return (current_time - entry_time).total_seconds() / 3600
