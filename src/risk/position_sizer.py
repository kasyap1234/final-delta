"""Position Sizing module for cryptocurrency trading bot.

This module provides position sizing calculations, stop-loss and take-profit
management for risk-controlled trading.
"""

from typing import Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PositionType(str, Enum):
    """Position direction types."""

    LONG = "long"
    SHORT = "short"


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    position_size: float
    risk_amount: float
    stop_loss_distance: float
    is_valid: bool
    error_message: Optional[str] = None
    adjusted_size: Optional[float] = None


@dataclass
class StopLossResult:
    """Result of stop loss calculation."""

    stop_loss_price: float
    atr_value: float
    atr_multiplier: float
    stop_loss_distance: float
    stop_loss_percent: float


@dataclass
class TakeProfitResult:
    """Result of take profit calculation."""

    take_profit_price: float
    risk_reward_ratio: float
    potential_profit: float
    potential_profit_percent: float
    stop_loss_distance: float


class PositionSizer:
    """Position sizing and risk management calculator.

    This class handles all position sizing calculations including:
    - Position size based on risk percentage
    - ATR-based stop loss calculations
    - Risk:Reward based take profit calculations
    - Position size validation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PositionSizer.

        Args:
            config: Configuration dictionary with risk management settings.
                   Expected keys:
                   - default_risk_percent: Default risk per trade (default: 1.0)
                   - default_atr_multiplier: Default ATR multiplier for SL (default: 2.0)
                   - default_risk_reward_ratio: Default R:R ratio (default: 2.0)
                   - min_position_size: Minimum position size (default: 0.001)
                   - max_position_size: Maximum position size (default: 100.0)
                   - trading_fee_percent: Trading fee percentage (default: 0.1)
        """
        self.config = config or {}
        self.default_risk_percent = self.config.get("default_risk_percent", 1.0)
        self.default_atr_multiplier = self.config.get("default_atr_multiplier", 2.0)
        self.default_risk_reward_ratio = self.config.get(
            "default_risk_reward_ratio", 2.0
        )
        self.min_position_size = self.config.get("min_position_size", 0.001)
        self.max_position_size = self.config.get("max_position_size", 100.0)
        self.trading_fee_percent = self.config.get("trading_fee_percent", 0.1)

        # Symbol-specific limits (can be overridden via config)
        self.symbol_limits: Dict[str, Dict[str, float]] = {}

    def calculate_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_price: float,
        symbol: str,
        trading_fee_percent: Optional[float] = None,
    ) -> PositionSizeResult:
        """Calculate position size based on risk parameters.

        Formula: Position Size = (Account Balance × Risk%) / (Entry Price - Stop Loss Price)

        Args:
            account_balance: Current account balance in USD
            risk_percent: Percentage of account to risk (e.g., 1.0 for 1%)
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            trading_fee_percent: Optional trading fee percentage override

        Returns:
            PositionSizeResult with calculated position size and metadata
        """
        # Validate inputs
        if account_balance <= 0:
            return PositionSizeResult(
                position_size=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                is_valid=False,
                error_message="Account balance must be positive",
            )

        if entry_price <= 0:
            return PositionSizeResult(
                position_size=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                is_valid=False,
                error_message="Entry price must be positive",
            )

        if stop_loss_price <= 0:
            return PositionSizeResult(
                position_size=0.0,
                risk_amount=0.0,
                stop_loss_distance=0.0,
                is_valid=False,
                error_message="Stop loss price must be positive",
            )

        # Calculate risk amount
        risk_amount = self.get_risk_amount(account_balance, risk_percent)

        # Calculate stop loss distance
        stop_loss_distance = abs(entry_price - stop_loss_price)

        if stop_loss_distance == 0:
            return PositionSizeResult(
                position_size=0.0,
                risk_amount=risk_amount,
                stop_loss_distance=0.0,
                is_valid=False,
                error_message="Stop loss distance cannot be zero",
            )

        # Calculate base position size
        position_size = risk_amount / stop_loss_distance

        # Account for trading fees
        fee_percent = (
            trading_fee_percent
            if trading_fee_percent is not None
            else self.trading_fee_percent
        )
        if fee_percent > 0:
            # Adjust position size to account for entry and exit fees
            total_fee_impact = 2 * (fee_percent / 100)  # Entry and exit
            position_size = position_size * (1 - total_fee_impact)

        # Validate position size
        validation_result = self.validate_position_size(position_size, symbol)

        adjusted_size = None
        if not validation_result["is_valid"]:
            if validation_result["too_small"]:
                adjusted_size = self._get_min_position_size(symbol)
            elif validation_result["too_large"]:
                adjusted_size = self._get_max_position_size(symbol)

        return PositionSizeResult(
            position_size=position_size,
            risk_amount=risk_amount,
            stop_loss_distance=stop_loss_distance,
            is_valid=validation_result["is_valid"],
            error_message=validation_result.get("error_message"),
            adjusted_size=adjusted_size,
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: Optional[float] = None,
        position_type: Union[PositionType, str] = PositionType.LONG,
    ) -> StopLossResult:
        """Calculate ATR-based stop loss price.

        Args:
            entry_price: Entry price for the position
            atr: Average True Range value
            atr_multiplier: Multiplier for ATR (default: 2.0)
            position_type: 'long' or 'short' position

        Returns:
            StopLossResult with calculated stop loss price and metadata
        """
        multiplier = (
            atr_multiplier
            if atr_multiplier is not None
            else self.default_atr_multiplier
        )

        # Convert string to enum if needed
        if isinstance(position_type, str):
            position_type = PositionType(position_type.lower())

        # Calculate stop loss distance
        stop_loss_distance = atr * multiplier

        # Calculate stop loss price based on position type
        if position_type == PositionType.LONG:
            stop_loss_price = entry_price - stop_loss_distance
        else:  # SHORT
            stop_loss_price = entry_price + stop_loss_distance

        # Calculate stop loss percentage
        stop_loss_percent = (stop_loss_distance / entry_price) * 100

        return StopLossResult(
            stop_loss_price=stop_loss_price,
            atr_value=atr,
            atr_multiplier=multiplier,
            stop_loss_distance=stop_loss_distance,
            stop_loss_percent=stop_loss_percent,
        )

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_reward_ratio: Optional[float] = None,
        position_type: Union[PositionType, str] = PositionType.LONG,
    ) -> TakeProfitResult:
        """Calculate take profit price based on risk:reward ratio.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            risk_reward_ratio: Risk:Reward ratio (default: 2.0 for 2:1)
            position_type: 'long' or 'short' position

        Returns:
            TakeProfitResult with calculated take profit price and metadata
        """
        ratio = (
            risk_reward_ratio
            if risk_reward_ratio is not None
            else self.default_risk_reward_ratio
        )

        # Convert string to enum if needed
        if isinstance(position_type, str):
            position_type = PositionType(position_type.lower())

        # Calculate stop loss distance (risk)
        stop_loss_distance = abs(entry_price - stop_loss_price)

        # Calculate take profit distance (reward)
        take_profit_distance = stop_loss_distance * ratio

        # Calculate take profit price based on position type
        if position_type == PositionType.LONG:
            take_profit_price = entry_price + take_profit_distance
        else:  # SHORT
            take_profit_price = entry_price - take_profit_distance

        # Calculate potential profit
        potential_profit = take_profit_distance
        potential_profit_percent = (potential_profit / entry_price) * 100

        return TakeProfitResult(
            take_profit_price=take_profit_price,
            risk_reward_ratio=ratio,
            potential_profit=potential_profit,
            potential_profit_percent=potential_profit_percent,
            stop_loss_distance=stop_loss_distance,
        )

    def validate_position_size(
        self,
        position_size: float,
        symbol: str,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate position size against minimum and maximum limits.

        Args:
            position_size: Calculated position size
            symbol: Trading pair symbol
            min_size: Optional minimum size override
            max_size: Optional maximum size override

        Returns:
            Dictionary with validation results
        """
        min_limit = (
            min_size if min_size is not None else self._get_min_position_size(symbol)
        )
        max_limit = (
            max_size if max_size is not None else self._get_max_position_size(symbol)
        )

        result = {
            "is_valid": True,
            "too_small": False,
            "too_large": False,
            "error_message": None,
            "position_size": position_size,
            "min_size": min_limit,
            "max_size": max_limit,
        }

        if position_size < min_limit:
            result["is_valid"] = False
            result["too_small"] = True
            result["error_message"] = (
                f"Position size {position_size} is below minimum {min_limit}"
            )
        elif position_size > max_limit:
            result["is_valid"] = False
            result["too_large"] = True
            result["error_message"] = (
                f"Position size {position_size} exceeds maximum {max_limit}"
            )

        return result

    def get_risk_amount(self, account_balance: float, risk_percent: float) -> float:
        """Calculate the dollar risk amount.

        Args:
            account_balance: Current account balance
            risk_percent: Percentage of account to risk

        Returns:
            Dollar amount to risk
        """
        return account_balance * (risk_percent / 100)

    def calculate_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        highest_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        position_type: Union[PositionType, str] = PositionType.LONG,
    ) -> Optional[float]:
        """Calculate trailing stop loss price.

        Args:
            current_price: Current market price
            entry_price: Original entry price
            highest_price: Highest price since entry (for longs) or lowest (for shorts)
            atr: Current ATR value
            atr_multiplier: ATR multiplier for trailing distance
            position_type: 'long' or 'short' position

        Returns:
            New trailing stop price or None if no update needed
        """
        # Convert string to enum if needed
        if isinstance(position_type, str):
            position_type = PositionType(position_type.lower())

        trailing_distance = atr * atr_multiplier

        if position_type == PositionType.LONG:
            # For longs, trail below the highest price
            new_stop = highest_price - trailing_distance
            # Only move stop up, never down
            current_stop = entry_price - (atr * atr_multiplier)
            if new_stop > current_stop:
                return new_stop
        else:  # SHORT
            # For shorts, trail above the lowest price
            new_stop = highest_price + trailing_distance
            # Only move stop down, never up
            current_stop = entry_price + (atr * atr_multiplier)
            if new_stop < current_stop:
                return new_stop

        return None

    def calculate_partial_take_profits(
        self,
        entry_price: float,
        stop_loss_price: float,
        position_size: float,
        levels: list = None,
    ) -> list:
        """Calculate partial take profit levels.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            position_size: Total position size
            levels: List of dicts with 'ratio' (R:R) and 'percent' (size to close)
                   Default: [{'ratio': 1.0, 'percent': 25}, {'ratio': 2.0, 'percent': 25}, {'ratio': 3.0, 'percent': 50}]

        Returns:
            List of take profit levels with prices and sizes
        """
        if levels is None:
            levels = [
                {"ratio": 1.0, "percent": 25},
                {"ratio": 2.0, "percent": 25},
                {"ratio": 3.0, "percent": 50},
            ]

        stop_loss_distance = abs(entry_price - stop_loss_price)
        results = []

        for level in levels:
            ratio = level["ratio"]
            percent = level["percent"]

            take_profit_distance = stop_loss_distance * ratio
            take_profit_price = entry_price + take_profit_distance  # For longs

            size_to_close = position_size * (percent / 100)

            results.append(
                {
                    "ratio": ratio,
                    "percent": percent,
                    "take_profit_price": take_profit_price,
                    "size_to_close": size_to_close,
                }
            )

        return results

    def _get_min_position_size(self, symbol: str) -> float:
        """Get minimum position size for a symbol."""
        if symbol in self.symbol_limits:
            return self.symbol_limits[symbol].get("min_size", self.min_position_size)
        return self.min_position_size

    def _get_max_position_size(self, symbol: str) -> float:
        """Get maximum position size for a symbol."""
        if symbol in self.symbol_limits:
            return self.symbol_limits[symbol].get("max_size", self.max_position_size)
        return self.max_position_size

    def set_symbol_limits(self, symbol: str, min_size: float, max_size: float):
        """Set position size limits for a specific symbol.

        Args:
            symbol: Trading pair symbol
            min_size: Minimum position size
            max_size: Maximum position size
        """
        self.symbol_limits[symbol] = {"min_size": min_size, "max_size": max_size}

    def calculate_kelly_criterion(
        self, win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5
    ) -> float:
        """Calculate Kelly Criterion for optimal position sizing.

        Kelly % = W - [(1 - W) / R]
        Where:
        - W = Win rate
        - R = Average win / Average loss

        Args:
            win_rate: Probability of winning (0.0 to 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount
            fraction: Kelly fraction (0.5 = Half-Kelly, 0.25 = Quarter-Kelly)

        Returns:
            Kelly percentage (0.0 to 1.0)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        if win_loss_ratio <= 0:
            return 0.0

        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fraction (Half-Kelly or Quarter-Kelly for safety)
        return max(0.0, min(kelly * fraction, 0.25))  # Cap at 25% max per trade

    def calculate_volatility_targeting_size(
        self,
        base_position_size: float,
        current_volatility: float,
        target_volatility: float = 0.15,
        max_leverage: float = 2.0,
    ) -> float:
        """
        Adjust position size to maintain target volatility.

        Position Size = Target Vol / Current Vol * Base Size

        Args:
            base_position_size: Base position size without volatility adjustment
            current_volatility: Current realized volatility (annualized)
            target_volatility: Target volatility level (default 15% annualized)
            max_leverage: Maximum leverage allowed

        Returns:
            Volatility-adjusted position size
        """
        if current_volatility <= 0:
            return base_position_size

        # Calculate adjustment factor
        vol_factor = target_volatility / current_volatility

        # Apply to base size
        adjusted_size = base_position_size * vol_factor

        # Apply leverage constraint
        return min(adjusted_size, base_position_size * max_leverage)

    def calculate_all_weather_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_price: float,
        symbol: str,
        signal_strength: float,
        regime_metrics: Optional[Any] = None,
        recent_performance: Optional[Dict[str, Any]] = None,
        current_drawdown: float = 0.0,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate position size using the all-weather multi-factor formula.

        Factors:
        1. Signal strength (0.2 to 1.0)
        2. Regime confidence (0.5 to 1.0)
        3. Regime position modifier (continuous)
        4. Recent performance decay (0.5 to 1.0)
        5. Drawdown protection (0.0 to 1.0, continuous)
        6. Kelly criterion (if performance data available)

        Args:
            account_balance: Current account balance
            risk_percent: Base risk percentage per trade
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            symbol: Trading symbol
            signal_strength: Signal strength from detector (0.0 to 1.0)
            regime_metrics: RegimeMetrics object with regime and confidence
            recent_performance: Dict with win_rate and other performance metrics
            current_drawdown: Current drawdown as decimal (e.g., 0.05 for 5%)
            win_rate: Historical win rate (optional, for Kelly)
            avg_win: Average win amount (optional, for Kelly)
            avg_loss: Average loss amount (optional, for Kelly)

        Returns:
            PositionSizeResult with calculated position size
        """
        # Calculate base position size
        base_result = self.calculate_position_size(
            account_balance=account_balance,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            symbol=symbol,
        )

        if not base_result.is_valid:
            return base_result

        position_size = base_result.position_size

        signal_factor = 0.3 + (signal_strength * 0.7)

        # Factor 2 & 3: Regime confidence and modifier
        regime_factor = 1.0
        regime_modifier = 1.0
        if regime_metrics is not None:
            # Regime confidence (0.5 to 1.0)
            regime_confidence = getattr(regime_metrics, "confidence", 0.5)
            regime_factor = 0.5 + (regime_confidence * 0.5)

            # Regime position modifier
            current_regime = getattr(regime_metrics, "regime", None)
            if current_regime is not None:
                regime_modifier = self._get_regime_position_modifier(current_regime)

        # Factor 4: Recent performance decay (0.5 to 1.0)
        performance_factor = 1.0
        if recent_performance is not None:
            recent_win_rate = recent_performance.get("win_rate", 0.5)
            # Scale from 0.5 (poor performance) to 1.0 (excellent)
            performance_factor = 0.5 + (recent_win_rate * 0.5)

        # Factor 5: Drawdown protection (continuous scaling)
        drawdown_factor = self._calculate_drawdown_factor(current_drawdown)

        # Factor 6: Kelly criterion (if we have performance data)
        kelly_factor = 1.0
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_pct = self.calculate_kelly_criterion(
                win_rate, avg_win, avg_loss, fraction=0.5
            )
            # Kelly factor: if Kelly suggests lower than base risk, reduce size
            base_risk_decimal = risk_percent / 100.0
            if base_risk_decimal > 0:
                kelly_factor = min(1.0, max(0.25, kelly_pct / base_risk_decimal))

        # Calculate combined multiplier
        total_multiplier = (
            signal_factor
            * regime_factor
            * regime_modifier
            * performance_factor
            * drawdown_factor
            * kelly_factor
        )

        # Apply multiplier to position size
        final_position_size = position_size * total_multiplier

        # Log the calculation breakdown
        logger.debug(
            f"All-weather sizing for {symbol}: "
            f"base={position_size:.4f}, "
            f"signal={signal_factor:.2f}, "
            f"regime_conf={regime_factor:.2f}, "
            f"regime_mod={regime_modifier:.2f}, "
            f"performance={performance_factor:.2f}, "
            f"drawdown={drawdown_factor:.2f}, "
            f"kelly={kelly_factor:.2f}, "
            f"total_mult={total_multiplier:.2f}, "
            f"final={final_position_size:.4f}"
        )

        # Validate final size
        validation = self.validate_position_size(final_position_size, symbol)

        return PositionSizeResult(
            position_size=final_position_size,
            risk_amount=base_result.risk_amount,
            stop_loss_distance=base_result.stop_loss_distance,
            is_valid=validation["is_valid"],
            error_message=validation.get("error_message"),
            adjusted_size=final_position_size
            if final_position_size != base_result.position_size
            else None,
        )

    def _get_regime_position_modifier(self, regime: Any) -> float:
        """Get position size modifier for a specific regime."""
        # Handle both enum and string regimes
        regime_value = regime.value if hasattr(regime, "value") else str(regime)

        modifiers = {
            "trending_up": 1.0,
            "trending_down": 1.0,
            "ranging": 0.4,
            "volatile": 0.3,
            "quiet": 0.6,
            "unknown": 0.0,
        }

        return modifiers.get(regime_value, 0.5)

    def _calculate_drawdown_factor(self, current_drawdown: float) -> float:
        """Calculate position size factor based on current drawdown."""
        if current_drawdown > 0.15:  # 15% drawdown
            return 0.0  # Stop trading
        elif current_drawdown > 0.10:  # 10% drawdown
            return 0.25  # Reduce to 25%
        elif current_drawdown > 0.07:  # 7% drawdown
            return 0.50  # Reduce to 50%
        elif current_drawdown > 0.05:  # 5% drawdown
            return 0.75  # Reduce to 75%
        return 1.0  # Full size

    def calculate_adaptive_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_price: float,
        symbol: str,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        current_volatility: Optional[float] = None,
        current_drawdown: float = 0.0,
        regime_position_modifier: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate adaptive position size using multiple methods.

        Combines:
        1. Base risk-based sizing
        2. Kelly criterion (if performance data available)
        3. Volatility targeting
        4. Drawdown adjustment
        5. Regime-based modifier

        Args:
            account_balance: Current account balance
            risk_percent: Base risk percentage
            entry_price: Entry price
            stop_loss_price: Stop loss price
            symbol: Trading symbol
            win_rate: Historical win rate (optional)
            avg_win: Average win amount (optional)
            avg_loss: Average loss amount (optional)
            current_volatility: Current volatility (optional)
            current_drawdown: Current drawdown percentage
            regime_position_modifier: Regime-based size modifier

        Returns:
            PositionSizeResult with adaptive sizing
        """
        # Calculate base position size
        base_result = self.calculate_position_size(
            account_balance=account_balance,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            symbol=symbol,
        )

        if not base_result.is_valid:
            return base_result

        position_size = base_result.position_size

        # Apply Kelly criterion if we have performance data
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_pct = self.calculate_kelly_criterion(
                win_rate, avg_win, avg_loss, fraction=0.5
            )

            # Adjust position size based on Kelly
            # If Kelly suggests higher than base, use base (conservative)
            # If Kelly suggests lower, reduce accordingly
            kelly_factor = min(1.0, max(0.25, kelly_pct / (risk_percent / 100)))
            position_size *= kelly_factor

            logger.debug(
                f"Kelly adjustment: factor={kelly_factor:.2f}, kelly_pct={kelly_pct:.4f}"
            )

        # Apply volatility targeting
        if current_volatility is not None:
            position_size = self.calculate_volatility_targeting_size(
                base_position_size=position_size, current_volatility=current_volatility
            )

            logger.debug(f"Volatility targeting applied: vol={current_volatility:.2%}")

        # Apply drawdown adjustment
        if current_drawdown > 0.05:  # 5% drawdown
            if current_drawdown > 0.10:  # 10% drawdown
                position_size *= 0.25  # Reduce to 25%
            elif current_drawdown > 0.07:  # 7% drawdown
                position_size *= 0.5  # Reduce to 50%
            else:
                position_size *= 0.75  # Reduce to 75%

            logger.debug(
                f"Drawdown adjustment: dd={current_drawdown:.2%}, size={position_size:.4f}"
            )

        # Apply regime modifier
        position_size *= regime_position_modifier

        # Validate final size
        validation = self.validate_position_size(position_size, symbol)

        return PositionSizeResult(
            position_size=position_size,
            risk_amount=base_result.risk_amount,
            stop_loss_distance=base_result.stop_loss_distance,
            is_valid=validation["is_valid"],
            error_message=validation.get("error_message"),
            adjusted_size=position_size
            if position_size != base_result.position_size
            else None,
        )

    def calculate_risk_metrics(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        position_size: float,
        account_balance: float,
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for a position.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            position_size: Position size
            account_balance: Account balance

        Returns:
            Dictionary with risk metrics
        """
        stop_loss_distance = abs(entry_price - stop_loss_price)
        take_profit_distance = abs(take_profit_price - entry_price)

        risk_amount = stop_loss_distance * position_size
        potential_profit = take_profit_distance * position_size

        risk_percent = (risk_amount / account_balance) * 100
        potential_profit_percent = (potential_profit / account_balance) * 100

        risk_reward_ratio = potential_profit / risk_amount if risk_amount > 0 else 0

        return {
            "risk_amount": risk_amount,
            "risk_percent": risk_percent,
            "potential_profit": potential_profit,
            "potential_profit_percent": potential_profit_percent,
            "risk_reward_ratio": risk_reward_ratio,
            "stop_loss_distance": stop_loss_distance,
            "take_profit_distance": take_profit_distance,
            "position_value": entry_price * position_size,
        }
