"""
Enhanced Signal Detector with Mean Reversion

Combines trend-following and mean-reversion signals for robust trading decisions.
Uses regime-based weighting to adapt to market conditions.
"""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

from .technical_indicators import (
    CrossoverType,
    is_near_resistance,
    is_near_support,
    detect_rsi_divergence,
    get_trend_direction,
    calculate_ema_crossover,
)
from .indicator_manager import IndicatorValues
from .market_regime import MarketRegimeDetector, MarketRegime, RegimeMetrics

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""

    NONE = "none"
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    MEAN_REVERSION_LONG = "mean_reversion_long"
    MEAN_REVERSION_SHORT = "mean_reversion_short"


@dataclass
class Signal:
    """Trading signal with metadata."""

    signal: SignalType
    reason: str
    strength: float
    symbol: str
    price: float
    timestamp: Optional[str] = None
    details: Dict[str, Any] = None
    regime: Optional[str] = None
    trend_weight: float = 0.7
    mr_weight: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "reason": self.reason,
            "strength": self.strength,
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp,
            "details": self.details or {},
            "regime": self.regime,
            "trend_weight": self.trend_weight,
            "mr_weight": self.mr_weight,
        }


class EnhancedSignalDetector:
    """
    Enhanced signal detector combining trend and mean reversion strategies.

    Adapts signal generation based on detected market regime:
    - Trending markets: Emphasize trend-following signals (70-90%)
    - Ranging markets: Emphasize mean reversion signals (70-80%)
    - Volatile markets: Require higher confirmation, reduce size
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # RSI thresholds
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_mid_high = self.config.get("rsi_mid_high", 60)
        self.rsi_mid_low = self.config.get("rsi_mid_low", 40)

        # Bollinger Band settings
        self.bb_std_dev = self.config.get("bb_std_dev", 2.0)
        self.bb_period = self.config.get("bb_period", 20)

        # Signal strength thresholds
        self.strong_signal_threshold = self.config.get("strong_signal_threshold", 0.75)
        self.weak_signal_threshold = self.config.get("weak_signal_threshold", 0.35)

        # Mean reversion settings
        self.mr_rsi_threshold = self.config.get("mr_rsi_threshold", 20)
        self.mr_bb_threshold = self.config.get("mr_bb_threshold", 0.02)

        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(config)

        # Historical performance tracking for adaptive weights
        self.trend_performance = {"wins": 0, "losses": 0}
        self.mr_performance = {"wins": 0, "losses": 0}

        logger.info(
            f"EnhancedSignalDetector initialized: RSI({self.rsi_oversold}/{self.rsi_overbought}), "
            f"MR_RSI={self.mr_rsi_threshold}"
        )

    def check_entry_signal(
        self,
        symbol: str,
        indicators: IndicatorValues,
        current_price: float,
        price_history: Optional[np.ndarray] = None,
        regime_metrics: Optional[RegimeMetrics] = None,
    ) -> Signal:
        """
        Check for entry signals combining trend and mean reversion.

        Args:
            symbol: Trading pair symbol
            indicators: Current indicator values
            current_price: Current market price
            price_history: Optional price history for divergence detection
            regime_metrics: Optional pre-calculated regime metrics

        Returns:
            Signal object with combined signal
        """
        # Detect regime if not provided
        if regime_metrics is None:
            prices = (
                price_history
                if price_history is not None
                else np.array([current_price])
            )
            regime_metrics = self.regime_detector.detect_regime(
                prices=prices,
                ema_fast=indicators.ema_9,
                ema_slow=indicators.ema_50,
                adx=indicators.adx,
                atr=indicators.atr,
            )

        regime = regime_metrics.regime

        # Get adaptive weights based on regime
        weights = self.regime_detector.get_adaptive_parameters(regime)
        trend_weight = weights["trend_weight"]
        mr_weight = weights["mean_reversion_weight"]

        # Generate trend-following signal
        trend_signal = self._check_trend_signal(
            indicators, current_price, price_history
        )

        # Generate mean reversion signal
        mr_signal = self._check_mean_reversion_signal(
            indicators, current_price, price_history
        )

        # Combine signals with regime-based weighting
        combined = self._combine_signals(
            trend_signal, mr_signal, trend_weight, mr_weight, regime
        )

        # Create final signal
        final_signal = Signal(
            signal=combined["signal_type"],
            reason=combined["reason"],
            strength=combined["strength"],
            symbol=symbol,
            price=current_price,
            regime=regime.value,
            trend_weight=trend_weight,
            mr_weight=mr_weight,
            details={
                "trend_signal": trend_signal["type"].value,
                "trend_strength": trend_signal["strength"],
                "mr_signal": mr_signal["type"].value,
                "mr_strength": mr_signal["strength"],
                "regime_confidence": regime_metrics.confidence,
                "adx": indicators.adx,
                "rsi": indicators.rsi,
                "ema_9": indicators.ema_9,
                "ema_50": indicators.ema_50,
                "trend": indicators.trend,
            },
        )

        return final_signal

    def _check_trend_signal(
        self,
        indicators: IndicatorValues,
        current_price: float,
        price_history: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Generate trend-following signal."""
        signals = []

        # EMA Crossover signal
        if indicators.last_crossover == CrossoverType.BULLISH:
            if indicators.trend == "uptrend":
                signals.append(
                    (
                        "crossover",
                        SignalType.BUY,
                        0.8,
                        "Bullish EMA crossover in uptrend",
                    )
                )
            else:
                signals.append(
                    ("crossover", SignalType.BUY, 0.5, "Bullish EMA crossover")
                )
        elif indicators.last_crossover == CrossoverType.BEARISH:
            if indicators.trend == "downtrend":
                signals.append(
                    (
                        "crossover",
                        SignalType.SELL,
                        0.8,
                        "Bearish EMA crossover in downtrend",
                    )
                )
            else:
                signals.append(
                    ("crossover", SignalType.SELL, 0.5, "Bearish EMA crossover")
                )

        # Trend alignment signal
        if indicators.ema_9 and indicators.ema_21 and indicators.ema_50:
            if indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
                signals.append(
                    ("alignment", SignalType.BUY, 0.4, "Bullish EMA alignment")
                )
            elif indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
                signals.append(
                    ("alignment", SignalType.SELL, 0.4, "Bearish EMA alignment")
                )

        # RSI confirmation for trend
        if indicators.rsi is not None:
            if indicators.rsi > self.rsi_mid_high and indicators.trend == "uptrend":
                signals.append(
                    (
                        "rsi",
                        SignalType.BUY,
                        0.3,
                        f"RSI confirming uptrend ({indicators.rsi:.1f})",
                    )
                )
            elif indicators.rsi < self.rsi_mid_low and indicators.trend == "downtrend":
                signals.append(
                    (
                        "rsi",
                        SignalType.SELL,
                        0.3,
                        f"RSI confirming downtrend ({indicators.rsi:.1f})",
                    )
                )

        # ADX trend strength
        if indicators.adx is not None and indicators.adx > 25:
            if indicators.trend == "uptrend":
                signals.append(
                    (
                        "adx",
                        SignalType.BUY,
                        0.35,
                        f"Strong uptrend (ADX: {indicators.adx:.1f})",
                    )
                )
            elif indicators.trend == "downtrend":
                signals.append(
                    (
                        "adx",
                        SignalType.SELL,
                        0.35,
                        f"Strong downtrend (ADX: {indicators.adx:.1f})",
                    )
                )

        # Combine trend signals
        return self._aggregate_signals(signals)

    def _check_mean_reversion_signal(
        self,
        indicators: IndicatorValues,
        current_price: float,
        price_history: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Generate mean reversion signal."""
        signals = []

        if indicators.rsi is None:
            return {"type": SignalType.NONE, "strength": 0.0, "reason": "No RSI data"}

        # RSI extreme with divergence
        if indicators.rsi <= self.rsi_oversold:
            divergence = None
            if price_history is not None and len(price_history) >= 28:
                rsi_history = np.array(
                    [indicators.rsi] * len(price_history)
                )  # Simplified
                divergence = detect_rsi_divergence(
                    price_history[-28:], rsi_history[-28:]
                )

            if divergence == "bullish":
                signals.append(
                    (
                        "divergence",
                        SignalType.MEAN_REVERSION_LONG,
                        0.9,
                        f"RSI oversold ({indicators.rsi:.1f}) with bullish divergence",
                    )
                )
            else:
                signals.append(
                    (
                        "oversold",
                        SignalType.MEAN_REVERSION_LONG,
                        0.6,
                        f"RSI oversold ({indicators.rsi:.1f})",
                    )
                )

        elif indicators.rsi >= self.rsi_overbought:
            divergence = None
            if price_history is not None and len(price_history) >= 28:
                rsi_history = np.array([indicators.rsi] * len(price_history))
                divergence = detect_rsi_divergence(
                    price_history[-28:], rsi_history[-28:]
                )

            if divergence == "bearish":
                signals.append(
                    (
                        "divergence",
                        SignalType.MEAN_REVERSION_SHORT,
                        0.9,
                        f"RSI overbought ({indicators.rsi:.1f}) with bearish divergence",
                    )
                )
            else:
                signals.append(
                    (
                        "overbought",
                        SignalType.MEAN_REVERSION_SHORT,
                        0.6,
                        f"RSI overbought ({indicators.rsi:.1f})",
                    )
                )

        # Bollinger Band mean reversion
        if indicators.pivot is not None:
            bb_distance = abs(current_price - indicators.pivot) / indicators.pivot

            if bb_distance > self.mr_bb_threshold:
                if current_price < indicators.pivot and indicators.rsi < 45:
                    signals.append(
                        (
                            "bb",
                            SignalType.MEAN_REVERSION_LONG,
                            0.5,
                            f"Price below pivot, potential mean reversion",
                        )
                    )
                elif current_price > indicators.pivot and indicators.rsi > 55:
                    signals.append(
                        (
                            "bb",
                            SignalType.MEAN_REVERSION_SHORT,
                            0.5,
                            f"Price above pivot, potential mean reversion",
                        )
                    )

        # Support/Resistance bounce
        if indicators.s1 is not None and indicators.s2 is not None:
            support_levels = np.array([indicators.s1, indicators.s2])
            near_support, nearest_s = is_near_support(
                current_price, support_levels, 0.005
            )
            if near_support and indicators.rsi < 50:
                signals.append(
                    (
                        "support",
                        SignalType.MEAN_REVERSION_LONG,
                        0.55,
                        f"Price near support ({nearest_s:.2f})",
                    )
                )

        if indicators.r1 is not None and indicators.r2 is not None:
            resistance_levels = np.array([indicators.r1, indicators.r2])
            near_resistance, nearest_r = is_near_resistance(
                current_price, resistance_levels, 0.005
            )
            if near_resistance and indicators.rsi > 50:
                signals.append(
                    (
                        "resistance",
                        SignalType.MEAN_REVERSION_SHORT,
                        0.55,
                        f"Price near resistance ({nearest_r:.2f})",
                    )
                )

        return self._aggregate_signals(signals)

    def _aggregate_signals(self, signals: List[Tuple]) -> Dict[str, Any]:
        """Aggregate multiple signals into one."""
        if not signals:
            return {"type": SignalType.NONE, "strength": 0.0, "reason": "No signal"}

        # Group by direction
        long_signals = [
            s
            for s in signals
            if s[1]
            in (SignalType.BUY, SignalType.STRONG_BUY, SignalType.MEAN_REVERSION_LONG)
        ]
        short_signals = [
            s
            for s in signals
            if s[1]
            in (
                SignalType.SELL,
                SignalType.STRONG_SELL,
                SignalType.MEAN_REVERSION_SHORT,
            )
        ]

        # Calculate total strength for each direction
        long_strength = sum(s[2] for s in long_signals)
        short_strength = sum(s[2] for s in short_signals)

        if long_strength > short_strength and long_strength > 0:
            signal_type = (
                SignalType.STRONG_BUY
                if long_strength >= self.strong_signal_threshold
                else SignalType.BUY
            )
            reasons = [s[3] for s in long_signals]
            return {
                "type": signal_type,
                "strength": min(long_strength, 1.0),
                "reason": "; ".join(reasons),
            }
        elif short_strength > long_strength and short_strength > 0:
            signal_type = (
                SignalType.STRONG_SELL
                if short_strength >= self.strong_signal_threshold
                else SignalType.SELL
            )
            reasons = [s[3] for s in short_signals]
            return {
                "type": signal_type,
                "strength": min(short_strength, 1.0),
                "reason": "; ".join(reasons),
            }

        return {"type": SignalType.NONE, "strength": 0.0, "reason": "No clear signal"}

    def _combine_signals(
        self,
        trend_signal: Dict[str, Any],
        mr_signal: Dict[str, Any],
        trend_weight: float,
        mr_weight: float,
        regime: MarketRegime,
    ) -> Dict[str, Any]:
        """Combine trend and mean reversion signals with regime-based weighting."""

        # Extract directions
        trend_long = trend_signal["type"] in (SignalType.BUY, SignalType.STRONG_BUY)
        trend_short = trend_signal["type"] in (SignalType.SELL, SignalType.STRONG_SELL)
        mr_long = mr_signal["type"] in (SignalType.MEAN_REVERSION_LONG,)
        mr_short = mr_signal["type"] in (SignalType.MEAN_REVERSION_SHORT,)

        # Calculate weighted scores
        long_score = 0.0
        short_score = 0.0

        if trend_long:
            long_score += trend_signal["strength"] * trend_weight
        if trend_short:
            short_score += trend_signal["strength"] * trend_weight
        if mr_long:
            long_score += mr_signal["strength"] * mr_weight
        if mr_short:
            short_score += mr_signal["strength"] * mr_weight

        # Determine final signal
        if long_score > short_score and long_score >= self.weak_signal_threshold:
            strength = min(long_score, 1.0)
            if strength >= self.strong_signal_threshold:
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY

            reasons = []
            if trend_long:
                reasons.append(f"Trend: {trend_signal['reason']}")
            if mr_long:
                reasons.append(f"MR: {mr_signal['reason']}")

            return {
                "signal_type": signal_type,
                "strength": strength,
                "reason": " | ".join(reasons),
            }

        elif short_score > long_score and short_score >= self.weak_signal_threshold:
            strength = min(short_score, 1.0)
            if strength >= self.strong_signal_threshold:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL

            reasons = []
            if trend_short:
                reasons.append(f"Trend: {trend_signal['reason']}")
            if mr_short:
                reasons.append(f"MR: {mr_signal['reason']}")

            return {
                "signal_type": signal_type,
                "strength": strength,
                "reason": " | ".join(reasons),
            }

        return {
            "signal_type": SignalType.NONE,
            "strength": 0.0,
            "reason": "No clear signal",
        }

    def check_exit_signal(
        self,
        symbol: str,
        indicators: IndicatorValues,
        current_price: float,
        entry_price: float,
        position_type: str,
    ) -> Signal:
        """Check for exit signals."""
        details = {
            "position_type": position_type,
            "entry_price": entry_price,
            "current_price": current_price,
            "unrealized_pnl": (current_price - entry_price) / entry_price,
        }

        if position_type == "long":
            # Exit on trend reversal
            if indicators.trend == "downtrend" and indicators.adx > 20:
                return Signal(
                    signal=SignalType.SELL,
                    reason="Trend reversal to downtrend",
                    strength=0.8,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

            # Exit on RSI overbought
            if indicators.rsi and indicators.rsi >= self.rsi_overbought - 5:
                return Signal(
                    signal=SignalType.SELL,
                    reason=f"RSI approaching overbought ({indicators.rsi:.1f})",
                    strength=0.7,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

            # Exit on bearish crossover
            if indicators.last_crossover == CrossoverType.BEARISH:
                return Signal(
                    signal=SignalType.SELL,
                    reason="Bearish EMA crossover",
                    strength=0.6,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

        else:  # short position
            # Exit on trend reversal
            if indicators.trend == "uptrend" and indicators.adx > 20:
                return Signal(
                    signal=SignalType.BUY,
                    reason="Trend reversal to uptrend",
                    strength=0.8,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

            # Exit on RSI oversold
            if indicators.rsi and indicators.rsi <= self.rsi_oversold + 5:
                return Signal(
                    signal=SignalType.BUY,
                    reason=f"RSI approaching oversold ({indicators.rsi:.1f})",
                    strength=0.7,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

            # Exit on bullish crossover
            if indicators.last_crossover == CrossoverType.BULLISH:
                return Signal(
                    signal=SignalType.BUY,
                    reason="Bullish EMA crossover",
                    strength=0.6,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

        return Signal(
            signal=SignalType.NONE,
            reason="No exit signal",
            strength=0.0,
            symbol=symbol,
            price=current_price,
            details=details,
        )

    def get_stop_loss_price(
        self,
        indicators: IndicatorValues,
        entry_price: float,
        position_type: str,
        multiplier: Optional[float] = None,
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            indicators: IndicatorValues with ATR
            entry_price: Entry price
            position_type: 'long' or 'short'
            multiplier: ATR multiplier (default: 2.0)

        Returns:
            Stop loss price
        """
        atr = indicators.atr if indicators.atr else entry_price * 0.02
        mult = multiplier if multiplier is not None else 2.0

        stop_distance = atr * mult

        if position_type == "long":
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance

    def get_take_profit_price(
        self,
        indicators: IndicatorValues,
        entry_price: float,
        position_type: str,
        risk_reward_ratio: Optional[float] = None,
    ) -> float:
        """
        Calculate take profit price based on risk:reward ratio.

        Args:
            indicators: IndicatorValues with ATR for calculating risk distance
            entry_price: Entry price
            position_type: 'long' or 'short'
            risk_reward_ratio: R:R ratio (default: 2.0)

        Returns:
            Take profit price
        """
        atr = indicators.atr if indicators.atr else entry_price * 0.02
        stop_distance = atr * 2.0  # Base stop distance

        rr = risk_reward_ratio if risk_reward_ratio is not None else 2.0
        profit_distance = stop_distance * rr

        if position_type == "long":
            return entry_price + profit_distance
        else:  # short
            return entry_price - profit_distance

    def update_performance(self, signal_type: str, was_profitable: bool):
        """Update performance tracking for adaptive weighting."""
        if "MEAN_REVERSION" in signal_type:
            if was_profitable:
                self.mr_performance["wins"] += 1
            else:
                self.mr_performance["losses"] += 1
        else:
            if was_profitable:
                self.trend_performance["wins"] += 1
            else:
                self.trend_performance["losses"] += 1
