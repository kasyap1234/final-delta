"""
Enhanced Signal Detector with Mean Reversion

Combines trend-following and mean-reversion signals for robust trading decisions.
Uses regime-based weighting to adapt to market conditions.

Phase 2: Layer B regime-specific alpha decomposition for expectancy-positive redesign.
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
from .market_regime import (
    MarketRegimeDetector,
    MarketRegime,
    RegimeMetrics,
    get_regime_profile,
)

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
        current_volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        regime_profile: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Check for entry signals combining trend and mean reversion.

        Args:
            symbol: Trading pair symbol
            indicators: Current indicator values
            current_price: Current market price
            price_history: Optional price history for divergence detection
            regime_metrics: Optional pre-calculated regime metrics
            current_volume: Current candle volume
            avg_volume: Average volume over lookback period
            regime_profile: Optional regime profile dict from REGIME_PROFILES

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

        # Use regime profile weights if provided, otherwise fall back to adaptive params
        if regime_profile is not None:
            trend_weight = regime_profile["trend_weight"]
            mr_weight = regime_profile["mr_weight"]
        else:
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

        # Volume confirmation filter: smooth scaling in low-volume candles.
        volume_penalty_applied = False
        if (
            current_volume is not None
            and avg_volume is not None
            and avg_volume > 0
            and combined["strength"] > 0
        ):
            vol_multiplier = 1.2  # default
            if regime_profile is not None:
                vol_multiplier = regime_profile.get("volume_multiplier", 1.2)
            volume_ratio = current_volume / (vol_multiplier * avg_volume)
            if volume_ratio < 1.0:
                # Clamp to avoid collapsing all signals in quieter sessions.
                volume_scale = max(0.65, min(1.0, volume_ratio))
                combined["strength"] *= volume_scale
                volume_penalty_applied = True

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
                "volume_penalty": volume_penalty_applied,
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
                from .technical_indicators import calculate_rsi

                rsi_history = calculate_rsi(price_history, period=14)
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
                from .technical_indicators import calculate_rsi

                rsi_history = calculate_rsi(price_history, period=14)
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
        """Check for exit signals.

        Only exits on strong confirmations to avoid cutting winners short:
        - Trend reversal requires ADX > 25 (strong trend against position)
        - RSI must hit actual overbought/oversold, not "approaching"
        - Bearish/bullish crossover requires trend confirmation
        """
        details = {
            "position_type": position_type,
            "entry_price": entry_price,
            "current_price": current_price,
            "unrealized_pnl": (current_price - entry_price) / entry_price,
        }

        if position_type == "long":
            # Exit on confirmed trend reversal (strong downtrend)
            if (
                indicators.trend == "downtrend"
                and indicators.adx is not None
                and indicators.adx > 25
            ):
                return Signal(
                    signal=SignalType.SELL,
                    reason="Trend reversal to downtrend",
                    strength=0.8,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

            if indicators.rsi and indicators.rsi >= self.rsi_overbought + 15:
                if indicators.adx is None or indicators.adx < 30:
                    return Signal(
                        signal=SignalType.SELL,
                        reason=f"RSI extremely overbought ({indicators.rsi:.1f})",
                        strength=0.7,
                        symbol=symbol,
                        price=current_price,
                        details=details,
                    )

            # Exit on bearish crossover with trend confirmation
            if (
                indicators.last_crossover == CrossoverType.BEARISH
                and indicators.trend == "downtrend"
            ):
                return Signal(
                    signal=SignalType.SELL,
                    reason="Bearish EMA crossover in downtrend",
                    strength=0.6,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

        else:  # short position
            # Exit on confirmed trend reversal (strong uptrend)
            if (
                indicators.trend == "uptrend"
                and indicators.adx is not None
                and indicators.adx > 25
            ):
                return Signal(
                    signal=SignalType.BUY,
                    reason="Trend reversal to uptrend",
                    strength=0.8,
                    symbol=symbol,
                    price=current_price,
                    details=details,
                )

            if indicators.rsi and indicators.rsi <= self.rsi_oversold - 15:
                if indicators.adx is None or indicators.adx < 30:
                    return Signal(
                        signal=SignalType.BUY,
                        reason=f"RSI extremely oversold ({indicators.rsi:.1f})",
                        strength=0.7,
                        symbol=symbol,
                        price=current_price,
                        details=details,
                    )

            # Exit on bullish crossover with trend confirmation
            if (
                indicators.last_crossover == CrossoverType.BULLISH
                and indicators.trend == "uptrend"
            ):
                return Signal(
                    signal=SignalType.BUY,
                    reason="Bullish EMA crossover in uptrend",
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


# ─── Phase 2: Layer B Regime-Specific Alpha Decomposition ─────────────────────

@dataclass
class RegimeAlphaResult:
    """Result of regime-specific alpha decomposition."""
    
    passed: bool  # Whether alpha confirmation passed
    reason: str  # Reason for pass/fail
    alpha_score: float  # Alpha quality score (0-1)
    confirmation_factors: Dict[str, bool]  # Individual factor results
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "alpha_score": self.alpha_score,
            "confirmation_factors": self.confirmation_factors,
        }


def check_trending_alpha(
    indicators: IndicatorValues,
    direction: str,
    price_history: Optional[np.ndarray] = None,
) -> RegimeAlphaResult:
    """
    Check trend regime alpha confirmation.
    
    Trending model requirements:
    - Continuation pullback entries with structure alignment
    - Breakout continuation entries with volume confirmation
    - Invalidation when trend structure breaks and momentum decays
    
    Args:
        indicators: IndicatorValues with technical indicators
        direction: 'long' or 'short'
        price_history: Price history for structure analysis
    
    Returns:
        RegimeAlphaResult with confirmation status
    """
    factors = {}
    score = 0.0
    
    # Factor 1: Trend structure alignment
    if indicators.trend:
        if direction == "long" and indicators.trend == "uptrend":
            factors["trend_alignment"] = True
            score += 0.35
        elif direction == "short" and indicators.trend == "downtrend":
            factors["trend_alignment"] = True
            score += 0.35
        else:
            factors["trend_alignment"] = False
    else:
        factors["trend_alignment"] = False
    
    # Factor 2: EMA spread for structure
    if indicators.ema_9 and indicators.ema_50:
        ema_spread = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50
        if ema_spread >= 0.005:  # 0.5% minimum spread
            factors["ema_structure"] = True
            score += 0.25
        else:
            factors["ema_structure"] = False
    else:
        factors["ema_structure"] = False
    
    # Factor 3: ADX for trend strength
    if indicators.adx is not None:
        if indicators.adx >= 20.0:  # Minimum trend strength
            factors["adx_strength"] = True
            score += 0.20
        else:
            factors["adx_strength"] = False
    else:
        factors["adx_strength"] = False
    
    # Factor 4: Volume confirmation (if available)
    # This is checked separately in multi-confirmation gate
    factors["volume_confirmation"] = True  # Assume checked elsewhere
    score += 0.20
    
    # Minimum score threshold for trending regime
    min_score = 0.60
    passed = score >= min_score
    
    reason = (
        f"Trending alpha passed: score {score:.2f} >= {min_score:.2f}"
        if passed
        else f"Trending alpha failed: score {score:.2f} < {min_score:.2f}"
    )
    
    return RegimeAlphaResult(
        passed=passed,
        reason=reason,
        alpha_score=score,
        confirmation_factors=factors,
    )


def check_ranging_alpha(
    indicators: IndicatorValues,
    direction: str,
    price_history: Optional[np.ndarray] = None,
) -> RegimeAlphaResult:
    """
    Check ranging regime alpha confirmation.
    
    Ranging model requirements:
    - Edge-of-range mean reversion only
    - Require rejection and structure confirmation near support or resistance
    - Invalidation on range break with momentum expansion
    
    Args:
        indicators: IndicatorValues with technical indicators
        direction: 'long' or 'short'
        price_history: Price history for structure analysis
    
    Returns:
        RegimeAlphaResult with confirmation status
    """
    factors = {}
    score = 0.0
    
    # Factor 1: RSI extreme for mean reversion
    if indicators.rsi is not None:
        if direction == "long" and indicators.rsi <= 35.0:
            factors["rsi_oversold"] = True
            score += 0.35
        elif direction == "short" and indicators.rsi >= 65.0:
            factors["rsi_overbought"] = True
            score += 0.35
        else:
            factors["rsi_extreme"] = False
    else:
        factors["rsi_extreme"] = False
    
    # Factor 2: Structure near support/resistance
    # Use EMA values as proxy for current price since close is not available
    current_price_proxy = indicators.ema_9 if indicators.ema_9 else (indicators.ema_50 if indicators.ema_50 else 0)
    
    if indicators.s1 is not None and indicators.s2 is not None and current_price_proxy > 0:
        if direction == "long":
            # Check if near support
            support_levels = np.array([indicators.s1, indicators.s2])
            near_support, _ = is_near_support(
                current_price_proxy,
                support_levels,
                0.005,
            )
            if near_support:
                factors["near_structure"] = True
                score += 0.30
            else:
                factors["near_structure"] = False
        else:
            # Check if near resistance
            if indicators.r1 is not None and indicators.r2 is not None:
                resistance_levels = np.array([indicators.r1, indicators.r2])
                near_resistance, _ = is_near_resistance(
                    current_price_proxy,
                    resistance_levels,
                    0.005,
                )
                if near_resistance:
                    factors["near_structure"] = True
                    score += 0.30
                else:
                    factors["near_structure"] = False
    else:
        factors["near_structure"] = False
    
    # Factor 3: Low ADX (confirming ranging, not trending)
    if indicators.adx is not None:
        if indicators.adx <= 25.0:  # Low ADX confirms ranging
            factors["low_adx"] = True
            score += 0.20
        else:
            factors["low_adx"] = False
    else:
        factors["low_adx"] = False
    
    # Factor 4: EMA convergence (tight spread)
    if indicators.ema_9 and indicators.ema_50:
        ema_spread = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50
        if ema_spread <= 0.01:  # Tight spread confirms ranging
            factors["ema_convergence"] = True
            score += 0.15
        else:
            factors["ema_convergence"] = False
    else:
        factors["ema_convergence"] = False
    
    # Minimum score threshold for ranging regime
    min_score = 0.65
    passed = score >= min_score
    
    reason = (
        f"Ranging alpha passed: score {score:.2f} >= {min_score:.2f}"
        if passed
        else f"Ranging alpha failed: score {score:.2f} < {min_score:.2f}"
    )
    
    return RegimeAlphaResult(
        passed=passed,
        reason=reason,
        alpha_score=score,
        confirmation_factors=factors,
    )


def check_volatile_alpha(
    indicators: IndicatorValues,
    direction: str,
    price_history: Optional[np.ndarray] = None,
    breakout_condition: bool = False,
    directional_momentum: bool = False,
) -> RegimeAlphaResult:
    """
    Check volatile regime alpha confirmation.
    
    Volatile model requirements:
    - Breakout and momentum agreement required
    - No countertrend entries in high-vol regime unless special override passes
    - Invalidation on failed breakout return inside range
    
    Args:
        indicators: IndicatorValues with technical indicators
        direction: 'long' or 'short'
        price_history: Price history for structure analysis
        breakout_condition: Whether breakout condition is met
        directional_momentum: Whether directional momentum agrees
    
    Returns:
        RegimeAlphaResult with confirmation status
    """
    factors = {}
    score = 0.0
    
    # Factor 1: Breakout condition (mandatory for volatile regime)
    if breakout_condition:
        factors["breakout"] = True
        score += 0.35
    else:
        factors["breakout"] = False
    
    # Factor 2: Directional momentum agreement
    if directional_momentum:
        factors["momentum_agreement"] = True
        score += 0.30
    else:
        factors["momentum_agreement"] = False
    
    # Factor 3: High ADX (confirming strong trend/volatility)
    if indicators.adx is not None:
        if indicators.adx >= 25.0:  # High ADX confirms volatility
            factors["high_adx"] = True
            score += 0.20
        else:
            factors["high_adx"] = False
    else:
        factors["high_adx"] = False
    
    # Factor 4: Wide EMA spread (confirming volatility)
    if indicators.ema_9 and indicators.ema_50:
        ema_spread = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50
        if ema_spread >= 0.01:  # Wide spread confirms volatility
            factors["wide_ema_spread"] = True
            score += 0.15
        else:
            factors["wide_ema_spread"] = False
    else:
        factors["wide_ema_spread"] = False
    
    # Minimum score threshold for volatile regime
    min_score = 0.70
    passed = score >= min_score
    
    reason = (
        f"Volatile alpha passed: score {score:.2f} >= {min_score:.2f}"
        if passed
        else f"Volatile alpha failed: score {score:.2f} < {min_score:.2f}"
    )
    
    return RegimeAlphaResult(
        passed=passed,
        reason=reason,
        alpha_score=score,
        confirmation_factors=factors,
    )


def check_quiet_alpha(
    indicators: IndicatorValues,
    direction: str,
    price_history: Optional[np.ndarray] = None,
) -> RegimeAlphaResult:
    """
    Check quiet regime alpha confirmation.
    
    Quiet model requirements:
    - Selective low-frequency entries only
    - Stricter ambiguity filter to avoid noise trading
    
    Args:
        indicators: IndicatorValues with technical indicators
        direction: 'long' or 'short'
        price_history: Price history for structure analysis
    
    Returns:
        RegimeAlphaResult with confirmation status
    """
    factors = {}
    score = 0.0
    
    # Factor 1: Moderate trend direction (quiet but directional)
    if indicators.trend:
        if (direction == "long" and indicators.trend == "uptrend") or \
           (direction == "short" and indicators.trend == "downtrend"):
            factors["trend_direction"] = True
            score += 0.30
        else:
            factors["trend_direction"] = False
    else:
        factors["trend_direction"] = False
    
    # Factor 2: Moderate ADX (not too strong, not too weak)
    if indicators.adx is not None:
        if 15.0 <= indicators.adx <= 25.0:  # Moderate ADX
            factors["moderate_adx"] = True
            score += 0.25
        else:
            factors["moderate_adx"] = False
    else:
        factors["moderate_adx"] = False
    
    # Factor 3: RSI not extreme (avoid mean reversion in quiet regime)
    if indicators.rsi is not None:
        if 40.0 <= indicators.rsi <= 60.0:  # Neutral RSI
            factors["neutral_rsi"] = True
            score += 0.25
        else:
            factors["neutral_rsi"] = False
    else:
        factors["neutral_rsi"] = False
    
    # Factor 4: Tight EMA spread (confirming quiet conditions)
    if indicators.ema_9 and indicators.ema_50:
        ema_spread = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50
        if ema_spread <= 0.008:  # Tight spread
            factors["tight_ema_spread"] = True
            score += 0.20
        else:
            factors["tight_ema_spread"] = False
    else:
        factors["tight_ema_spread"] = False
    
    # Minimum score threshold for quiet regime
    min_score = 0.60
    passed = score >= min_score
    
    reason = (
        f"Quiet alpha passed: score {score:.2f} >= {min_score:.2f}"
        if passed
        else f"Quiet alpha failed: score {score:.2f} < {min_score:.2f}"
    )
    
    return RegimeAlphaResult(
        passed=passed,
        reason=reason,
        alpha_score=score,
        confirmation_factors=factors,
    )


def check_regime_alpha(
    regime: MarketRegime,
    indicators: IndicatorValues,
    direction: str,
    price_history: Optional[np.ndarray] = None,
    breakout_condition: bool = False,
    directional_momentum: bool = False,
) -> RegimeAlphaResult:
    """
    Check regime-specific alpha confirmation (Layer B).
    
    Routes to the appropriate regime-specific alpha check based on the current regime.
    
    Args:
        regime: Current market regime
        indicators: IndicatorValues with technical indicators
        direction: 'long' or 'short'
        price_history: Price history for structure analysis
        breakout_condition: Whether breakout condition is met (for volatile regime)
        directional_momentum: Whether directional momentum agrees (for volatile regime)
    
    Returns:
        RegimeAlphaResult with confirmation status
    """
    regime_value = regime.value
    
    if regime_value in ("trending_up", "trending_down"):
        return check_trending_alpha(indicators, direction, price_history)
    elif regime_value == "ranging":
        return check_ranging_alpha(indicators, direction, price_history)
    elif regime_value == "volatile":
        return check_volatile_alpha(
            indicators, direction, price_history, breakout_condition, directional_momentum
        )
    elif regime_value == "quiet":
        return check_quiet_alpha(indicators, direction, price_history)
    else:  # unknown
        # Conservative: require higher threshold for unknown regime
        return RegimeAlphaResult(
            passed=False,
            reason="Unknown regime - alpha confirmation blocked",
            alpha_score=0.0,
            confirmation_factors={},
        )
