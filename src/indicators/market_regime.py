"""
Enhanced Market Regime Detection Module with Dynamic Thresholds

Detects market regimes (trending, ranging, volatile, quiet) using adaptive methods:
1. Dynamic thresholds based on rolling volatility percentiles
2. Enhanced confidence scoring using multiple indicator agreement
3. Historical context for better regime classification

This module enables truly adaptive trading strategies that adjust to current market conditions.
"""

from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Container for regime detection metrics."""

    regime: MarketRegime
    confidence: float
    adx: float
    volatility: float
    bb_width: float
    ema_spread: float
    trend_strength: float
    timestamp: Optional[str] = None
    # New fields for enhanced detection
    dynamic_thresholds: Optional[Dict[str, float]] = None
    indicator_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "adx": self.adx,
            "volatility": self.volatility,
            "bb_width": self.bb_width,
            "ema_spread": self.ema_spread,
            "trend_strength": self.trend_strength,
            "timestamp": self.timestamp,
            "dynamic_thresholds": self.dynamic_thresholds,
            "indicator_scores": self.indicator_scores,
        }


class AdaptiveMarketRegimeDetector:
    """
    Enhanced regime detector with dynamic thresholds and confidence scoring.

    Key improvements:
    1. Dynamic thresholds based on rolling volatility percentiles
    2. Multi-factor confidence calculation
    3. Historical context for better classification
    4. Smooth regime transitions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive regime detector.

        Args:
            config: Configuration dictionary with regime detection settings
        """
        self.config = config or {}

        # Base thresholds (will be adjusted dynamically)
        self.base_adx_strong_trend = self.config.get("adx_strong_trend", 25.0)
        self.base_adx_weak_trend = self.config.get("adx_weak_trend", 20.0)
        self.base_bb_squeeze_threshold = self.config.get("bb_squeeze_threshold", 0.06)
        self.base_bb_volatile_threshold = self.config.get("bb_volatile_threshold", 0.10)
        self.base_vol_low_threshold = self.config.get("vol_low_threshold", 0.015)
        self.base_vol_high_threshold = self.config.get("vol_high_threshold", 0.035)

        # EMA spread thresholds (less volatile, keep fixed)
        self.ema_spread_trending = self.config.get("ema_spread_trending", 0.02)
        self.ema_spread_ranging = self.config.get("ema_spread_ranging", 0.01)

        # Minimum confidence to act
        self.min_confidence = self.config.get("min_confidence", 0.6)

        # History for smoothing and dynamic thresholds
        self.regime_history: List[MarketRegime] = []
        self.max_history = self.config.get("max_history", 5)

        # Rolling history for dynamic threshold calculation
        self.volatility_lookback = self.config.get("volatility_lookback_days", 90)
        self.adx_lookback = self.config.get("adx_lookback_days", 60)
        self.volatility_history: deque = deque(maxlen=self.volatility_lookback)
        self.adx_history: deque = deque(maxlen=self.adx_lookback)
        self.bb_width_history: deque = deque(maxlen=self.volatility_lookback)

        # Current dynamic thresholds
        self.current_thresholds: Dict[str, float] = {}
        self._update_dynamic_thresholds()

        logger.info(
            f"AdaptiveMarketRegimeDetector initialized: "
            f"volatility_lookback={self.volatility_lookback}, "
            f"adx_lookback={self.adx_lookback}"
        )

    def _update_dynamic_thresholds(self) -> Dict[str, float]:
        """
        Calculate dynamic thresholds based on historical percentiles.

        Returns:
            Dictionary of dynamic threshold values
        """
        thresholds = {
            "adx_strong_trend": self.base_adx_strong_trend,
            "adx_weak_trend": self.base_adx_weak_trend,
            "bb_squeeze": self.base_bb_squeeze_threshold,
            "bb_volatile": self.base_bb_volatile_threshold,
            "vol_low": self.base_vol_low_threshold,
            "vol_high": self.base_vol_high_threshold,
        }

        # Adjust based on volatility history if we have enough data
        if len(self.volatility_history) >= 30:
            vol_array = np.array(self.volatility_history)
            vol_50th = np.percentile(vol_array, 50)
            vol_75th = np.percentile(vol_array, 75)
            vol_90th = np.percentile(vol_array, 90)
            vol_10th = np.percentile(vol_array, 10)

            # Dynamic volatility thresholds
            thresholds["vol_low"] = vol_10th * 1.2
            thresholds["vol_high"] = vol_90th * 0.8

            # Adjust BB thresholds based on volatility regime
            if len(self.bb_width_history) >= 30:
                bb_array = np.array(self.bb_width_history)
                bb_50th = np.percentile(bb_array, 50)
                bb_75th = np.percentile(bb_array, 75)

                thresholds["bb_squeeze"] = bb_50th * 0.6
                thresholds["bb_volatile"] = bb_75th * 1.3

        # Adjust ADX thresholds based on ADX history
        if len(self.adx_history) >= 30:
            adx_array = np.array(self.adx_history)
            adx_50th = np.percentile(adx_array, 50)
            adx_75th = np.percentile(adx_array, 75)

            # ADX thresholds adapt to typical market conditions
            thresholds["adx_strong_trend"] = max(20.0, adx_75th * 0.9)
            thresholds["adx_weak_trend"] = max(15.0, adx_50th * 0.8)

        self.current_thresholds = thresholds
        return thresholds

    def detect_regime(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        ema_fast: Optional[float] = None,
        ema_slow: Optional[float] = None,
        adx: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> RegimeMetrics:
        """
        Detect current market regime using adaptive methods.

        Args:
            prices: Array of closing prices
            highs: Array of high prices (optional)
            lows: Array of low prices (optional)
            ema_fast: Fast EMA value (optional)
            ema_slow: Slow EMA value (optional)
            adx: ADX value (optional)
            atr: ATR value (optional)

        Returns:
            RegimeMetrics with detected regime and confidence
        """
        if len(prices) < 20:
            return RegimeMetrics(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                adx=adx or 0.0,
                volatility=0.0,
                bb_width=0.0,
                ema_spread=0.0,
                trend_strength=adx or 0.0,
                dynamic_thresholds=self.current_thresholds,
            )

        # Calculate metrics
        volatility = self._calculate_volatility(prices)
        bb_width = self._calculate_bb_width(prices)
        ema_spread = self._calculate_ema_spread(ema_fast, ema_slow, prices)

        # Update history
        self.volatility_history.append(volatility)
        if adx is not None:
            self.adx_history.append(adx)
        self.bb_width_history.append(bb_width)

        # Update dynamic thresholds
        thresholds = self._update_dynamic_thresholds()

        # Determine regime with confidence scoring
        regime, confidence, indicator_scores = self._classify_regime_with_confidence(
            adx=adx or 0.0,
            volatility=volatility,
            bb_width=bb_width,
            ema_spread=ema_spread,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            thresholds=thresholds,
        )

        # Apply smoothing to avoid rapid regime switches
        regime = self._smooth_regime(regime)

        metrics = RegimeMetrics(
            regime=regime,
            confidence=confidence,
            adx=adx or 0.0,
            volatility=volatility,
            bb_width=bb_width,
            ema_spread=ema_spread,
            trend_strength=adx or 0.0,
            dynamic_thresholds=thresholds,
            indicator_scores=indicator_scores,
        )

        logger.debug(
            f"Regime detected: {regime.value} (confidence: {confidence:.2f}, "
            f"adx_strong={thresholds['adx_strong_trend']:.1f})"
        )

        return metrics

    def _calculate_volatility(self, prices: np.ndarray, window: int = 20) -> float:
        """Calculate rolling volatility."""
        if len(prices) < window:
            return 0.0

        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns[-window:]) * np.sqrt(365))

    def _calculate_bb_width(
        self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0
    ) -> float:
        """Calculate Bollinger Band width as percentage."""
        if len(prices) < period:
            return 0.0

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        if sma == 0:
            return 0.0

        return float((upper_band - lower_band) / sma)

    def _calculate_ema_spread(
        self, ema_fast: Optional[float], ema_slow: Optional[float], prices: np.ndarray
    ) -> float:
        """Calculate EMA spread as percentage."""
        if ema_fast is not None and ema_slow is not None and ema_slow > 0:
            return abs(ema_fast - ema_slow) / ema_slow

        # Calculate from prices if EMAs not provided
        if len(prices) >= 50:
            ema_9 = self._calculate_ema(prices, 9)
            ema_50 = self._calculate_ema(prices, 50)
            if ema_50 > 0:
                return abs(ema_9 - ema_50) / ema_50

        return 0.0

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA for the last value."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0

        multiplier = 2.0 / (period + 1.0)
        ema = float(np.mean(prices[:period]))

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return float(ema)

    def _classify_regime_with_confidence(
        self,
        adx: float,
        volatility: float,
        bb_width: float,
        ema_spread: float,
        ema_fast: Optional[float],
        ema_slow: Optional[float],
        thresholds: Dict[str, float],
    ) -> Tuple[MarketRegime, float, Dict[str, float]]:
        """
        Classify market regime with confidence scoring.

        Returns:
            Tuple of (regime, confidence, indicator_scores)
        """
        scores: Dict[MarketRegime, float] = {}
        indicator_scores: Dict[str, float] = {}

        # ADX scoring
        adx_strong = thresholds["adx_strong_trend"]
        adx_weak = thresholds["adx_weak_trend"]

        if adx > adx_strong and ema_spread > self.ema_spread_trending:
            if ema_fast and ema_slow:
                if ema_fast > ema_slow:
                    scores[MarketRegime.TRENDING_UP] = 0.9
                else:
                    scores[MarketRegime.TRENDING_DOWN] = 0.9
            else:
                scores[MarketRegime.TRENDING_UP] = 0.7
            indicator_scores["adx"] = min(1.0, adx / 40.0)
        elif adx > adx_weak:
            scores[MarketRegime.TRENDING_UP] = 0.5
            indicator_scores["adx"] = 0.5
        else:
            indicator_scores["adx"] = max(0.0, adx / adx_strong)

        # Ranging score
        if adx < adx_weak and ema_spread < self.ema_spread_ranging:
            scores[MarketRegime.RANGING] = 0.8
            indicator_scores["ema_spread"] = 1.0 - (
                ema_spread / self.ema_spread_ranging
            )
        elif bb_width < thresholds["bb_squeeze"]:
            scores[MarketRegime.RANGING] = 0.6
            indicator_scores["bb_squeeze"] = 0.6
        else:
            indicator_scores["ema_spread"] = max(
                0.0, 1.0 - (ema_spread / self.ema_spread_trending)
            )

        # Volatile score
        vol_high = thresholds["vol_high"]
        bb_volatile = thresholds["bb_volatile"]

        if volatility > vol_high or bb_width > bb_volatile:
            scores[MarketRegime.VOLATILE] = 0.85
            indicator_scores["volatility"] = min(1.0, volatility / (vol_high * 1.5))
        else:
            indicator_scores["volatility"] = max(0.0, volatility / vol_high)

        # Quiet score
        vol_low = thresholds["vol_low"]
        bb_squeeze = thresholds["bb_squeeze"]

        if volatility < vol_low and bb_width < bb_squeeze:
            scores[MarketRegime.QUIET] = 0.75
            indicator_scores["quiet"] = 0.75
        else:
            indicator_scores["quiet"] = max(0.0, 1.0 - (volatility / vol_low))

        # Select regime with highest score
        if scores:
            best_regime = max(scores.items(), key=lambda x: x[1])[0]
            base_confidence = scores[best_regime]

            # Calculate confidence based on indicator agreement
            # Higher confidence when multiple indicators agree
            agreeing_indicators = sum(1 for s in scores.values() if s > 0.5)

            if agreeing_indicators == 1:
                # Single indicator - use base confidence
                confidence = base_confidence
            elif agreeing_indicators == 2:
                # Two indicators agree - boost confidence
                confidence = min(1.0, base_confidence * 1.1)
            else:
                # Multiple indicators - high confidence
                confidence = min(1.0, base_confidence * 1.15)

            # Adjust confidence based on how clear the signal is
            score_values = list(scores.values())
            if len(score_values) > 1:
                score_values.sort(reverse=True)
                # Higher confidence if top score is significantly better than second
                if len(score_values) > 1 and score_values[0] > 0:
                    clarity = (score_values[0] - score_values[1]) / score_values[0]
                    confidence = min(1.0, confidence * (0.9 + clarity * 0.2))

            return best_regime, confidence, indicator_scores

        return MarketRegime.UNKNOWN, 0.0, indicator_scores

    def _smooth_regime(self, new_regime: MarketRegime) -> MarketRegime:
        """Apply smoothing to avoid rapid regime switches."""
        self.regime_history.append(new_regime)

        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)

        if len(self.regime_history) < 3:
            return new_regime

        # Require at least 2 out of 3 recent regimes to match for a switch
        from collections import Counter

        regime_counts = Counter(self.regime_history[-3:])
        most_common = regime_counts.most_common(1)[0]

        if most_common[1] >= 2:
            return most_common[0]

        # Stay with previous regime if no clear consensus
        return self.regime_history[-2] if len(self.regime_history) >= 2 else new_regime

    def should_trade_in_regime(self, regime: MarketRegime) -> Tuple[bool, float]:
        """
        Determine if trading should occur in the detected regime.

        Returns:
            Tuple of (should_trade, position_size_modifier)
        """
        regime_settings = {
            MarketRegime.TRENDING_UP: (True, 1.0),
            MarketRegime.TRENDING_DOWN: (True, 1.0),
            MarketRegime.RANGING: (True, 0.35),
            MarketRegime.VOLATILE: (False, 0.0),
            MarketRegime.QUIET: (True, 0.6),
            MarketRegime.UNKNOWN: (False, 0.0),
        }

        return regime_settings.get(regime, (False, 0.0))

    def get_regime_suitability(self, regime: MarketRegime, direction: str) -> float:
        """
        Get how suitable a regime is for a specific trade direction.

        Args:
            regime: Current market regime
            direction: 'long' or 'short'

        Returns:
            Suitability score (0.0 to 1.0)
        """
        suitability_map = {
            MarketRegime.TRENDING_UP: {"long": 1.0, "short": 0.3},
            MarketRegime.TRENDING_DOWN: {"long": 0.3, "short": 1.0},
            MarketRegime.RANGING: {"long": 0.5, "short": 0.5},
            MarketRegime.VOLATILE: {"long": 0.4, "short": 0.4},
            MarketRegime.QUIET: {"long": 0.6, "short": 0.6},
            MarketRegime.UNKNOWN: {"long": 0.0, "short": 0.0},
        }

        return suitability_map.get(regime, {}).get(direction, 0.5)

    def get_adaptive_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get adaptive strategy parameters for the current regime.

        Returns:
            Dictionary with parameter adjustments
        """
        base_params = {
            "atr_multiplier": 2.0,
            "risk_reward_ratio": 2.0,
            "signal_threshold": 0.6,
            "position_size_modifier": 1.0,
            "trend_weight": 0.7,
            "mean_reversion_weight": 0.3,
        }

        regime_params = {
            MarketRegime.TRENDING_UP: {
                "atr_multiplier": 2.5,
                "risk_reward_ratio": 3.0,
                "signal_threshold": 0.65,
                "position_size_modifier": 1.0,
                "trend_weight": 0.9,
                "mean_reversion_weight": 0.1,
            },
            MarketRegime.TRENDING_DOWN: {
                "atr_multiplier": 2.5,
                "risk_reward_ratio": 3.0,
                "signal_threshold": 0.65,
                "position_size_modifier": 1.0,
                "trend_weight": 0.9,
                "mean_reversion_weight": 0.1,
            },
            MarketRegime.RANGING: {
                "atr_multiplier": 1.5,
                "risk_reward_ratio": 1.5,
                "signal_threshold": 0.72,
                "position_size_modifier": 0.35,
                "trend_weight": 0.2,
                "mean_reversion_weight": 0.8,
            },
            MarketRegime.VOLATILE: {
                "atr_multiplier": 3.0,
                "risk_reward_ratio": 2.0,
                "signal_threshold": 0.80,
                "position_size_modifier": 0.0,
                "trend_weight": 0.5,
                "mean_reversion_weight": 0.5,
            },
            MarketRegime.QUIET: {
                "atr_multiplier": 1.8,
                "risk_reward_ratio": 1.8,
                "signal_threshold": 0.70,
                "position_size_modifier": 0.5,
                "trend_weight": 0.5,
                "mean_reversion_weight": 0.5,
            },
        }

        params: Dict[str, Any] = regime_params.get(regime, base_params).copy()
        params["regime"] = regime.value

        return params

    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current dynamic threshold values."""
        return self.current_thresholds.copy()


# Backwards compatibility - alias old name to new class
MarketRegimeDetector = AdaptiveMarketRegimeDetector
