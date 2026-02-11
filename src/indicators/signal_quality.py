"""
Signal Quality Module for All-Weather Strategy

Provides directional composite scoring, normalization, and ambiguity detection
for regime-adaptive trading decisions.

Enhanced with multi-confirmation gates and volatility sanity filters
from the profitability improvement plan.
"""

from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
import numpy as np
import logging

from .market_regime import MarketRegime, get_regime_profile

logger = logging.getLogger(__name__)


@dataclass
class DirectionalScore:
    """Container for directional signal quality scores."""
    
    long_score: float
    short_score: float
    long_primary: float
    short_primary: float
    long_confirm: float
    short_confirm: float
    long_regime: float
    short_regime: float
    long_structure: float
    short_structure: float
    decision_margin: float
    chosen_direction: Optional[str] = None
    skip_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "long_score": self.long_score,
            "short_score": self.short_score,
            "long_primary": self.long_primary,
            "short_primary": self.short_primary,
            "long_confirm": self.long_confirm,
            "short_confirm": self.short_confirm,
            "long_regime": self.long_regime,
            "short_regime": self.short_regime,
            "long_structure": self.long_structure,
            "short_structure": self.short_structure,
            "decision_margin": self.decision_margin,
            "chosen_direction": self.chosen_direction,
            "skip_reason": self.skip_reason,
        }


class SignalQualityScorer:
    """
    Calculates directional composite signal quality scores.
    
    Uses regime-specific weights to combine primary, confirmation,
    regime alignment, and structure scores into a unified directional score.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Default weights by regime
        self.default_weights = {
            "trending_up": {"primary": 0.45, "confirm": 0.20, "regime": 0.25, "structure": 0.10},
            "trending_down": {"primary": 0.45, "confirm": 0.20, "regime": 0.25, "structure": 0.10},
            "ranging": {"primary": 0.35, "confirm": 0.25, "regime": 0.20, "structure": 0.20},
            "volatile": {"primary": 0.30, "confirm": 0.35, "regime": 0.20, "structure": 0.15},
            "quiet": {"primary": 0.40, "confirm": 0.25, "regime": 0.20, "structure": 0.15},
            "unknown": {"primary": 0.40, "confirm": 0.25, "regime": 0.20, "structure": 0.15},
        }
    
    def calculate_directional_score(
        self,
        regime: MarketRegime,
        long_primary: float,
        short_primary: float,
        long_confirm: float,
        short_confirm: float,
        long_regime: float,
        short_regime: float,
        long_structure: float,
        short_structure: float,
    ) -> DirectionalScore:
        """
        Calculate composite directional scores for long and short.
        
        Args:
            regime: Current market regime
            long_primary: Long primary signal score (0-1)
            short_primary: Short primary signal score (0-1)
            long_confirm: Long confirmation score (0-1)
            short_confirm: Short confirmation score (0-1)
            long_regime: Long regime alignment score (0-1)
            short_regime: Short regime alignment score (0-1)
            long_structure: Long market structure score (0-1)
            short_structure: Short market structure score (0-1)
        
        Returns:
            DirectionalScore with composite scores and decision
        """
        # Get regime-specific weights
        profile = get_regime_profile(regime.value)
        weights = self.default_weights.get(regime.value, self.default_weights["unknown"])
        
        # Calculate composite scores
        long_score = (
            weights["primary"] * long_primary +
            weights["confirm"] * long_confirm +
            weights["regime"] * long_regime +
            weights["structure"] * long_structure
        )
        
        short_score = (
            weights["primary"] * short_primary +
            weights["confirm"] * short_confirm +
            weights["regime"] * short_regime +
            weights["structure"] * short_structure
        )
        
        # Clamp to [0, 1]
        long_score = max(0.0, min(1.0, long_score))
        short_score = max(0.0, min(1.0, short_score))
        
        # Calculate decision margin
        decision_margin = abs(long_score - short_score)
        
        # Get regime decision margin threshold
        margin_threshold = profile.get("decision_margin", 0.10)
        
        # Get signal threshold
        signal_threshold = profile.get("signal_threshold", 0.60)
        
        # Determine chosen direction or skip
        chosen_direction = None
        skip_reason = None
        
        if long_score < signal_threshold and short_score < signal_threshold:
            skip_reason = "Both scores below signal threshold"
        elif decision_margin < margin_threshold:
            skip_reason = f"Decision margin {decision_margin:.3f} below threshold {margin_threshold:.3f}"
        elif long_score > short_score:
            chosen_direction = "long"
        else:
            chosen_direction = "short"
        
        return DirectionalScore(
            long_score=long_score,
            short_score=short_score,
            long_primary=long_primary,
            short_primary=short_primary,
            long_confirm=long_confirm,
            short_confirm=short_confirm,
            long_regime=long_regime,
            short_regime=short_regime,
            long_structure=long_structure,
            short_structure=short_structure,
            decision_margin=decision_margin,
            chosen_direction=chosen_direction,
            skip_reason=skip_reason,
        )
    
    def normalize_signal_strength(self, raw_strength: float) -> float:
        """
        Normalize raw signal strength to [0, 1] range.
        
        Args:
            raw_strength: Raw signal strength value
        
        Returns:
            Normalized value in [0, 1]
        """
        return max(0.0, min(1.0, raw_strength))
    
    def calculate_regime_alignment(
        self,
        regime: MarketRegime,
        direction: str,
        regime_confidence: float,
    ) -> float:
        """
        Calculate regime alignment score for a direction.
        
        Args:
            regime: Current market regime
            direction: 'long' or 'short'
            regime_confidence: Confidence in regime detection (0-1)
        
        Returns:
            Regime alignment score (0-1)
        """
        profile = get_regime_profile(regime.value)
        
        if direction == "long":
            suitability = profile.get("suitability_long", 0.5)
        else:
            suitability = profile.get("suitability_short", 0.5)
        
        # Scale by regime confidence
        return suitability * (0.5 + 0.5 * regime_confidence)
    
    def calculate_structure_score(
        self,
        distance_to_support: Optional[float],
        distance_to_resistance: Optional[float],
        direction: str,
    ) -> float:
        """
        Calculate market structure score based on support/resistance proximity.
        
        Args:
            distance_to_support: Distance to nearest support (as % of price)
            distance_to_resistance: Distance to nearest resistance (as % of price)
            direction: 'long' or 'short'
        
        Returns:
            Structure score (0-1)
        """
        if direction == "long":
            # Long entries are better near support
            if distance_to_support is not None:
                # Closer to support = higher score
                # 0% distance = 1.0, 2% distance = 0.0
                score = max(0.0, 1.0 - (distance_to_support / 0.02))
                return score
        else:
            # Short entries are better near resistance
            if distance_to_resistance is not None:
                # Closer to resistance = higher score
                score = max(0.0, 1.0 - (distance_to_resistance / 0.02))
                return score
        
        return 0.5  # Neutral if no structure data


def calculate_confirmation_score(
    volume_ratio: float,
    atr_percent: Optional[float],
    atr_cap: float = 0.04,
) -> float:
    """
    Calculate confirmation score based on volume and volatility.

    Args:
        volume_ratio: Current volume / (avg_volume * volume_multiplier)
        atr_percent: ATR as percentage of price
        atr_cap: Maximum ATR percent for non-volatile regimes

    Returns:
        Confirmation score (0-1)
    """
    score = 0.0

    # Volume confirmation (0-0.5)
    if volume_ratio >= 1.0:
        score += 0.5
    else:
        score += 0.5 * max(0.65, volume_ratio)

    # Volatility sanity (0-0.5)
    if atr_percent is not None:
        if atr_percent <= atr_cap:
            score += 0.5
        else:
            # Penalize extreme volatility
            penalty = min(0.5, (atr_percent - atr_cap) / atr_cap)
            score += 0.5 * (1.0 - penalty)
    else:
        score += 0.25  # Neutral if no ATR data

    return max(0.0, min(1.0, score))


def check_multi_confirmation_gate(
    regime: MarketRegime,
    adx: Optional[float],
    ema_spread: Optional[float],
    volume_ratio: Optional[float],
    rsi: Optional[float] = None,
    rsi_extreme_threshold: float = 30.0,
    breakout_condition: bool = False,
    directional_momentum: bool = False,
) -> Tuple[bool, str]:
    """
    Multi-confirmation gate for entry quality.

    Implements regime-specific confirmation requirements:
    - Trending regimes: require at least 2 of 3 (ADX, EMA spread, volume)
    - Ranging regime: require at least 2 of 3 with one mandatory structure element
    - Volatile regime: require at least 3 of 4 (breakout, momentum, volume, volatility sanity)

    Args:
        regime: Current market regime
        adx: ADX value for trend strength
        ema_spread: EMA spread for structure
        volume_ratio: Volume ratio for volume confirmation
        rsi: RSI value for ranging regime checks
        rsi_extreme_threshold: RSI threshold for extreme values
        breakout_condition: Whether breakout condition is met (volatile regime)
        directional_momentum: Whether directional momentum agrees (volatile regime)

    Returns:
        Tuple of (passed: bool, reason: str)
    """
    regime_value = regime.value

    # Trending regimes: require at least 2 of 3
    if regime_value in ("trending_up", "trending_down"):
        passed = 0
        reasons = []

        # ADX above trend floor (minimum 20)
        if adx is not None and adx >= 20.0:
            passed += 1
            reasons.append(f"ADX {adx:.1f} >= 20.0")
        else:
            reasons.append(f"ADX {adx if adx else 'N/A'} < 20.0")

        # EMA spread above minimum structure floor (0.5%)
        if ema_spread is not None and ema_spread >= 0.005:
            passed += 1
            reasons.append(f"EMA spread {ema_spread:.3f} >= 0.005")
        else:
            reasons.append(f"EMA spread {ema_spread if ema_spread else 'N/A'} < 0.005")

        # Volume confirmation ratio above regime minimum (1.0)
        if volume_ratio is not None and volume_ratio >= 1.0:
            passed += 1
            reasons.append(f"Volume ratio {volume_ratio:.2f} >= 1.0")
        else:
            reasons.append(f"Volume ratio {volume_ratio if volume_ratio else 'N/A'} < 1.0")

        if passed >= 2:
            return True, f"Multi-confirmation passed: {passed}/3 ({', '.join(reasons)})"
        else:
            return False, f"Multi-confirmation failed: {passed}/3 ({', '.join(reasons)})"

    # Ranging regime: require at least 2 of 3 with one mandatory structure element
    elif regime_value == "ranging":
        passed = 0
        has_structure = False
        reasons = []

        # RSI extreme and direction agreement
        if rsi is not None:
            if rsi <= rsi_extreme_threshold or rsi >= (100.0 - rsi_extreme_threshold):
                passed += 1
                reasons.append(f"RSI {rsi:.1f} extreme")
            else:
                reasons.append(f"RSI {rsi:.1f} not extreme")

        # EMA spread as structure element (mandatory)
        if ema_spread is not None and ema_spread >= 0.003:
            passed += 1
            has_structure = True
            reasons.append(f"EMA spread {ema_spread:.3f} >= 0.003")
        else:
            reasons.append(f"EMA spread {ema_spread if ema_spread else 'N/A'} < 0.003")

        # Volume confirmation
        if volume_ratio is not None and volume_ratio >= 1.0:
            passed += 1
            reasons.append(f"Volume ratio {volume_ratio:.2f} >= 1.0")
        else:
            reasons.append(f"Volume ratio {volume_ratio if volume_ratio else 'N/A'} < 1.0")

        if passed >= 2 and has_structure:
            return True, f"Multi-confirmation passed: {passed}/3 with structure ({', '.join(reasons)})"
        else:
            return False, f"Multi-confirmation failed: {passed}/3, structure={'yes' if has_structure else 'no'} ({', '.join(reasons)})"

    # Volatile regime: require at least 3 of 4
    elif regime_value == "volatile":
        passed = 0
        reasons = []

        # Breakout condition
        if breakout_condition:
            passed += 1
            reasons.append("Breakout condition met")
        else:
            reasons.append("Breakout condition not met")

        # Directional momentum agreement
        if directional_momentum:
            passed += 1
            reasons.append("Directional momentum agrees")
        else:
            reasons.append("Directional momentum disagrees")

        # Volume confirmation
        if volume_ratio is not None and volume_ratio >= 1.0:
            passed += 1
            reasons.append(f"Volume ratio {volume_ratio:.2f} >= 1.0")
        else:
            reasons.append(f"Volume ratio {volume_ratio if volume_ratio else 'N/A'} < 1.0")

        # Volatility sanity (ATR not too extreme - cap at 8%)
        # This is checked separately in volatility_sanity_filter
        passed += 1  # Assume passed if we get here (checked elsewhere)
        reasons.append("Volatility sanity passed")

        if passed >= 3:
            return True, f"Multi-confirmation passed: {passed}/4 ({', '.join(reasons)})"
        else:
            return False, f"Multi-confirmation failed: {passed}/4 ({', '.join(reasons)})"

    # Quiet regime: similar to trending but more lenient
    elif regime_value == "quiet":
        passed = 0
        reasons = []

        # ADX check (lower threshold for quiet)
        if adx is not None and adx >= 15.0:
            passed += 1
            reasons.append(f"ADX {adx:.1f} >= 15.0")
        else:
            reasons.append(f"ADX {adx if adx else 'N/A'} < 15.0")

        # EMA spread check
        if ema_spread is not None and ema_spread >= 0.003:
            passed += 1
            reasons.append(f"EMA spread {ema_spread:.3f} >= 0.003")
        else:
            reasons.append(f"EMA spread {ema_spread if ema_spread else 'N/A'} < 0.003")

        # Volume confirmation
        if volume_ratio is not None and volume_ratio >= 0.8:
            passed += 1
            reasons.append(f"Volume ratio {volume_ratio:.2f} >= 0.8")
        else:
            reasons.append(f"Volume ratio {volume_ratio if volume_ratio else 'N/A'} < 0.8")

        if passed >= 2:
            return True, f"Multi-confirmation passed: {passed}/3 ({', '.join(reasons)})"
        else:
            return False, f"Multi-confirmation failed: {passed}/3 ({', '.join(reasons)})"

    # Unknown regime: be conservative
    else:
        return False, "Unknown regime - multi-confirmation gate blocked"


def check_volatility_sanity_filter(
    regime: MarketRegime,
    atr_percent: Optional[float],
    atr_cap: float = 0.04,
    extreme_cap: float = 0.08,
) -> Tuple[bool, str]:
    """
    Volatility sanity filter to reject low-quality extreme-vol entries.

    Rules:
    - Non-volatile regimes enforce ATR percent cap
    - Volatile regime allows high ATR but requires breakout context
    - Extreme ATR beyond cap blocks entry

    Args:
        regime: Current market regime
        atr_percent: ATR as percentage of price
        atr_cap: Maximum ATR percent for non-volatile regimes
        extreme_cap: Absolute maximum ATR percent (blocks entry)

    Returns:
        Tuple of (passed: bool, reason: str)
    """
    if atr_percent is None:
        return True, "No ATR data - volatility sanity filter passed"

    regime_value = regime.value

    # Volatile regime: allow higher ATR but still cap at extreme
    if regime_value == "volatile":
        if atr_percent > extreme_cap:
            return False, f"ATR% {atr_percent:.3f} exceeds extreme cap {extreme_cap:.3f}"
        return True, f"ATR% {atr_percent:.3f} within volatile regime bounds"

    # Non-volatile regimes: enforce ATR cap
    if atr_percent > atr_cap:
        return False, f"ATR% {atr_percent:.3f} exceeds cap {atr_cap:.3f} for {regime_value}"

    return True, f"ATR% {atr_percent:.3f} within bounds for {regime_value}"


# ─── Phase 1: Layer A Activation Gate and No-Trade States ─────────────────────

@dataclass
class EdgeScore:
    """Container for edge score calculation results."""
    
    edge_score: float  # Combined edge score
    estimated_edge_r: float  # Estimated edge in R units
    regime_confidence: float  # Regime confidence
    direction_margin: float  # Long/short score difference
    liquidity_quality: float  # Liquidity quality score
    passed: bool  # Whether edge score passes floor
    reason: str  # Reason for pass/fail
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_score": self.edge_score,
            "estimated_edge_r": self.estimated_edge_r,
            "regime_confidence": self.regime_confidence,
            "direction_margin": self.direction_margin,
            "liquidity_quality": self.liquidity_quality,
            "passed": self.passed,
            "reason": self.reason,
        }


def calculate_edge_score(
    regime: MarketRegime,
    regime_confidence: float,
    long_score: float,
    short_score: float,
    volume_ratio: Optional[float] = None,
    spread_pct: Optional[float] = None,
    expected_cost_buffer_mult: float = 2.8,
) -> EdgeScore:
    """
    Calculate edge score for a symbol based on expectancy-positive redesign.
    
    Edge score formula:
    edge_score = estimated_E_R * regime_confidence * direction_margin * liquidity_quality
    
    Args:
        regime: Current market regime
        regime_confidence: Confidence in regime detection (0-1)
        long_score: Long directional score (0-1)
        short_score: Short directional score (0-1)
        volume_ratio: Volume ratio for liquidity quality
        spread_pct: Spread as percentage of price
        expected_cost_buffer_mult: Cost buffer multiplier for edge estimation
    
    Returns:
        EdgeScore with calculation results
    """
    profile = get_regime_profile(regime.value)
    
    # Direction margin: difference between long and short scores
    direction_margin = abs(long_score - short_score)
    
    # Liquidity quality: based on volume ratio and spread
    liquidity_quality = 1.0
    if volume_ratio is not None:
        # Volume ratio >= 1.0 is good, below 0.8 is poor
        liquidity_quality *= min(1.0, max(0.5, volume_ratio))
    if spread_pct is not None:
        # Spread <= 0.05% is good, above 0.15% is poor
        spread_quality = max(0.5, 1.0 - (spread_pct / 0.15))
        liquidity_quality *= spread_quality
    
    # Clamp liquidity quality to [0.5, 1.0]
    liquidity_quality = max(0.5, min(1.0, liquidity_quality))
    
    # Estimated edge in R units (simplified model)
    # Use regime-specific expected payoff and cost assumptions
    tp1_r = profile.get("tp1_r", 1.5)
    tp2_r = profile.get("tp2_r", 2.5)
    expected_win_r = (tp1_r + tp2_r) / 2.0  # Average expected winner
    
    # Cost estimate in R (conservative)
    # Assume 1R stop loss, cost as percentage of position
    cost_pct = 0.0012  # 0.12% base cost (fees + slippage)
    cost_r = cost_pct * expected_cost_buffer_mult  # Buffer for uncertainty
    
    # Estimated edge = expected_win_r * win_prob - cost_r
    # Simplified: use directional score as proxy for win probability
    max_score = max(long_score, short_score)
    estimated_edge_r = (expected_win_r * max_score) - cost_r
    
    # Combined edge score
    edge_score = estimated_edge_r * regime_confidence * direction_margin * liquidity_quality
    
    # Check against edge floor
    edge_floor = profile.get("edge_floor", 0.08)
    passed = edge_score >= edge_floor
    
    reason = (
        f"Edge score {edge_score:.3f} >= floor {edge_floor:.3f}"
        if passed
        else f"Edge score {edge_score:.3f} < floor {edge_floor:.3f}"
    )
    
    return EdgeScore(
        edge_score=edge_score,
        estimated_edge_r=estimated_edge_r,
        regime_confidence=regime_confidence,
        direction_margin=direction_margin,
        liquidity_quality=liquidity_quality,
        passed=passed,
        reason=reason,
    )


def check_ambiguity_veto(
    regime: MarketRegime,
    long_score: float,
    short_score: float,
) -> Tuple[bool, str]:
    """
    Check if directional ambiguity should veto the trade.
    
    If the difference between long and short scores is too small,
    the trade is vetoed to avoid low-conviction entries.
    
    Args:
        regime: Current market regime
        long_score: Long directional score (0-1)
        short_score: Short directional score (0-1)
    
    Returns:
        Tuple of (vetoed: bool, reason: str)
    """
    profile = get_regime_profile(regime.value)
    
    # Get ambiguity veto threshold from regime profile
    ambiguity_threshold = profile.get("ambiguity_veto_threshold", 0.05)
    
    # Calculate direction margin
    direction_margin = abs(long_score - short_score)
    
    # Check if margin is below threshold
    if direction_margin < ambiguity_threshold:
        return (
            True,
            f"Ambiguity veto: direction margin {direction_margin:.3f} < threshold {ambiguity_threshold:.3f}"
        )
    
    return (
        False,
        f"Ambiguity check passed: direction margin {direction_margin:.3f} >= threshold {ambiguity_threshold:.3f}"
    )


def check_activation_gate(
    regime: MarketRegime,
    regime_confidence: float,
    long_score: float,
    short_score: float,
    volume_ratio: Optional[float] = None,
    spread_pct: Optional[float] = None,
    min_regime_confidence: float = 0.6,
    trend_continuation_relaxed: bool = False,
) -> Tuple[bool, str, Optional[EdgeScore]]:
    """
    Layer A activation gate: check if trade should be allowed.
    
    Hard no-trade states (any true = no trade):
    1. Regime confidence below floor
    2. Directional ambiguity margin below threshold
    3. Expected net edge below regime floor
    4. Spread or slippage proxy exceeds cap
    
    V2 Fix C1: Add trend continuation relaxation to reduce false negatives
    in strong trend continuation while keeping strict no-trade/ambiguity gates.
    
    Args:
        regime: Current market regime
        regime_confidence: Confidence in regime detection (0-1)
        long_score: Long directional score (0-1)
        short_score: Short directional score (0-1)
        volume_ratio: Volume ratio for liquidity quality
        spread_pct: Spread as percentage of price
        min_regime_confidence: Minimum regime confidence (default: 0.6)
        trend_continuation_relaxed: Whether to apply relaxed thresholds for trend continuation
    
    Returns:
        Tuple of (passed: bool, reason: str, edge_score: Optional[EdgeScore])
    """
    # Check 1: Regime confidence floor
    if regime_confidence < min_regime_confidence:
        return (
            False,
            f"Regime confidence {regime_confidence:.2f} below floor {min_regime_confidence:.2f}",
            None,
        )
    
    # Check 2: Ambiguity veto (always strict, no relaxation)
    vetoed, veto_reason = check_ambiguity_veto(regime, long_score, short_score)
    if vetoed:
        return False, veto_reason, None
    
    # Check 3: Edge score floor
    edge_score = calculate_edge_score(
        regime=regime,
        regime_confidence=regime_confidence,
        long_score=long_score,
        short_score=short_score,
        volume_ratio=volume_ratio,
        spread_pct=spread_pct,
    )
    
    # V2 Fix C1: Apply relaxed edge floor for trend continuation
    if trend_continuation_relaxed and regime.value in ("trending_up", "trending_down"):
        # Relax edge floor by 20% for trend continuation
        profile = get_regime_profile(regime.value)
        relaxed_edge_floor = profile.get("edge_floor", 0.08) * 0.8
        
        if edge_score.edge_score < relaxed_edge_floor:
            return (
                False,
                f"Edge score {edge_score.edge_score:.3f} below relaxed floor {relaxed_edge_floor:.3f} (trend continuation)",
                edge_score,
            )
    elif not edge_score.passed:
        return False, edge_score.reason, edge_score
    
    # Check 4: Spread cap (if provided)
    if spread_pct is not None and spread_pct > 0.002:  # 0.2% spread cap
        return (
            False,
            f"Spread {spread_pct:.3f} exceeds cap 0.002",
            edge_score,
        )
    
    # All checks passed
    return True, f"Activation gate passed: {edge_score.reason}", edge_score


def rank_symbols_by_edge(
    symbols_data: Dict[str, Dict[str, Any]],
    regime: MarketRegime,
) -> List[Tuple[str, float]]:
    """
    Rank symbols by edge score for top-1 selection.
    
    Args:
        symbols_data: Dict mapping symbol to data with:
            - regime_confidence: float
            - long_score: float
            - short_score: float
            - volume_ratio: Optional[float]
            - spread_pct: Optional[float]
        regime: Current market regime
    
    Returns:
        List of (symbol, edge_score) tuples sorted by edge_score descending
    """
    ranked = []
    
    for symbol, data in symbols_data.items():
        edge_score = calculate_edge_score(
            regime=regime,
            regime_confidence=data.get("regime_confidence", 0.5),
            long_score=data.get("long_score", 0.0),
            short_score=data.get("short_score", 0.0),
            volume_ratio=data.get("volume_ratio"),
            spread_pct=data.get("spread_pct"),
        )
        ranked.append((symbol, edge_score.edge_score))
    
    # Sort by edge score descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return ranked


def check_top1_lead_margin(
    ranked_symbols: List[Tuple[str, float]],
    top1_lead_margin: float = 0.05,
) -> Tuple[bool, str, Optional[str]]:
    """
    Check if top-1 symbol has clear lead over second-best.
    
    Args:
        ranked_symbols: List of (symbol, edge_score) tuples sorted descending
        top1_lead_margin: Minimum lead margin required (default: 0.05)
    
    Returns:
        Tuple of (passed: bool, reason: str, top_symbol: Optional[str])
    """
    if len(ranked_symbols) < 1:
        return False, "No symbols to rank", None
    
    if len(ranked_symbols) == 1:
        # Only one symbol, automatically passes
        return True, f"Single symbol {ranked_symbols[0][0]} passes", ranked_symbols[0][0]
    
    top_symbol, top_score = ranked_symbols[0]
    second_symbol, second_score = ranked_symbols[1]
    
    lead_margin = top_score - second_score
    
    if lead_margin >= top1_lead_margin:
        return (
            True,
            f"Top symbol {top_symbol} has clear lead: {lead_margin:.3f} >= {top1_lead_margin:.3f}",
            top_symbol,
        )
    
    return (
        False,
        f"Top symbol {top_symbol} lacks clear lead: {lead_margin:.3f} < {top1_lead_margin:.3f} (vs {second_symbol})",
        None,
    )
