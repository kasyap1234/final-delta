"""
Enhanced Market Regime Detection Module with Dynamic Thresholds

Detects market regimes (trending, ranging, volatile, quiet) using adaptive methods:
1. Dynamic thresholds based on rolling volatility percentiles
2. Enhanced confidence scoring using multiple indicator agreement
3. Hurst exponent for trending vs mean-reverting distinction
4. Hysteresis thresholds to prevent regime flickering
5. Centralized regime strategy profiles for the strategy switcher

This module enables truly adaptive trading strategies that adjust to current market conditions.
"""

from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─── V3 Operating Modes ────────────────────────────────────────────────────────

class V3OperatingMode(Enum):
    """V3 operating modes for the strategy.

    RISK_ON_BULL_CAPTURE: Capture persistent bull trends with long-biased active trading
    RISK_ON_SELECTIVE: Trade only high-conviction setups with symmetric but selective entries
    PASSIVE_DEFER: Avoid low-edge overtrading while preserving reactivation path
    RISK_OFF_NO_TRADE: Protect capital in adverse conditions with no new entries
    """

    RISK_ON_BULL_CAPTURE = "risk_on_bull_capture"
    RISK_ON_SELECTIVE = "risk_on_selective"
    PASSIVE_DEFER = "passive_defer"
    RISK_OFF_NO_TRADE = "risk_off_no_trade"


# ─── Centralized Regime Strategy Profiles ───────────────────────────────────
# Each regime gets its own parameter set. The strategy engine consults these
# instead of having scattered regime logic. This is the "strategy switcher".

REGIME_PROFILES: Dict[str, Dict[str, Any]] = {
    "trending_up": {
        # Conservative signal thresholds (raised from 0.62 to 0.72)
        "signal_threshold": 0.72,
        # Wider decision margin (raised from 0.08 to 0.10)
        "decision_margin": 0.10,
        # Wider ATR stop multiplier (raised from 2.8 to 3.8)
        "atr_multiplier": 3.8,
        # Stop percent floor/cap for ATR clamping
        "stop_pct_floor": 0.018,  # 1.8%
        "stop_pct_cap": 0.060,   # 6.0%
        # V2 Fix B1: Improved TP ladder with larger R multiples for better winner retention
        "rr_ratio": 2.4,
        "tp1_r": 1.8,  # Raised from 1.5 to 1.8 for better first target
        "tp1_scale": 0.20,  # Reduced from 0.25 to 0.20 to keep more position for runner
        "tp2_r": 3.5,  # Raised from 3.2 to 3.5 for better second target
        "tp2_scale": 0.30,  # Reduced from 0.35 to 0.30 to keep more position for runner
        "use_trailing_tp": True,
        "trailing_activation_atr": 1.2,  # Raised from 1.0 to 1.2 for later activation
        "trailing_distance_atr": 2.0,  # Raised from 1.8 to 2.0 for looser trailing
        "profit_retracement_pct": 0.50,  # Raised from 0.40 to 0.50 to allow more retracement
        "position_size_mod": 1.00,
        # Increased cooldown (raised from 6 to 10)
        "cooldown_candles": 10,
        # Reduced daily trades (from 3 to 2)
        "max_trades_per_day": 2,
        "time_limit_hours": 72,
        "volume_multiplier": 1.10,
        "trend_weight": 0.85,
        "mr_weight": 0.15,
        "breakeven_atr": 0.9,
        "suitability_long": 1.0,
        "suitability_short": 0.65,
        "min_progress_r": 0.5,
        # Minimum hold time (new field)
        "min_hold_candles": 10,
        # Cost-aware edge gate (new field)
        "min_reward_cost_ratio": 2.5,
        # Phase 1: Edge floor for activation gate (expectancy-positive redesign)
        "edge_floor": 0.10,  # Minimum edge in R units to enter trend trades
        # Phase 1: Ambiguity veto threshold
        "ambiguity_veto_threshold": 0.05,  # Minimum long/short score difference
        # V2 Fix B2: Break-even progression after TP1
        "breakeven_after_tp1": True,  # Move stop to breakeven after TP1 hit
        "breakeven_buffer_atr": 0.3,  # Buffer above breakeven to account for fees
        # ─── V3: Directional Policy and Tactical Short Exception ───────────────
        "v3_long_bias": True,  # Long-biased in strong uptrend
        "v3_tactical_short_allowed": True,  # Tactical shorts allowed as exceptions
        "v3_tactical_short_edge_mult": 1.8,  # Edge floor multiplier for tactical shorts (1.5-2.0)
        "v3_tactical_short_size_cap": 0.25,  # Size cap for tactical shorts (0.20-0.35)
        "v3_tactical_short_hold_cap": 8,  # Hold cap in candles for tactical shorts (6-16)
        "v3_tactical_short_no_pyramiding": True,  # No pyramiding of tactical shorts
        # ─── V3: Risk-on Bull-Capture Activation ───────────────────────────────
        "v3_risk_on_bull_capture": True,  # Enable risk-on bull-capture mode
        "v3_regime_confidence_min": 0.72,  # Minimum regime confidence (0.68-0.80)
        "v3_benchmark_trend_strength_min": 0.68,  # Minimum benchmark trend strength (0.60-0.80)
        "v3_drawdown_max_for_risk_on": 0.08,  # Maximum drawdown for risk-on mode (8%)
        # ─── V3: Payoff Mechanics ───────────────────────────────────────────────────
        "v3_payoff_enabled": True,  # Enable V3 payoff mechanics
        "v3_hybrid_stop_atr_mult": 1.5,  # Hybrid stop ATR multiplier (1.2-2.0)
        "v3_stop_pct_floor": 0.015,  # Stop percent floor (1.5%)
        "v3_stop_pct_cap": 0.055,  # Stop percent cap (5.5%)
        "v3_lockin_buffer_atr": 0.5,  # Lock-in buffer after TP1 (0.3-0.8 ATR)
        "v3_runner_retention_enabled": True,  # Enable runner retention after TP2
    },
    "trending_down": {
        # Conservative signal thresholds (raised from 0.62 to 0.72)
        "signal_threshold": 0.72,
        # Wider decision margin (raised from 0.08 to 0.10)
        "decision_margin": 0.10,
        # Wider ATR stop multiplier (raised from 2.8 to 3.8)
        "atr_multiplier": 3.8,
        # Stop percent floor/cap for ATR clamping
        "stop_pct_floor": 0.018,  # 1.8%
        "stop_pct_cap": 0.060,   # 6.0%
        # V2 Fix B1: Improved TP ladder with larger R multiples for better winner retention
        "rr_ratio": 2.4,
        "tp1_r": 1.8,  # Raised from 1.5 to 1.8 for better first target
        "tp1_scale": 0.20,  # Reduced from 0.25 to 0.20 to keep more position for runner
        "tp2_r": 3.5,  # Raised from 3.2 to 3.5 for better second target
        "tp2_scale": 0.30,  # Reduced from 0.35 to 0.30 to keep more position for runner
        "use_trailing_tp": True,
        "trailing_activation_atr": 1.2,  # Raised from 1.0 to 1.2 for later activation
        "trailing_distance_atr": 2.0,  # Raised from 1.8 to 2.0 for looser trailing
        "profit_retracement_pct": 0.50,  # Raised from 0.40 to 0.50 to allow more retracement
        "position_size_mod": 1.00,
        # Increased cooldown (raised from 6 to 10)
        "cooldown_candles": 10,
        # Reduced daily trades (from 3 to 2)
        "max_trades_per_day": 2,
        "time_limit_hours": 72,
        "volume_multiplier": 1.10,
        # ─── V3: Directional Policy (Short-biased in downtrend) ─────────────────
        "v3_long_bias": False,  # Short-biased in strong downtrend
        "v3_tactical_short_allowed": True,  # Tactical shorts allowed as exceptions
        "v3_tactical_short_edge_mult": 1.8,  # Edge floor multiplier for tactical shorts (1.5-2.0)
        "v3_tactical_short_size_cap": 0.25,  # Size cap for tactical shorts (0.20-0.35)
        "v3_tactical_short_hold_cap": 8,  # Hold cap in candles for tactical shorts (6-16)
        "v3_tactical_short_no_pyramiding": True,  # No pyramiding of tactical shorts
        # ─── V3: Risk-on Bull-Capture Activation (not applicable for downtrend) ───
        "v3_risk_on_bull_capture": False,  # Not applicable for downtrend
        "v3_regime_confidence_min": 0.72,  # Minimum regime confidence (0.68-0.80)
        "v3_benchmark_trend_strength_min": 0.68,  # Minimum benchmark trend strength (0.60-0.80)
        "v3_drawdown_max_for_risk_on": 0.08,  # Maximum drawdown for risk-on mode (8%)
        # ─── V3: Payoff Mechanics ───────────────────────────────────────────────────
        "v3_payoff_enabled": True,  # Enable V3 payoff mechanics
        "v3_hybrid_stop_atr_mult": 1.5,  # Hybrid stop ATR multiplier (1.2-2.0)
        "v3_stop_pct_floor": 0.015,  # Stop percent floor (1.5%)
        "v3_stop_pct_cap": 0.055,  # Stop percent cap (5.5%)
        "v3_lockin_buffer_atr": 0.5,  # Lock-in buffer after TP1 (0.3-0.8 ATR)
        "v3_runner_retention_enabled": True,  # Enable runner retention after TP2
        "trend_weight": 0.85,
        "mr_weight": 0.15,
        "breakeven_atr": 0.9,
        "suitability_long": 0.65,
        "suitability_short": 1.0,
        "min_progress_r": 0.5,
        # Minimum hold time (new field)
        "min_hold_candles": 10,
        # Cost-aware edge gate (new field)
        "min_reward_cost_ratio": 2.5,
        # Phase 1: Edge floor for activation gate (expectancy-positive redesign)
        "edge_floor": 0.10,  # Minimum edge in R units to enter trend trades
        # Phase 1: Ambiguity veto threshold
        "ambiguity_veto_threshold": 0.05,  # Minimum long/short score difference
        # V2 Fix B2: Break-even progression after TP1
        "breakeven_after_tp1": True,  # Move stop to breakeven after TP1 hit
        "breakeven_buffer_atr": 0.3,  # Buffer above breakeven to account for fees
    },
    "ranging": {
        # Higher signal threshold for ranging (raised from 0.70 to 0.78)
        "signal_threshold": 0.78,
        # Wider decision margin (raised from 0.10 to 0.12)
        "decision_margin": 0.12,
        # Wider ATR stop multiplier (raised from 2.0 to 2.9)
        "atr_multiplier": 2.9,
        # Stop percent floor/cap for ATR clamping
        "stop_pct_floor": 0.016,  # 1.6%
        "stop_pct_cap": 0.050,   # 5.0%
        # Improved TP ladder with larger R multiples
        "rr_ratio": 1.6,
        "tp1_r": 1.2,  # Raised from 0.8
        "tp1_scale": 0.30,  # 25-35% range
        "tp2_r": 2.4,  # Raised from 1.6
        "tp2_scale": 0.50,  # 45-55% range
        "use_trailing_tp": True,
        "trailing_activation_atr": 0.6,
        "trailing_distance_atr": 1.2,
        "profit_retracement_pct": 0.35,
        "position_size_mod": 0.35,
        # Increased cooldown (raised from 12 to 20)
        "cooldown_candles": 20,
        # Reduced daily trades (from 2 to 1)
        "max_trades_per_day": 1,
        "time_limit_hours": 20,
        "volume_multiplier": 1.30,
        "trend_weight": 0.25,
        "mr_weight": 0.75,
        "breakeven_atr": 0.6,
        "suitability_long": 0.75,
        "suitability_short": 0.75,
        "min_progress_r": 0.3,
        # Minimum hold time (new field)
        "min_hold_candles": 6,
        # Cost-aware edge gate (new field)
        "min_reward_cost_ratio": 2.8,
        # Phase 1: Edge floor for activation gate (expectancy-positive redesign)
        "edge_floor": 0.08,  # Minimum edge in R units to enter range trades
        # Phase 1: Ambiguity veto threshold
        "ambiguity_veto_threshold": 0.06,  # Minimum long/short score difference
        # ─── V3: Directional Policy (Symmetric in ranging) ───────────────────────
        "v3_long_bias": False,  # Symmetric in ranging
        "v3_tactical_short_allowed": True,  # Tactical shorts allowed as exceptions
        "v3_tactical_short_edge_mult": 1.8,  # Edge floor multiplier for tactical shorts (1.5-2.0)
        "v3_tactical_short_size_cap": 0.25,  # Size cap for tactical shorts (0.20-0.35)
        "v3_tactical_short_hold_cap": 8,  # Hold cap in candles for tactical shorts (6-16)
        "v3_tactical_short_no_pyramiding": True,  # No pyramiding of tactical shorts
        # ─── V3: Risk-on Bull-Capture Activation (not applicable for ranging) ────
        "v3_risk_on_bull_capture": False,  # Not applicable for ranging
        "v3_regime_confidence_min": 0.72,  # Minimum regime confidence (0.68-0.80)
        "v3_benchmark_trend_strength_min": 0.68,  # Minimum benchmark trend strength (0.60-0.80)
        "v3_drawdown_max_for_risk_on": 0.08,  # Maximum drawdown for risk-on mode (8%)
        # ─── V3: Payoff Mechanics ───────────────────────────────────────────────────
        "v3_payoff_enabled": True,  # Enable V3 payoff mechanics
        "v3_hybrid_stop_atr_mult": 1.5,  # Hybrid stop ATR multiplier (1.2-2.0)
        "v3_stop_pct_floor": 0.015,  # Stop percent floor (1.5%)
        "v3_stop_pct_cap": 0.055,  # Stop percent cap (5.5%)
        "v3_lockin_buffer_atr": 0.5,  # Lock-in buffer after TP1 (0.3-0.8 ATR)
        "v3_runner_retention_enabled": True,  # Enable runner retention after TP2
    },
    "volatile": {
        # Highest signal threshold (raised from 0.78 to 0.86)
        "signal_threshold": 0.86,
        # Wider decision margin (raised from 0.12 to 0.15)
        "decision_margin": 0.15,
        # Widest ATR stop multiplier (raised from 3.6 to 4.6)
        "atr_multiplier": 4.6,
        # Stop percent floor/cap for ATR clamping
        "stop_pct_floor": 0.025,  # 2.5%
        "stop_pct_cap": 0.080,   # 8.0%
        # Improved TP ladder with larger R multiples
        "rr_ratio": 2.2,
        "tp1_r": 1.6,  # Raised from 1.0
        "tp1_scale": 0.35,  # 30-40% range
        "tp2_r": 3.5,  # Raised from 2.0
        "tp2_scale": 0.40,  # 35-45% range
        "use_trailing_tp": True,
        "trailing_activation_atr": 1.4,
        "trailing_distance_atr": 2.6,
        "profit_retracement_pct": 0.30,
        "position_size_mod": 0.20,
        # Increased cooldown (raised from 20 to 30)
        "cooldown_candles": 30,
        # Single trade per day
        "max_trades_per_day": 1,
        "time_limit_hours": 10,
        "volume_multiplier": 1.60,
        "trend_weight": 0.55,
        "mr_weight": 0.45,
        "breakeven_atr": 1.2,
        "suitability_long": 0.55,
        "suitability_short": 0.55,
        "min_progress_r": 0.7,
        # Minimum hold time (new field)
        "min_hold_candles": 5,
        # Cost-aware edge gate (new field)
        "min_reward_cost_ratio": 3.2,
        # Phase 1: Edge floor for activation gate (expectancy-positive redesign)
        "edge_floor": 0.14,  # Minimum edge in R units to enter volatile trades
        # Phase 1: Ambiguity veto threshold
        "ambiguity_veto_threshold": 0.07,  # Minimum long/short score difference
        # ─── V3: Directional Policy (Symmetric in volatile) ──────────────────────
        "v3_long_bias": False,  # Symmetric in volatile
        "v3_tactical_short_allowed": True,  # Tactical shorts allowed as exceptions
        "v3_tactical_short_edge_mult": 1.8,  # Edge floor multiplier for tactical shorts (1.5-2.0)
        "v3_tactical_short_size_cap": 0.25,  # Size cap for tactical shorts (0.20-0.35)
        "v3_tactical_short_hold_cap": 8,  # Hold cap in candles for tactical shorts (6-16)
        "v3_tactical_short_no_pyramiding": True,  # No pyramiding of tactical shorts
        # ─── V3: Risk-on Bull-Capture Activation (not applicable for volatile) ───
        "v3_risk_on_bull_capture": False,  # Not applicable for volatile
        "v3_regime_confidence_min": 0.72,  # Minimum regime confidence (0.68-0.80)
        "v3_benchmark_trend_strength_min": 0.68,  # Minimum benchmark trend strength (0.60-0.80)
        "v3_drawdown_max_for_risk_on": 0.08,  # Maximum drawdown for risk-on mode (8%)
        # ─── V3: Payoff Mechanics ───────────────────────────────────────────────────
        "v3_payoff_enabled": True,  # Enable V3 payoff mechanics
        "v3_hybrid_stop_atr_mult": 1.5,  # Hybrid stop ATR multiplier (1.2-2.0)
        "v3_stop_pct_floor": 0.015,  # Stop percent floor (1.5%)
        "v3_stop_pct_cap": 0.055,  # Stop percent cap (5.5%)
        "v3_lockin_buffer_atr": 0.5,  # Lock-in buffer after TP1 (0.3-0.8 ATR)
        "v3_runner_retention_enabled": True,  # Enable runner retention after TP2
    },
    "quiet": {
        # Moderate signal threshold (raised from 0.55 to 0.68)
        "signal_threshold": 0.68,
        # Wider decision margin (raised from 0.10 to 0.12)
        "decision_margin": 0.12,
        # Wider ATR stop multiplier (raised from 2.5 to 3.2)
        "atr_multiplier": 3.2,
        # Stop percent floor/cap for ATR clamping
        "stop_pct_floor": 0.015,  # 1.5%
        "stop_pct_cap": 0.055,   # 5.5%
        # Improved TP ladder with larger R multiples
        "rr_ratio": 2.0,
        "tp1_r": 1.3,  # Raised from 1.0
        "tp1_scale": 0.25,  # 20-30% range
        "tp2_r": 2.5,  # Raised from 2.0
        "tp2_scale": 0.45,  # 40-50% range
        "use_trailing_tp": True,
        "trailing_activation_atr": 0.6,
        "trailing_distance_atr": 1.5,
        "profit_retracement_pct": 0.50,
        "position_size_mod": 0.50,
        # Increased cooldown (raised from 6 to 13)
        "cooldown_candles": 13,
        # Reduced daily trades (from 2 to 1)
        "max_trades_per_day": 1,
        "time_limit_hours": 48,
        "volume_multiplier": 1.0,
        "trend_weight": 0.5,
        "mr_weight": 0.5,
        "breakeven_atr": 0.6,
        "suitability_long": 0.6,
        "suitability_short": 0.6,
        "min_progress_r": 0.4,
        # Minimum hold time (new field)
        "min_hold_candles": 8,
        # Cost-aware edge gate (new field)
        "min_reward_cost_ratio": 2.4,
        # Phase 1: Edge floor for activation gate (expectancy-positive redesign)
        "edge_floor": 0.07,  # Minimum edge in R units to enter quiet trades
        # Phase 1: Ambiguity veto threshold
        "ambiguity_veto_threshold": 0.05,  # Minimum long/short score difference
        # ─── V3: Directional Policy (Symmetric in quiet) ─────────────────────────
        "v3_long_bias": False,  # Symmetric in quiet
        "v3_tactical_short_allowed": True,  # Tactical shorts allowed as exceptions
        "v3_tactical_short_edge_mult": 1.8,  # Edge floor multiplier for tactical shorts (1.5-2.0)
        "v3_tactical_short_size_cap": 0.25,  # Size cap for tactical shorts (0.20-0.35)
        "v3_tactical_short_hold_cap": 8,  # Hold cap in candles for tactical shorts (6-16)
        "v3_tactical_short_no_pyramiding": True,  # No pyramiding of tactical shorts
        # ─── V3: Risk-on Bull-Capture Activation (not applicable for quiet) ───────
        "v3_risk_on_bull_capture": False,  # Not applicable for quiet
        "v3_regime_confidence_min": 0.72,  # Minimum regime confidence (0.68-0.80)
        "v3_benchmark_trend_strength_min": 0.68,  # Minimum benchmark trend strength (0.60-0.80)
        "v3_drawdown_max_for_risk_on": 0.08,  # Maximum drawdown for risk-on mode (8%)
        # ─── V3: Payoff Mechanics ───────────────────────────────────────────────────
        "v3_payoff_enabled": True,  # Enable V3 payoff mechanics
        "v3_hybrid_stop_atr_mult": 1.5,  # Hybrid stop ATR multiplier (1.2-2.0)
        "v3_stop_pct_floor": 0.015,  # Stop percent floor (1.5%)
        "v3_stop_pct_cap": 0.055,  # Stop percent cap (5.5%)
        "v3_lockin_buffer_atr": 0.5,  # Lock-in buffer after TP1 (0.3-0.8 ATR)
        "v3_runner_retention_enabled": True,  # Enable runner retention after TP2
    },
    "unknown": {
        # Conservative defaults for unknown regime
        "signal_threshold": 0.68,
        "decision_margin": 0.12,
        "atr_multiplier": 3.2,
        "stop_pct_floor": 0.015,  # 1.5%
        "stop_pct_cap": 0.055,   # 5.5%
        "rr_ratio": 2.0,
        "tp1_r": 1.3,
        "tp1_scale": 0.25,
        "tp2_r": 2.5,
        "tp2_scale": 0.45,
        "use_trailing_tp": True,
        "trailing_activation_atr": 0.8,
        "trailing_distance_atr": 1.8,
        "profit_retracement_pct": 0.50,
        "position_size_mod": 0.40,
        "cooldown_candles": 13,
        "max_trades_per_day": 1,
        "time_limit_hours": 24,
        "volume_multiplier": 1.5,
        "trend_weight": 0.5,
        "mr_weight": 0.5,
        "breakeven_atr": 0.8,
        "suitability_long": 0.4,
        "suitability_short": 0.4,
        "min_progress_r": 0.4,
        # Minimum hold time (new field)
        "min_hold_candles": 8,
        # Cost-aware edge gate (new field)
        "min_reward_cost_ratio": 2.4,
        # Phase 1: Edge floor for activation gate (expectancy-positive redesign)
        "edge_floor": 0.07,  # Minimum edge in R units to enter quiet trades
        # Phase 1: Ambiguity veto threshold
        "ambiguity_veto_threshold": 0.05,  # Minimum long/short score difference
        # ─── V3: Directional Policy (Symmetric in unknown) ───────────────────────
        "v3_long_bias": False,  # Symmetric in unknown
        "v3_tactical_short_allowed": True,  # Tactical shorts allowed as exceptions
        "v3_tactical_short_edge_mult": 1.8,  # Edge floor multiplier for tactical shorts (1.5-2.0)
        "v3_tactical_short_size_cap": 0.25,  # Size cap for tactical shorts (0.20-0.35)
        "v3_tactical_short_hold_cap": 8,  # Hold cap in candles for tactical shorts (6-16)
        "v3_tactical_short_no_pyramiding": True,  # No pyramiding of tactical shorts
        # ─── V3: Risk-on Bull-Capture Activation (not applicable for unknown) ─────
        "v3_risk_on_bull_capture": False,  # Not applicable for unknown
        "v3_regime_confidence_min": 0.72,  # Minimum regime confidence (0.68-0.80)
        "v3_benchmark_trend_strength_min": 0.68,  # Minimum benchmark trend strength (0.60-0.80)
        "v3_drawdown_max_for_risk_on": 0.08,  # Maximum drawdown for risk-on mode (8%)
        # ─── V3: Payoff Mechanics ───────────────────────────────────────────────────
        "v3_payoff_enabled": True,  # Enable V3 payoff mechanics
        "v3_hybrid_stop_atr_mult": 1.5,  # Hybrid stop ATR multiplier (1.2-2.0)
        "v3_stop_pct_floor": 0.015,  # Stop percent floor (1.5%)
        "v3_stop_pct_cap": 0.055,  # Stop percent cap (5.5%)
        "v3_lockin_buffer_atr": 0.5,  # Lock-in buffer after TP1 (0.3-0.8 ATR)
        "v3_runner_retention_enabled": True,  # Enable runner retention after TP2
    },
}


def get_regime_profile(regime_value: str) -> Dict[str, Any]:
    """Get the strategy profile for a given regime value string."""
    return REGIME_PROFILES.get(regime_value, REGIME_PROFILES["unknown"])


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
    Enhanced regime detector with dynamic thresholds, Hurst exponent,
    hysteresis, and confidence scoring.

    Key improvements:
    1. Dynamic thresholds based on rolling volatility percentiles
    2. Multi-factor confidence calculation
    3. Hurst exponent for trending vs mean-reverting distinction
    4. Hysteresis thresholds to prevent regime flickering
    5. Minimum regime duration tracking
    6. Smooth regime transitions
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

        # Hysteresis: different thresholds for entering vs exiting a regime
        # Entry thresholds are stricter, exit thresholds are looser
        self.adx_enter_trend = self.config.get("adx_enter_trend", 25.0)
        self.adx_exit_trend = self.config.get("adx_exit_trend", 18.0)
        self.ema_spread_enter_trend = self.config.get("ema_spread_enter_trend", 0.02)
        self.ema_spread_exit_trend = self.config.get("ema_spread_exit_trend", 0.008)

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

        # Hurst exponent cache (expensive to compute, update periodically)
        self._hurst_cache: Optional[float] = None
        self._hurst_update_counter: int = 0
        self._hurst_update_interval: int = 20  # Recalculate every 20 candles
        self._hurst_min_samples: int = 200  # Minimum candles for R/S analysis

        # Minimum regime duration tracking
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._regime_candle_count: int = 0
        self._min_regime_duration: Dict[str, int] = {
            "trending_up": 4,
            "trending_down": 4,
            "ranging": 6,
            "volatile": 2,
            "quiet": 4,
            "unknown": 0,
        }

        # Regime transition tracking
        self._candles_since_regime_change: int = 999
        self._transition_buffer_candles: int = (
            3  # No entries for 3 candles after change
        )

        logger.info(
            f"AdaptiveMarketRegimeDetector initialized: "
            f"volatility_lookback={self.volatility_lookback}, "
            f"adx_lookback={self.adx_lookback}, "
            f"hurst_interval={self._hurst_update_interval}"
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
            vol_25th = np.percentile(vol_array, 25)
            vol_75th = np.percentile(vol_array, 75)

            # Dynamic volatility thresholds using 25th/75th percentiles
            thresholds["vol_low"] = vol_25th
            thresholds["vol_high"] = vol_75th

            # Adjust BB thresholds based on volatility regime
            if len(self.bb_width_history) >= 30:
                bb_array = np.array(self.bb_width_history)
                bb_25th = np.percentile(bb_array, 25)
                bb_75th = np.percentile(bb_array, 75)

                thresholds["bb_squeeze"] = bb_25th
                thresholds["bb_volatile"] = bb_75th

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

        # Calculate Hurst exponent (cached, only recomputed every N candles)
        hurst = self._calculate_hurst_exponent(prices)

        # Determine regime with confidence scoring
        regime, confidence, indicator_scores = self._classify_regime_with_confidence(
            adx=adx or 0.0,
            volatility=volatility,
            bb_width=bb_width,
            ema_spread=ema_spread,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            thresholds=thresholds,
            hurst=hurst,
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

    def _calculate_hurst_exponent(self, prices: np.ndarray) -> Optional[float]:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) analysis.

        H > 0.5: trending (persistent), H < 0.5: mean-reverting, H ≈ 0.5: random walk.
        Uses vectorized computation over multiple sub-series lengths.

        Args:
            prices: Array of closing prices (needs >= 200 samples)

        Returns:
            Hurst exponent (0.0-1.0) or None if insufficient data
        """
        self._hurst_update_counter += 1
        if (
            self._hurst_cache is not None
            and self._hurst_update_counter % self._hurst_update_interval != 0
        ):
            return self._hurst_cache

        if len(prices) < self._hurst_min_samples:
            return None

        series = prices[-self._hurst_min_samples :]
        log_returns = np.diff(np.log(series))
        n = len(log_returns)

        # Sub-series sizes: powers of 2 from 8 to n//2
        sizes = []
        s = 8
        while s <= n // 2:
            sizes.append(s)
            s *= 2
        if len(sizes) < 3:
            return None

        rs_values = np.empty(len(sizes))
        for idx, size in enumerate(sizes):
            num_subseries = n // size
            subseries = log_returns[: num_subseries * size].reshape(num_subseries, size)
            means = subseries.mean(axis=1, keepdims=True)
            deviations = subseries - means
            cumulative = np.cumsum(deviations, axis=1)
            ranges = cumulative.max(axis=1) - cumulative.min(axis=1)
            stds = subseries.std(axis=1, ddof=1)
            valid = stds > 0
            if not valid.any():
                rs_values[idx] = np.nan
                continue
            rs_values[idx] = np.mean(ranges[valid] / stds[valid])

        # Linear regression of log(R/S) vs log(size) for Hurst exponent
        valid_mask = ~np.isnan(rs_values)
        if valid_mask.sum() < 3:
            return None

        log_sizes = np.log(np.array(sizes)[valid_mask])
        log_rs = np.log(rs_values[valid_mask])

        # Least squares: H = slope of log(R/S) vs log(n)
        n_valid = len(log_sizes)
        sum_x = log_sizes.sum()
        sum_y = log_rs.sum()
        sum_xy = (log_sizes * log_rs).sum()
        sum_xx = (log_sizes * log_sizes).sum()
        hurst = (n_valid * sum_xy - sum_x * sum_y) / (n_valid * sum_xx - sum_x * sum_x)
        hurst = float(np.clip(hurst, 0.0, 1.0))

        self._hurst_cache = hurst
        return hurst

    def is_in_transition(self) -> bool:
        """Check if the regime recently changed (within transition buffer).

        The strategy engine should skip new entries during transitions
        to avoid whipsaws.

        Returns:
            True if within the transition buffer period
        """
        return self._candles_since_regime_change < self._transition_buffer_candles

    def _classify_regime_with_confidence(
        self,
        adx: float,
        volatility: float,
        bb_width: float,
        ema_spread: float,
        ema_fast: Optional[float],
        ema_slow: Optional[float],
        thresholds: Dict[str, float],
        hurst: Optional[float] = None,
    ) -> Tuple[MarketRegime, float, Dict[str, float]]:
        """
        Classify market regime with confidence scoring.

        Uses Hurst exponent as a tiebreaker: H > 0.55 favours trending,
        H < 0.45 favours ranging/mean-reverting.

        Returns:
            Tuple of (regime, confidence, indicator_scores)
        """
        scores: Dict[MarketRegime, float] = {}
        indicator_scores: Dict[str, float] = {}

        if hurst is not None:
            indicator_scores["hurst"] = hurst

        # Hysteresis-aware ADX thresholds
        is_currently_trending = self._current_regime in (
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
        )
        adx_threshold = (
            self.adx_exit_trend if is_currently_trending else self.adx_enter_trend
        )
        ema_threshold = (
            self.ema_spread_exit_trend
            if is_currently_trending
            else self.ema_spread_enter_trend
        )

        # ADX scoring
        adx_strong = thresholds["adx_strong_trend"]
        adx_weak = thresholds["adx_weak_trend"]

        if adx > adx_threshold and ema_spread > ema_threshold:
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

        # Hurst exponent tiebreaker: nudge scores based on persistence
        if hurst is not None and scores:
            if hurst > 0.55:
                # Persistent / trending — boost trending scores, penalise ranging
                for r in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                    if r in scores:
                        scores[r] = min(1.0, scores[r] + 0.10)
                if MarketRegime.RANGING in scores:
                    scores[MarketRegime.RANGING] = max(
                        0.0, scores[MarketRegime.RANGING] - 0.10
                    )
            elif hurst < 0.45:
                # Mean-reverting — boost ranging, penalise trending
                if MarketRegime.RANGING in scores:
                    scores[MarketRegime.RANGING] = min(
                        1.0, scores[MarketRegime.RANGING] + 0.10
                    )
                for r in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                    if r in scores:
                        scores[r] = max(0.0, scores[r] - 0.10)

        # Select regime with highest score
        if scores:
            best_regime = max(scores.items(), key=lambda x: x[1])[0]
            base_confidence = scores[best_regime]

            # Calculate confidence based on indicator agreement
            agreeing_indicators = sum(1 for s in scores.values() if s > 0.5)

            if agreeing_indicators == 1:
                confidence = base_confidence
            elif agreeing_indicators == 2:
                confidence = min(1.0, base_confidence * 1.1)
            else:
                confidence = min(1.0, base_confidence * 1.15)

            # Adjust confidence based on how clear the signal is
            score_values = list(scores.values())
            if len(score_values) > 1:
                score_values.sort(reverse=True)
                if score_values[0] > 0:
                    clarity = (score_values[0] - score_values[1]) / score_values[0]
                    confidence = min(1.0, confidence * (0.9 + clarity * 0.2))

            return best_regime, confidence, indicator_scores

        return MarketRegime.UNKNOWN, 0.0, indicator_scores

    def _smooth_regime(self, new_regime: MarketRegime) -> MarketRegime:
        """Apply smoothing to avoid rapid regime switches.

        Smoothing only applies when the regime is CHANGING. If the new regime
        matches the current regime, it passes through immediately (no lag).
        For regime changes, require 2 out of 3 recent detections to confirm,
        AND the current regime must have lasted its minimum duration.
        Also tracks candles since last regime change for transition buffer.
        """
        self.regime_history.append(new_regime)

        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)

        self._regime_candle_count += 1
        self._candles_since_regime_change += 1

        if len(self.regime_history) < 2:
            self._current_regime = new_regime
            return new_regime

        # If new regime matches the current regime, no smoothing needed
        if new_regime == self._current_regime:
            return new_regime

        # Enforce minimum regime duration before allowing a change
        min_duration = self._min_regime_duration.get(self._current_regime.value, 0)
        if self._regime_candle_count < min_duration:
            return self._current_regime

        # Regime is changing - require confirmation (2 out of 3)
        if len(self.regime_history) < 3:
            return self._current_regime

        from collections import Counter

        regime_counts = Counter(self.regime_history[-3:])
        most_common = regime_counts.most_common(1)[0]

        if most_common[1] >= 2 and most_common[0] != self._current_regime:
            # Confirmed regime change
            self._current_regime = most_common[0]
            self._regime_candle_count = 0
            self._candles_since_regime_change = 0
            return most_common[0]

        # No clear consensus for change - stay with current regime
        return self._current_regime

    def should_trade_in_regime(self, regime: MarketRegime) -> Tuple[bool, float]:
        """
        Determine if trading should occur in the detected regime.

        Consults REGIME_PROFILES for position_size_mod and max_trades_per_day.
        Returns (False, 0.0) if position_size_mod is 0 or max_trades_per_day is 0.

        Returns:
            Tuple of (should_trade, position_size_modifier)
        """
        profile = get_regime_profile(regime.value)
        pos_mod = profile["position_size_mod"]
        max_trades = profile["max_trades_per_day"]

        if pos_mod == 0.0 or max_trades == 0:
            return (False, 0.0)

        return (True, pos_mod)

    def get_regime_suitability(self, regime: MarketRegime, direction: str) -> float:
        """
        Get how suitable a regime is for a specific trade direction.

        Uses the suitability_long and suitability_short fields from REGIME_PROFILES
        for regime-specific direction preferences.

        Args:
            regime: Current market regime
            direction: 'long' or 'short'

        Returns:
            Suitability score (0.0 to 1.0)
        """
        profile = get_regime_profile(regime.value)
        if direction == "long":
            return profile.get("suitability_long", 0.5)
        elif direction == "short":
            return profile.get("suitability_short", 0.5)
        return 0.5

    def get_adaptive_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get adaptive strategy parameters for the current regime.

        Delegates to the centralized REGIME_PROFILES dictionary so there is
        a single source of truth for all regime-specific parameters.

        Returns:
            Dictionary with parameter adjustments
        """
        profile = get_regime_profile(regime.value)
        return {
            "atr_multiplier": profile["atr_multiplier"],
            "risk_reward_ratio": profile["rr_ratio"],
            "signal_threshold": profile["signal_threshold"],
            "position_size_modifier": profile["position_size_mod"],
            "trend_weight": profile["trend_weight"],
            "mean_reversion_weight": profile["mr_weight"],
            "regime": regime.value,
        }

    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current dynamic threshold values."""
        return self.current_thresholds.copy()


# Backwards compatibility - alias old name to new class
MarketRegimeDetector = AdaptiveMarketRegimeDetector
