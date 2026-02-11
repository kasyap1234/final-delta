"""
Regime Risk Router Module

Provides pure functions for mapping regime profiles to risk management parameters
including stop loss, risk-reward ratios, take profit ladders, and drawdown clamps.
"""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import logging

from src.indicators.market_regime import MarketRegime, get_regime_profile

logger = logging.getLogger(__name__)


@dataclass
class TakeProfitLevel:
    """Single take profit level configuration."""
    
    level: int  # 1, 2, 3
    r_multiple: float  # Risk multiple (e.g., 1.2 for 1.2R)
    scale_percent: float  # Percentage of position to close (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "r_multiple": self.r_multiple,
            "scale_percent": self.scale_percent,
        }


@dataclass
class RiskParameters:
    """Complete risk parameters for a regime."""

    atr_multiplier: float
    rr_ratio: float
    tp_levels: List[TakeProfitLevel]
    use_trailing: bool
    trailing_activation_atr: float
    trailing_distance_atr: float
    profit_retracement_pct: float
    position_size_mod: float
    min_progress_r: float  # Minimum R progress before time exit
    stop_pct_floor: Optional[float] = None  # Minimum stop loss as percentage of entry
    stop_pct_cap: Optional[float] = None  # Maximum stop loss as percentage of entry
    min_hold_candles: int = 0  # Minimum hold time in candles
    min_reward_cost_ratio: float = 2.0  # Minimum reward/cost ratio for entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atr_multiplier": self.atr_multiplier,
            "rr_ratio": self.rr_ratio,
            "tp_levels": [tp.to_dict() for tp in self.tp_levels],
            "use_trailing": self.use_trailing,
            "trailing_activation_atr": self.trailing_activation_atr,
            "trailing_distance_atr": self.trailing_distance_atr,
            "profit_retracement_pct": self.profit_retracement_pct,
            "position_size_mod": self.position_size_mod,
            "min_progress_r": self.min_progress_r,
            "stop_pct_floor": self.stop_pct_floor,
            "stop_pct_cap": self.stop_pct_cap,
            "min_hold_candles": self.min_hold_candles,
            "min_reward_cost_ratio": self.min_reward_cost_ratio,
        }


@dataclass
class DrawdownLadder:
    """Drawdown-based position size reduction ladder."""
    
    drawdown_thresholds: List[float]  # e.g., [0.05, 0.08, 0.10, 0.12]
    size_multipliers: List[float]  # e.g., [0.75, 0.50, 0.25, 0.0]
    
    def get_size_multiplier(self, current_drawdown: float) -> float:
        """Get position size multiplier based on current drawdown."""
        # Return the highest threshold that is met
        for threshold, multiplier in zip(reversed(self.drawdown_thresholds), reversed(self.size_multipliers)):
            if current_drawdown >= threshold:
                return multiplier
        return 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drawdown_thresholds": self.drawdown_thresholds,
            "size_multipliers": self.size_multipliers,
        }


# Default drawdown ladder (stricter than current implementation)
DEFAULT_DRAWDOWN_LADDER = DrawdownLadder(
    drawdown_thresholds=[0.05, 0.08, 0.10, 0.12],
    size_multipliers=[0.75, 0.50, 0.25, 0.0],
)


def get_risk_parameters(regime: MarketRegime) -> RiskParameters:
    """
    Get risk parameters for a regime from REGIME_PROFILES.

    Args:
        regime: Current market regime

    Returns:
        RiskParameters with all risk management settings
    """
    profile = get_regime_profile(regime.value)

    # Build TP ladder from profile
    tp_levels = []

    # TP1
    tp1_r = profile.get("tp1_r", 1.0)
    tp1_scale = profile.get("tp1_scale", 0.25)
    if tp1_r > 0 and tp1_scale > 0:
        tp_levels.append(TakeProfitLevel(level=1, r_multiple=tp1_r, scale_percent=tp1_scale))

    # TP2
    tp2_r = profile.get("tp2_r", 2.0)
    tp2_scale = profile.get("tp2_scale", 0.50)
    if tp2_r > 0 and tp2_scale > 0:
        tp_levels.append(TakeProfitLevel(level=2, r_multiple=tp2_r, scale_percent=tp2_scale))

    # TP3 (optional - for trailing continuation)
    # If rr_ratio is 0, use trailing-only mode
    rr_ratio = profile.get("rr_ratio", 2.0)

    return RiskParameters(
        atr_multiplier=profile.get("atr_multiplier", 2.0),
        rr_ratio=rr_ratio,
        tp_levels=tp_levels,
        use_trailing=profile.get("use_trailing_tp", True),
        trailing_activation_atr=profile.get("trailing_activation_atr", 1.0),
        trailing_distance_atr=profile.get("trailing_distance_atr", 2.0),
        profit_retracement_pct=profile.get("profit_retracement_pct", 0.50),
        position_size_mod=profile.get("position_size_mod", 1.0),
        min_progress_r=profile.get("min_progress_r", 0.5),
        stop_pct_floor=profile.get("stop_pct_floor"),
        stop_pct_cap=profile.get("stop_pct_cap"),
        min_hold_candles=profile.get("min_hold_candles", 0),
        min_reward_cost_ratio=profile.get("min_reward_cost_ratio", 2.0),
    )


def calculate_stop_loss_price(
    entry_price: float,
    atr: float,
    atr_multiplier: float,
    position_type: str,
    stop_pct_floor: Optional[float] = None,
    stop_pct_cap: Optional[float] = None,
) -> float:
    """
    Calculate stop loss price using ATR-based method with floor/cap enforcement.

    Implements the stop loss framework redesign from profitability improvement plan:
    - raw_stop_distance = ATR * atr_multiplier
    - raw_stop_pct = raw_stop_distance / entry_price
    - clamped_stop_pct = clamp raw_stop_pct by regime floor cap
    - final_stop_distance = clamped_stop_pct * entry_price

    Args:
        entry_price: Entry price
        atr: Average True Range
        atr_multiplier: ATR multiplier from regime profile
        position_type: 'long' or 'short'
        stop_pct_floor: Minimum stop loss as percentage of entry price (e.g., 0.018 for 1.8%)
        stop_pct_cap: Maximum stop loss as percentage of entry price (e.g., 0.060 for 6.0%)

    Returns:
        Stop loss price
    """
    # Calculate raw stop distance
    raw_stop_distance = atr * atr_multiplier
    raw_stop_pct = raw_stop_distance / entry_price

    # Apply floor/cap clamping if provided
    clamped_stop_pct = raw_stop_pct
    if stop_pct_floor is not None:
        clamped_stop_pct = max(clamped_stop_pct, stop_pct_floor)
    if stop_pct_cap is not None:
        clamped_stop_pct = min(clamped_stop_pct, stop_pct_cap)

    # Calculate final stop distance from clamped percentage
    final_stop_distance = clamped_stop_pct * entry_price

    if position_type == "long":
        return entry_price - final_stop_distance
    else:  # short
        return entry_price + final_stop_distance


def calculate_take_profit_prices(
    entry_price: float,
    stop_loss_price: float,
    position_type: str,
    tp_levels: List[TakeProfitLevel],
) -> List[Tuple[float, float]]:
    """
    Calculate take profit prices for each TP level.
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        position_type: 'long' or 'short'
        tp_levels: List of TakeProfitLevel configurations
    
    Returns:
        List of (price, scale_percent) tuples
    """
    stop_distance = abs(entry_price - stop_loss_price)
    
    tp_prices = []
    for tp in tp_levels:
        profit_distance = stop_distance * tp.r_multiple
        
        if position_type == "long":
            tp_price = entry_price + profit_distance
        else:  # short
            tp_price = entry_price - profit_distance
        
        tp_prices.append((tp_price, tp.scale_percent))
    
    return tp_prices


def calculate_trailing_stop_price(
    current_price: float,
    entry_price: float,
    highest_price: float,
    lowest_price: float,
    atr: float,
    trailing_distance_atr: float,
    position_type: str,
) -> Optional[float]:
    """
    Calculate trailing stop price.
    
    Args:
        current_price: Current market price
        entry_price: Original entry price
        highest_price: Highest price since entry (for longs)
        lowest_price: Lowest price since entry (for shorts)
        atr: Current ATR value
        trailing_distance_atr: Trailing distance ATR multiplier
        position_type: 'long' or 'short'
    
    Returns:
        New trailing stop price or None if no update needed
    """
    trailing_distance = atr * trailing_distance_atr
    
    if position_type == "long":
        # For longs, trail below the highest price
        new_stop = highest_price - trailing_distance
        # Only move stop up, never down
        current_stop = entry_price - (atr * trailing_distance_atr)
        if new_stop > current_stop:
            return new_stop
    else:  # short
        # For shorts, trail above the lowest price
        new_stop = lowest_price + trailing_distance
        # Only move stop down, never up
        current_stop = entry_price + (atr * trailing_distance_atr)
        if new_stop < current_stop:
            return new_stop
    
    return None


def check_profit_retracement_exit(
    current_price: float,
    entry_price: float,
    highest_price: float,
    lowest_price: float,
    profit_retracement_pct: float,
    position_type: str,
) -> bool:
    """
    Check if profit retracement threshold has been exceeded.
    
    Args:
        current_price: Current market price
        entry_price: Entry price
        highest_price: Highest price since entry (for longs)
        lowest_price: Lowest price since entry (for shorts)
        profit_retracement_pct: Maximum allowed retracement as percentage of peak profit
        position_type: 'long' or 'short'
    
    Returns:
        True if retracement threshold exceeded
    """
    if position_type == "long":
        peak_profit = highest_price - entry_price
        current_profit = current_price - entry_price
        
        if peak_profit > 0:
            retracement = peak_profit - current_profit
            retracement_pct = retracement / peak_profit
            return retracement_pct > profit_retracement_pct
    else:  # short
        peak_profit = entry_price - lowest_price
        current_profit = entry_price - current_price
        
        if peak_profit > 0:
            retracement = peak_profit - current_profit
            retracement_pct = retracement / peak_profit
            return retracement_pct > profit_retracement_pct
    
    return False


def get_drawdown_size_multiplier(
    current_drawdown: float,
    ladder: Optional[DrawdownLadder] = None,
) -> float:
    """
    Get position size multiplier based on current drawdown.
    
    Args:
        current_drawdown: Current drawdown as decimal (e.g., 0.05 for 5%)
        ladder: Optional custom drawdown ladder
    
    Returns:
        Position size multiplier (0.0 to 1.0)
    """
    if ladder is None:
        ladder = DEFAULT_DRAWDOWN_LADDER
    
    return ladder.get_size_multiplier(current_drawdown)


def calculate_r_per_r(
    current_price: float,
    entry_price: float,
    stop_loss_price: float,
    position_type: str,
) -> float:
    """
    Calculate current unrealized profit in R multiples.
    
    Args:
        current_price: Current market price
        entry_price: Entry price
        stop_loss_price: Stop loss price
        position_type: 'long' or 'short'
    
    Returns:
        Current profit in R multiples
    """
    stop_distance = abs(entry_price - stop_loss_price)
    
    if position_type == "long":
        current_profit = current_price - entry_price
    else:  # short
        current_profit = entry_price - current_price
    
    return current_profit / stop_distance if stop_distance > 0 else 0.0
