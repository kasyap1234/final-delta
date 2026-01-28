"""
Price impact model for realistic market impact simulation.

This module provides market impact modeling based on the square-root law,
which estimates how large orders affect market prices.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImpactConfig:
    """Configuration for price impact model."""
    # Square-root law coefficient (typical range: 0.1 to 1.0)
    impact_coefficient: float = 0.5
    # Temporary impact decay factor (0-1, higher = faster decay)
    temporary_decay: float = 0.7
    # Permanent impact ratio (0-1, portion of impact that is permanent)
    permanent_ratio: float = 0.3
    # Minimum impact threshold (bps)
    min_impact_bps: float = 1.0
    # Maximum impact cap (bps)
    max_impact_bps: float = 500.0
    # Volatility scaling factor
    volatility_factor: float = 1.0
    # Market depth factor (higher = deeper market, less impact)
    market_depth_factor: float = 1.0


@dataclass
class PriceImpact:
    """Price impact calculation result."""
    temporary_impact: float  # Short-term price movement
    permanent_impact: float  # Long-term price movement
    total_impact: float  # Combined impact
    expected_slippage: float  # Expected execution slippage
    fill_probability: float  # Probability of complete fill


class PriceImpactModel:
    """
    Market impact model using square-root law.
    
    The square-root law states that price impact is proportional to
    the square root of order size relative to average volume:
    
        Impact = coefficient * sqrt(order_size / avg_volume)
    
    This model separates impact into:
    - Temporary impact: Short-lived price movement from immediate execution
    - Permanent impact: Lasting price change from information leakage
    
    Example:
        ```python
        model = PriceImpactModel()
        
        impact = model.calculate_impact(
            order_size=10.0,
            avg_volume=100.0,
            current_price=50000.0,
            volatility=0.02
        )
        
        print(f"Expected slippage: {impact.expected_slippage:.2f}")
        print(f"Fill probability: {impact.fill_probability:.2%}")
        ```
    """
    
    def __init__(self, config: Optional[ImpactConfig] = None):
        """
        Initialize the price impact model.
        
        Args:
            config: Impact configuration. Uses defaults if not provided.
        """
        self.config = config or ImpactConfig()
    
    def calculate_impact(
        self,
        order_size: float,
        avg_volume: float,
        current_price: float,
        volatility: float = 0.02,
        side: str = 'buy'
    ) -> PriceImpact:
        """
        Calculate price impact for an order.
        
        Args:
            order_size: Order size in base currency units
            avg_volume: Average trading volume (same time period)
            current_price: Current market price
            volatility: Price volatility (standard deviation)
            side: Order side ('buy' or 'sell')
            
        Returns:
            PriceImpact with impact calculations
        """
        if avg_volume <= 0 or order_size <= 0:
            return PriceImpact(0.0, 0.0, 0.0, 0.0, 1.0)
        
        # Calculate participation rate
        participation_rate = order_size / avg_volume
        
        # Base impact using square-root law
        base_impact = self.config.impact_coefficient * math.sqrt(participation_rate)
        
        # Scale by volatility (higher vol = higher impact)
        vol_adjustment = 1.0 + (volatility * self.config.volatility_factor * 10)
        
        # Scale by market depth
        depth_adjustment = 1.0 / max(0.1, self.config.market_depth_factor)
        
        # Calculate total impact as percentage
        total_impact_pct = base_impact * vol_adjustment * depth_adjustment
        
        # Convert to basis points and clamp
        total_impact_bps = total_impact_pct * 10000
        total_impact_bps = max(
            self.config.min_impact_bps,
            min(self.config.max_impact_bps, total_impact_bps)
        )
        
        # Split into temporary and permanent
        permanent_bps = total_impact_bps * self.config.permanent_ratio
        temporary_bps = total_impact_bps * (1 - self.config.permanent_ratio)
        
        # Calculate expected slippage in price terms
        expected_slippage = current_price * (total_impact_bps / 10000)
        
        # Calculate fill probability based on order size vs volume
        fill_probability = self._calculate_fill_probability(
            order_size, avg_volume, total_impact_bps
        )
        
        return PriceImpact(
            temporary_impact=temporary_bps,
            permanent_impact=permanent_bps,
            total_impact=total_impact_bps,
            expected_slippage=expected_slippage,
            fill_probability=fill_probability
        )
    
    def _calculate_fill_probability(
        self,
        order_size: float,
        avg_volume: float,
        impact_bps: float
    ) -> float:
        """
        Calculate probability of complete fill.
        
        Args:
            order_size: Order size
            avg_volume: Average volume
            impact_bps: Price impact in basis points
            
        Returns:
            Fill probability (0-1)
        """
        if avg_volume <= 0:
            return 0.0
        
        participation_rate = order_size / avg_volume
        
        # Base probability decreases with participation rate
        if participation_rate <= 0.01:
            base_prob = 0.95
        elif participation_rate <= 0.05:
            base_prob = 0.85
        elif participation_rate <= 0.10:
            base_prob = 0.70
        elif participation_rate <= 0.20:
            base_prob = 0.50
        elif participation_rate <= 0.50:
            base_prob = 0.30
        else:
            base_prob = 0.15
        
        # Higher impact reduces fill probability
        impact_penalty = min(0.3, impact_bps / 1000)
        
        probability = base_prob - impact_penalty
        return max(0.05, min(0.99, probability))
    
    def calculate_temporary_impact_decay(
        self,
        initial_impact: float,
        time_elapsed: float,
        half_life: float = 60.0
    ) -> float:
        """
        Calculate decayed temporary impact over time.
        
        Args:
            initial_impact: Initial temporary impact
            time_elapsed: Time elapsed since order (seconds)
            half_life: Impact half-life in seconds
            
        Returns:
            Remaining temporary impact
        """
        if time_elapsed <= 0:
            return initial_impact
        
        # Exponential decay
        decay_factor = math.exp(-time_elapsed / half_life)
        return initial_impact * decay_factor * self.config.temporary_decay
    
    def estimate_market_depth(
        self,
        avg_volume: float,
        price: float,
        spread_pct: float = 0.0002
    ) -> Dict[str, float]:
        """
        Estimate order book depth at various price levels.
        
        Args:
            avg_volume: Average trading volume
            price: Current price
            spread_pct: Bid-ask spread percentage
            
        Returns:
            Dictionary with depth at different price levels
        """
        # Estimate depth based on volume
        base_depth = avg_volume * 0.1  # 10% of avg volume at best price
        
        depth = {
            'best': base_depth,
            '1bp': base_depth * 1.5,   # 1 basis point away
            '5bp': base_depth * 3.0,   # 5 basis points away
            '10bp': base_depth * 5.0,  # 10 basis points away
            '50bp': base_depth * 10.0, # 50 basis points away
        }
        
        return depth
    
    def simulate_execution_price(
        self,
        order_size: float,
        avg_volume: float,
        current_price: float,
        side: str,
        volatility: float = 0.02,
        add_noise: bool = True
    ) -> float:
        """
        Simulate realistic execution price with impact.
        
        Args:
            order_size: Order size
            avg_volume: Average volume
            current_price: Current market price
            side: Order side
            volatility: Price volatility
            add_noise: Whether to add random noise
            
        Returns:
            Simulated execution price
        """
        impact = self.calculate_impact(
            order_size, avg_volume, current_price, volatility, side
        )
        
        # Base price movement from impact
        impact_pct = impact.total_impact / 10000
        
        if side == 'buy':
            # Buy orders execute at higher prices
            execution_price = current_price * (1 + impact_pct)
        else:
            # Sell orders execute at lower prices
            execution_price = current_price * (1 - impact_pct)
        
        # Add random noise for realism
        if add_noise:
            noise = random.gauss(0, volatility * 0.1)
            execution_price *= (1 + noise)
        
        return execution_price
    
    def calculate_optimal_order_size(
        self,
        target_size: float,
        avg_volume: float,
        max_impact_bps: float = 50.0
    ) -> float:
        """
        Calculate optimal order size to limit market impact.
        
        Args:
            target_size: Desired total position size
            avg_volume: Average trading volume
            max_impact_bps: Maximum acceptable impact in basis points
            
        Returns:
            Optimal single order size
        """
        if avg_volume <= 0 or max_impact_bps <= 0:
            return target_size
        
        # Rearrange square-root law to solve for order size
        # impact = coefficient * sqrt(order_size / avg_volume)
        # order_size = avg_volume * (impact / coefficient)^2
        
        max_impact_pct = max_impact_bps / 10000
        optimal_size = avg_volume * (max_impact_pct / self.config.impact_coefficient) ** 2
        
        # Don't exceed target
        return min(optimal_size, target_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get model statistics and configuration.
        
        Returns:
            Dictionary with model info
        """
        return {
            'impact_coefficient': self.config.impact_coefficient,
            'temporary_decay': self.config.temporary_decay,
            'permanent_ratio': self.config.permanent_ratio,
            'min_impact_bps': self.config.min_impact_bps,
            'max_impact_bps': self.config.max_impact_bps,
        }
