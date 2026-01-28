"""
Order book simulation module for realistic market depth modeling.

This module provides order book depth simulation for backtesting, including:
- Bid/ask spread modeling
- Order book depth at multiple price levels
- Order book imbalance tracking
- Liquidity simulation
- Realistic fill price calculation based on order size
"""

import random
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """Represents a single price level in the order book."""
    price: float
    volume: float
    order_count: int = 1
    
    @property
    def value(self) -> float:
        """Calculate total value at this level."""
        return self.price * self.volume


@dataclass
class OrderBookSide:
    """Represents one side of the order book (bids or asks)."""
    levels: List[PriceLevel] = field(default_factory=list)
    
    @property
    def best_price(self) -> Optional[float]:
        """Get the best price on this side."""
        if not self.levels:
            return None
        return self.levels[0].price
    
    @property
    def total_volume(self) -> float:
        """Get total volume on this side."""
        return sum(level.volume for level in self.levels)
    
    @property
    def total_value(self) -> float:
        """Get total value on this side."""
        return sum(level.value for level in self.levels)
    
    def get_volume_up_to_price(self, price: float) -> float:
        """Get total volume up to a specific price level."""
        volume = 0.0
        for level in self.levels:
            if level.price <= price:
                volume += level.volume
            else:
                break
        return volume
    
    def get_volume_at_price(self, price: float) -> float:
        """Get volume at a specific price."""
        for level in self.levels:
            if abs(level.price - price) < 0.0001:
                return level.volume
        return 0.0


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time."""
    symbol: str
    timestamp: datetime
    bids: OrderBookSide
    asks: OrderBookSide
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids.best_price
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks.best_price
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate absolute spread."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        """Calculate spread percentage."""
        mid = self.mid_price
        spread = self.spread
        if mid is not None and spread is not None and mid > 0:
            return spread / mid
        return None
    
    @property
    def bid_ask_imbalance(self) -> Optional[float]:
        """
        Calculate bid-ask imbalance (-1 to 1).
        Positive = more bids (buy pressure)
        Negative = more asks (sell pressure)
        """
        bid_volume = self.bids.total_volume
        ask_volume = self.asks.total_volume
        total = bid_volume + ask_volume
        if total > 0:
            return (bid_volume - ask_volume) / total
        return None


@dataclass
class OrderBookConfig:
    """Configuration for order book simulation."""
    # Depth levels to simulate
    depth_levels: int = 20
    
    # Spread configuration
    min_spread_bps: float = 2.0  # Minimum spread in basis points
    max_spread_bps: float = 50.0  # Maximum spread in basis points
    volatility_spread_factor: float = 10.0  # Spread increases with volatility
    
    # Volume configuration
    base_volume_at_best: float = 1.0  # Base volume at best price
    volume_decay_factor: float = 0.8  # Volume decay per level
    volume_concentration: float = 0.6  # % of volume at best 3 levels
    
    # Imbalance configuration
    enable_imbalance: bool = True
    imbalance_update_prob: float = 0.3  # Probability of imbalance change per update
    max_imbalance: float = 0.7  # Maximum imbalance (-0.7 to 0.7)
    
    # Price level configuration
    price_level_spacing_bps: float = 5.0  # Spacing between levels
    
    # Fill simulation
    fill_price_impact_bps: float = 1.0  # Impact per 1% of level volume consumed


class SimulatedOrderBook:
    """
    Simulates a realistic order book with depth and liquidity.
    
    This class creates synthetic order book data based on OHLCV candles,
    simulating realistic bid/ask spreads, depth at multiple price levels,
    and order book imbalance that affects fill prices.
    
    Features:
    - Dynamic spread based on volatility
    - Volume distribution across price levels
    - Bid-ask imbalance tracking
    - Realistic fill price calculation based on order size
    - Order book depth visualization
    
    Example:
        ```python
        config = OrderBookConfig(depth_levels=20)
        order_book = SimulatedOrderBook('BTC/USD', config)
        
        # Update from candle data
        order_book.update_from_candle(candle, volatility=0.02)
        
        # Get current snapshot
        snapshot = order_book.get_snapshot()
        print(f"Spread: {snapshot.spread_pct:.4%}")
        print(f"Imbalance: {snapshot.bid_ask_imbalance:.2f}")
        
        # Calculate fill price for an order
        fill_price = order_book.calculate_fill_price('buy', 0.5)
        ```
    """
    
    def __init__(self, symbol: str, config: Optional[OrderBookConfig] = None):
        """
        Initialize the simulated order book.
        
        Args:
            symbol: Trading pair symbol
            config: Order book configuration
        """
        self.symbol = symbol
        self.config = config or OrderBookConfig()
        
        self._current_snapshot: Optional[OrderBookSnapshot] = None
        self._last_update: Optional[datetime] = None
        self._current_volatility: float = 0.02
        self._current_imbalance: float = 0.0
        
        # Historical snapshots for analysis
        self._history: List[OrderBookSnapshot] = []
        self._max_history_size = 1000
        
        logger.info(f"SimulatedOrderBook initialized for {symbol}")
    
    def update_from_candle(
        self,
        candle: Any,
        volatility: float = 0.02,
        timestamp: Optional[datetime] = None
    ) -> OrderBookSnapshot:
        """
        Update order book from OHLCV candle data.
        
        Args:
            candle: OHLCV candle with open, high, low, close, volume
            volatility: Current market volatility
            timestamp: Optional timestamp for the update
            
        Returns:
            Updated order book snapshot
        """
        self._current_volatility = volatility
        
        # Calculate mid price from candle
        mid_price = (candle.high + candle.low) / 2
        
        # Calculate spread based on volatility
        spread = self._calculate_spread(mid_price, volatility)
        
        # Update imbalance occasionally
        if self.config.enable_imbalance and random.random() < self.config.imbalance_update_prob:
            self._current_imbalance = self._generate_imbalance()
        
        # Generate bid and ask sides
        bids = self._generate_side(
            mid_price - spread / 2,
            'bid',
            candle.volume,
            self._current_imbalance
        )
        asks = self._generate_side(
            mid_price + spread / 2,
            'ask',
            candle.volume,
            -self._current_imbalance
        )
        
        # Create snapshot
        snapshot = OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=timestamp or datetime.now(),
            bids=OrderBookSide(levels=bids),
            asks=OrderBookSide(levels=asks)
        )
        
        self._current_snapshot = snapshot
        self._last_update = snapshot.timestamp
        
        # Store in history
        self._history.append(snapshot)
        if len(self._history) > self._max_history_size:
            self._history.pop(0)
        
        return snapshot
    
    def _calculate_spread(self, mid_price: float, volatility: float) -> float:
        """Calculate bid-ask spread based on price and volatility."""
        # Base spread in basis points
        base_spread_bps = self.config.min_spread_bps
        
        # Add volatility component
        vol_spread_bps = volatility * self.config.volatility_spread_factor * 10000
        
        # Total spread in bps
        total_spread_bps = min(
            base_spread_bps + vol_spread_bps,
            self.config.max_spread_bps
        )
        
        # Convert to price
        return mid_price * (total_spread_bps / 10000)
    
    def _generate_imbalance(self) -> float:
        """Generate a random imbalance value."""
        # Use normal distribution centered at 0
        imbalance = random.gauss(0, 0.3)
        return max(-self.config.max_imbalance, min(self.config.max_imbalance, imbalance))
    
    def _generate_side(
        self,
        best_price: float,
        side: str,
        candle_volume: float,
        imbalance: float
    ) -> List[PriceLevel]:
        """Generate price levels for one side of the book."""
        levels = []
        
        # Adjust base volume by imbalance
        base_volume = self.config.base_volume_at_best * (1 + imbalance)
        
        # Generate levels
        for i in range(self.config.depth_levels):
            # Price for this level
            if side == 'bid':
                price = best_price * (1 - i * self.config.price_level_spacing_bps / 10000)
            else:
                price = best_price * (1 + i * self.config.price_level_spacing_bps / 10000)
            
            # Volume at this level (decays with depth)
            volume = base_volume * (self.config.volume_decay_factor ** i)
            
            # Add some randomness
            volume *= random.uniform(0.8, 1.2)
            
            # Scale by candle volume
            volume *= candle_volume * 0.01  # 1% of candle volume per level
            
            levels.append(PriceLevel(
                price=price,
                volume=volume,
                order_count=random.randint(1, 10)
            ))
        
        return levels
    
    def calculate_fill_price(
        self,
        side: str,
        amount: float,
        allow_partial: bool = True
    ) -> Tuple[float, float, bool]:
        """
        Calculate realistic fill price for an order.
        
        Args:
            side: 'buy' or 'sell'
            amount: Order amount to fill
            allow_partial: Whether to allow partial fills
            
        Returns:
            Tuple of (average_fill_price, filled_amount, is_complete)
        """
        if self._current_snapshot is None:
            logger.warning("Order book not initialized, cannot calculate fill price")
            return 0.0, 0.0, False
        
        # Determine which side to consume
        if side.lower() == 'buy':
            levels = self._current_snapshot.asks.levels
        else:
            levels = self._current_snapshot.bids.levels
        
        if not levels:
            return 0.0, 0.0, False
        
        remaining = amount
        total_value = 0.0
        total_filled = 0.0
        
        for level in levels:
            if remaining <= 0:
                break
            
            # Calculate how much can be filled at this level
            available = level.volume
            fill_at_level = min(remaining, available)
            
            # Apply price impact based on consumption
            impact_bps = (fill_at_level / level.volume) * self.config.fill_price_impact_bps
            
            if side.lower() == 'buy':
                # Buy orders pay more as they consume liquidity
                effective_price = level.price * (1 + impact_bps / 10000)
            else:
                # Sell orders receive less as they consume liquidity
                effective_price = level.price * (1 - impact_bps / 10000)
            
            total_value += fill_at_level * effective_price
            total_filled += fill_at_level
            remaining -= fill_at_level
        
        if total_filled > 0:
            avg_fill_price = total_value / total_filled
        else:
            avg_fill_price = 0.0
        
        is_complete = remaining <= 0.0001 or allow_partial
        
        return avg_fill_price, total_filled, is_complete
    
    def estimate_market_impact(self, side: str, amount: float) -> float:
        """
        Estimate market impact for an order without executing it.
        
        Args:
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Estimated price impact in basis points
        """
        if self._current_snapshot is None:
            return 0.0
        
        mid_price = self._current_snapshot.mid_price
        if mid_price is None or mid_price == 0:
            return 0.0
        
        fill_price, _, _ = self.calculate_fill_price(side, amount)
        if fill_price == 0:
            return 0.0
        
        if side.lower() == 'buy':
            impact_pct = (fill_price - mid_price) / mid_price
        else:
            impact_pct = (mid_price - fill_price) / mid_price
        
        return impact_pct * 10000  # Convert to bps
    
    def get_liquidity_at_price(self, price: float, side: str) -> float:
        """
        Get available liquidity at a specific price level.
        
        Args:
            price: Price level to check
            side: 'bid' or 'ask'
            
        Returns:
            Available volume at that price
        """
        if self._current_snapshot is None:
            return 0.0
        
        if side.lower() == 'bid':
            return self._current_snapshot.bids.get_volume_at_price(price)
        else:
            return self._current_snapshot.asks.get_volume_at_price(price)
    
    def get_depth_summary(self, levels: int = 5) -> Dict[str, Any]:
        """
        Get a summary of order book depth.
        
        Args:
            levels: Number of levels to include
            
        Returns:
            Dictionary with depth information
        """
        if self._current_snapshot is None:
            return {}
        
        bids = self._current_snapshot.bids.levels[:levels]
        asks = self._current_snapshot.asks.levels[:levels]
        
        return {
            'symbol': self.symbol,
            'timestamp': self._current_snapshot.timestamp,
            'mid_price': self._current_snapshot.mid_price,
            'spread': self._current_snapshot.spread,
            'spread_pct': self._current_snapshot.spread_pct,
            'imbalance': self._current_snapshot.bid_ask_imbalance,
            'bids': [
                {'price': b.price, 'volume': b.volume}
                for b in bids
            ],
            'asks': [
                {'price': a.price, 'volume': a.volume}
                for a in asks
            ],
            'total_bid_volume': self._current_snapshot.bids.total_volume,
            'total_ask_volume': self._current_snapshot.asks.total_volume,
        }
    
    def get_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Get the current order book snapshot."""
        return self._current_snapshot
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices."""
        if self._current_snapshot is None:
            return None, None
        return self._current_snapshot.best_bid, self._current_snapshot.best_ask
    
    def get_imbalance_history(self, lookback: int = 10) -> List[float]:
        """Get historical imbalance values."""
        if not self._history:
            return []
        
        recent = self._history[-lookback:]
        return [
            snap.bid_ask_imbalance for snap in recent
            if snap.bid_ask_imbalance is not None
        ]
    
    def get_average_imbalance(self, lookback: int = 10) -> float:
        """Calculate average imbalance over recent history."""
        history = self.get_imbalance_history(lookback)
        if not history:
            return 0.0
        return sum(history) / len(history)
    
    def is_liquid_enough(self, amount: float, side: str, max_impact_bps: float = 50.0) -> bool:
        """
        Check if there's enough liquidity for an order.
        
        Args:
            amount: Order amount
            side: 'buy' or 'sell'
            max_impact_bps: Maximum acceptable impact in basis points
            
        Returns:
            True if market is liquid enough
        """
        impact = self.estimate_market_impact(side, amount)
        return impact <= max_impact_bps
    
    def reset(self) -> None:
        """Reset the order book state."""
        self._current_snapshot = None
        self._last_update = None
        self._current_volatility = 0.02
        self._current_imbalance = 0.0
        self._history.clear()
        logger.info(f"Order book reset for {self.symbol}")


class MultiSymbolOrderBook:
    """Manages order books for multiple symbols."""
    
    def __init__(self, config: Optional[OrderBookConfig] = None):
        """
        Initialize multi-symbol order book manager.
        
        Args:
            config: Configuration to use for all order books
        """
        self.config = config or OrderBookConfig()
        self._order_books: Dict[str, SimulatedOrderBook] = {}
    
    def get_or_create(self, symbol: str) -> SimulatedOrderBook:
        """Get or create order book for a symbol."""
        if symbol not in self._order_books:
            self._order_books[symbol] = SimulatedOrderBook(symbol, self.config)
        return self._order_books[symbol]
    
    def get(self, symbol: str) -> Optional[SimulatedOrderBook]:
        """Get order book for a symbol if it exists."""
        return self._order_books.get(symbol)
    
    def update_all(
        self,
        candles: Dict[str, Any],
        volatility: float = 0.02,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, OrderBookSnapshot]:
        """Update all order books from candle data."""
        snapshots = {}
        for symbol, candle in candles.items():
            ob = self.get_or_create(symbol)
            snapshots[symbol] = ob.update_from_candle(candle, volatility, timestamp)
        return snapshots
    
    def get_all_snapshots(self) -> Dict[str, Optional[OrderBookSnapshot]]:
        """Get snapshots for all symbols."""
        return {
            symbol: ob.get_snapshot()
            for symbol, ob in self._order_books.items()
        }
    
    def reset(self) -> None:
        """Reset all order books."""
        for ob in self._order_books.values():
            ob.reset()
        self._order_books.clear()
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols."""
        return list(self._order_books.keys())
