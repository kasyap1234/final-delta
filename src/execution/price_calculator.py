"""
Price calculator module for optimal order pricing.

This module provides utilities for calculating optimal limit prices,
adjusting for slippage, and rounding to exchange tick sizes.
"""

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PriceCalculator:
    """
    Calculator for optimal order pricing.
    
    This class provides methods to:
    - Calculate optimal limit prices with offsets from market price
    - Adjust prices for slippage
    - Round prices to exchange tick sizes
    
    The calculator ensures post-only limit orders are placed at prices
    that won't immediately match (to ensure maker fees).
    
    Example:
        ```python
        calc = PriceCalculator()
        
        # For a buy order, place slightly below best bid
        limit_price = calc.calculate_limit_price(
            side='buy',
            current_price=50000,
            offset_percent=0.01
        )
        # Returns: 49995.0 (0.01% below market)
        
        # Round to tick size
        rounded = calc.round_to_tick_size(49995.3, tick_size=0.5)
        # Returns: 49995.0
        ```
    """
    
    def __init__(self, default_offset_percent: float = 0.01):
        """
        Initialize the price calculator.
        
        Args:
            default_offset_percent: Default price offset percentage from market price.
                                   Default is 0.01% (0.0001 in decimal).
        """
        self.default_offset_percent = default_offset_percent
    
    def calculate_limit_price(
        self,
        side: str,
        current_price: float,
        offset_percent: Optional[float] = None,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None
    ) -> float:
        """
        Calculate optimal limit price for post-only orders.
        
        For BUY orders: Places order slightly BELOW best bid (or current price)
                       to ensure it doesn't immediately match.
        For SELL orders: Places order slightly ABOVE best ask (or current price)
                        to ensure it doesn't immediately match.
        
        Args:
            side: Order side ('buy' or 'sell').
            current_price: Current market price (used if best_bid/ask not provided).
            offset_percent: Price offset percentage from market. Uses default if None.
            best_bid: Best bid price from order book (optional, preferred for buys).
            best_ask: Best ask price from order book (optional, preferred for sells).
        
        Returns:
            Optimal limit price for post-only order.
        
        Raises:
            ValueError: If side is not 'buy' or 'sell'.
        """
        side = side.lower()
        if side not in ('buy', 'sell'):
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
        
        offset = offset_percent if offset_percent is not None else self.default_offset_percent
        offset_multiplier = offset / 100.0  # Convert percentage to decimal
        
        if side == 'buy':
            # For buy orders, place slightly below best bid (or current price)
            reference_price = best_bid if best_bid is not None else current_price
            limit_price = reference_price * (1 - offset_multiplier)
            logger.debug(
                f"Calculated buy limit price: {limit_price:.8f} "
                f"({offset}% below reference: {reference_price})"
            )
        else:  # sell
            # For sell orders, place slightly above best ask (or current price)
            reference_price = best_ask if best_ask is not None else current_price
            limit_price = reference_price * (1 + offset_multiplier)
            logger.debug(
                f"Calculated sell limit price: {limit_price:.8f} "
                f"({offset}% above reference: {reference_price})"
            )
        
        return limit_price
    
    def calculate_post_only_price(
        self,
        side: str,
        best_bid: float,
        best_ask: float,
        offset_percent: Optional[float] = None,
        tick_size: Optional[float] = None
    ) -> float:
        """
        Calculate post-only limit price using order book data.
        
        This is the preferred method when order book data is available,
        as it ensures the order will be a maker order (post-only).
        
        Args:
            side: Order side ('buy' or 'sell').
            best_bid: Best bid price from order book.
            best_ask: Best ask price from order book.
            offset_percent: Price offset percentage. Uses default if None.
            tick_size: Exchange tick size for rounding (optional).
        
        Returns:
            Post-only limit price.
        
        Raises:
            ValueError: If side is not 'buy' or 'sell'.
        """
        side = side.lower()
        if side not in ('buy', 'sell'):
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
        
        offset = offset_percent if offset_percent is not None else self.default_offset_percent
        offset_multiplier = offset / 100.0
        
        if side == 'buy':
            # Place buy order slightly below best bid
            # This ensures we're adding liquidity, not taking it
            limit_price = best_bid * (1 - offset_multiplier)
            # Ensure we don't cross the spread
            limit_price = min(limit_price, best_bid)
        else:  # sell
            # Place sell order slightly above best ask
            limit_price = best_ask * (1 + offset_multiplier)
            # Ensure we don't cross the spread
            limit_price = max(limit_price, best_ask)
        
        if tick_size is not None:
            limit_price = self.round_to_tick_size(limit_price, tick_size, side)
        
        logger.debug(
            f"Calculated post-only {side} price: {limit_price:.8f} "
            f"(spread: {best_bid} - {best_ask})"
        )
        
        return limit_price
    
    def adjust_for_slippage(
        self,
        price: float,
        side: str,
        slippage_percent: float
    ) -> float:
        """
        Adjust price for slippage tolerance.
        
        This is used when placing market orders or when we need to
        account for potential price movement.
        
        Args:
            price: Original price.
            side: Order side ('buy' or 'sell').
            slippage_percent: Maximum acceptable slippage percentage.
        
        Returns:
            Price adjusted for slippage.
        
        Raises:
            ValueError: If side is not 'buy' or 'sell'.
        """
        side = side.lower()
        if side not in ('buy', 'sell'):
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
        
        slippage_multiplier = slippage_percent / 100.0
        
        if side == 'buy':
            # For buys, slippage means paying more
            adjusted_price = price * (1 + slippage_multiplier)
        else:  # sell
            # For sells, slippage means receiving less
            adjusted_price = price * (1 - slippage_multiplier)
        
        logger.debug(
            f"Price adjusted for slippage: {adjusted_price:.8f} "
            f"(original: {price}, slippage: {slippage_percent}%)"
        )
        
        return adjusted_price
    
    def round_to_tick_size(
        self,
        price: float,
        tick_size: float,
        side: Optional[str] = None
    ) -> float:
        """
        Round price to exchange tick size.
        
        Args:
            price: Price to round.
            tick_size: Exchange tick size (e.g., 0.5, 0.01, 0.0001).
            side: Order side ('buy' or 'sell') for directional rounding.
                  Buy orders round down, sell orders round up to ensure
                  post-only behavior is maintained.
        
        Returns:
            Price rounded to tick size.
        
        Raises:
            ValueError: If tick_size is zero or negative.
        """
        if tick_size <= 0:
            raise ValueError(f"Tick size must be positive, got {tick_size}")
        
        # Use Decimal for precise arithmetic
        price_decimal = Decimal(str(price))
        tick_decimal = Decimal(str(tick_size))
        
        if side:
            side = side.lower()
            if side == 'buy':
                # Round down for buy orders (more conservative)
                rounded = (price_decimal // tick_decimal) * tick_decimal
            elif side == 'sell':
                # Round up for sell orders (more conservative)
                rounded = ((price_decimal + tick_decimal - 1) // tick_decimal) * tick_decimal
            else:
                # Default rounding for invalid side
                rounded = (price_decimal / tick_decimal).quantize(
                    Decimal('1'), rounding=ROUND_DOWN
                ) * tick_decimal
        else:
            # Default: round to nearest tick
            rounded = (price_decimal / tick_decimal).quantize(
                Decimal('1'), rounding=ROUND_DOWN
            ) * tick_decimal
        
        return float(rounded)
    
    def get_tick_size_from_market(
        self,
        market_info: Dict[str, Any]
    ) -> Optional[float]:
        """
        Extract tick size from market information.
        
        Args:
            market_info: Market information dictionary from exchange.
        
        Returns:
            Tick size if available, None otherwise.
        """
        # Try common field names for tick size
        tick_size = (
            market_info.get('tickSize') or
            market_info.get('tick_size') or
            market_info.get('precision', {}).get('price')
        )
        
        if tick_size is not None:
            try:
                return float(tick_size)
            except (ValueError, TypeError):
                logger.warning(f"Invalid tick size value: {tick_size}")
        
        return None
    
    def calculate_fee_estimate(
        self,
        amount: float,
        price: float,
        is_maker: bool = True,
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0005   # 0.05%
    ) -> Dict[str, float]:
        """
        Calculate estimated trading fees.
        
        Args:
            amount: Order amount in base currency.
            price: Order price.
            is_maker: True if maker order, False if taker.
            maker_fee: Maker fee rate (default 0.02%).
            taker_fee: Taker fee rate (default 0.05%).
        
        Returns:
            Dictionary with fee details.
        """
        notional_value = amount * price
        fee_rate = maker_fee if is_maker else taker_fee
        fee_amount = notional_value * fee_rate
        
        return {
            'notional_value': notional_value,
            'fee_rate': fee_rate,
            'fee_amount': fee_amount,
            'is_maker': is_maker,
            'net_value': notional_value - fee_amount if is_maker else notional_value + fee_amount
        }
    
    def is_price_valid_for_post_only(
        self,
        side: str,
        price: float,
        best_bid: float,
        best_ask: float
    ) -> bool:
        """
        Check if a price is valid for a post-only order.
        
        A post-only order is valid if:
        - Buy order: price < best_ask (won't cross the spread)
        - Sell order: price > best_bid (won't cross the spread)
        
        Args:
            side: Order side ('buy' or 'sell').
            price: Order price.
            best_bid: Best bid price from order book.
            best_ask: Best ask price from order book.
        
        Returns:
            True if price is valid for post-only, False otherwise.
        """
        side = side.lower()
        if side not in ('buy', 'sell'):
            return False
        
        if side == 'buy':
            # Buy order must be below best ask to be post-only
            return price < best_ask
        else:  # sell
            # Sell order must be above best bid to be post-only
            return price > best_bid
