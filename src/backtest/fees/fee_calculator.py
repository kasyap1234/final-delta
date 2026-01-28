"""
Fee calculator module for backtesting.

This module provides realistic fee structures with maker/taker tiers and funding rates.
Supports exchange-specific fee schedules and volume-based discounts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeeType(Enum):
    """Type of fee transaction."""
    MAKER = "maker"
    TAKER = "taker"
    FUNDING = "funding"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"


class OrderType(Enum):
    """Order type for fee calculation."""
    LIMIT = "limit"
    MARKET = "market"
    POST_ONLY = "post_only"


@dataclass
class FeeRecord:
    """Record of a fee transaction."""
    timestamp: datetime
    symbol: str
    fee_type: FeeType
    amount: float
    notional_value: float
    fee_rate: float
    fee_paid: float
    currency: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'fee_type': self.fee_type.value,
            'amount': self.amount,
            'notional_value': self.notional_value,
            'fee_rate': self.fee_rate,
            'fee_paid': self.fee_paid,
            'currency': self.currency,
            'metadata': self.metadata
        }


@dataclass
class VolumeTier:
    """Volume-based fee tier."""
    min_volume_30d: float
    maker_fee_rate: float
    taker_fee_rate: float
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'min_volume_30d': self.min_volume_30d,
            'maker_fee_rate': self.maker_fee_rate,
            'taker_fee_rate': self.taker_fee_rate,
            'name': self.name
        }


@dataclass
class FundingRateConfig:
    """Funding rate configuration."""
    enabled: bool = True
    default_rate_annual: float = 0.10  # 10% annual default
    rate_update_interval_hours: int = 8  # Funding every 8 hours
    max_rate_annual: float = 1.0  # 100% annual max
    min_rate_annual: float = -1.0  # -100% annual min
    rate_source: str = "synthetic"  # synthetic, historical, or fixed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'default_rate_annual': self.default_rate_annual,
            'rate_update_interval_hours': self.rate_update_interval_hours,
            'max_rate_annual': self.max_rate_annual,
            'min_rate_annual': self.min_rate_annual,
            'rate_source': self.rate_source
        }


@dataclass
class FeeSchedule:
    """Complete fee schedule for an exchange."""
    exchange_name: str
    default_maker_fee: float
    default_taker_fee: float
    volume_tiers: List[VolumeTier] = field(default_factory=list)
    funding_config: FundingRateConfig = field(default_factory=FundingRateConfig)
    withdrawal_fee_fixed: float = 0.0
    deposit_fee_percent: float = 0.0
    use_volume_tiers: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'exchange_name': self.exchange_name,
            'default_maker_fee': self.default_maker_fee,
            'default_taker_fee': self.default_taker_fee,
            'volume_tiers': [tier.to_dict() for tier in self.volume_tiers],
            'funding_config': self.funding_config.to_dict(),
            'withdrawal_fee_fixed': self.withdrawal_fee_fixed,
            'deposit_fee_percent': self.deposit_fee_percent,
            'use_volume_tiers': self.use_volume_tiers
        }


class FeeSchedulePresets:
    """Pre-configured fee schedules for popular exchanges."""
    
    @staticmethod
    def delta_exchange() -> FeeSchedule:
        """Delta Exchange fee schedule."""
        return FeeSchedule(
            exchange_name="delta",
            default_maker_fee=0.0002,  # 0.02%
            default_taker_fee=0.0006,  # 0.06%
            volume_tiers=[
                VolumeTier(0, 0.0002, 0.0006, "Tier 0"),
                VolumeTier(100000, 0.00015, 0.0005, "Tier 1"),
                VolumeTier(1000000, 0.0001, 0.0004, "Tier 2"),
                VolumeTier(10000000, 0.00005, 0.0003, "Tier 3"),
                VolumeTier(100000000, 0.00002, 0.00025, "Tier 4"),
            ],
            funding_config=FundingRateConfig(
                enabled=True,
                default_rate_annual=0.10,
                rate_update_interval_hours=8,
                max_rate_annual=0.50,
                min_rate_annual=-0.50
            )
        )
    
    @staticmethod
    def binance() -> FeeSchedule:
        """Binance fee schedule."""
        return FeeSchedule(
            exchange_name="binance",
            default_maker_fee=0.001,  # 0.1%
            default_taker_fee=0.001,  # 0.1%
            volume_tiers=[
                VolumeTier(0, 0.001, 0.001, "VIP 0"),
                VolumeTier(1000000, 0.0009, 0.001, "VIP 1"),
                VolumeTier(5000000, 0.0008, 0.001, "VIP 2"),
                VolumeTier(20000000, 0.0007, 0.0009, "VIP 3"),
                VolumeTier(100000000, 0.0006, 0.0008, "VIP 4"),
            ],
            funding_config=FundingRateConfig(
                enabled=True,
                default_rate_annual=0.10,
                rate_update_interval_hours=8,
                max_rate_annual=0.50,
                min_rate_annual=-0.50
            )
        )
    
    @staticmethod
    def bybit() -> FeeSchedule:
        """Bybit fee schedule."""
        return FeeSchedule(
            exchange_name="bybit",
            default_maker_fee=0.0001,  # 0.01%
            default_taker_fee=0.0006,  # 0.06%
            volume_tiers=[
                VolumeTier(0, 0.0001, 0.0006, "VIP 0"),
                VolumeTier(2000000, 0.0001, 0.00055, "VIP 1"),
                VolumeTier(10000000, 0.00008, 0.0005, "VIP 2"),
                VolumeTier(50000000, 0.00006, 0.00045, "VIP 3"),
            ],
            funding_config=FundingRateConfig(
                enabled=True,
                default_rate_annual=0.10,
                rate_update_interval_hours=8,
                max_rate_annual=0.50,
                min_rate_annual=-0.50
            )
        )
    
    @staticmethod
    def dydx() -> FeeSchedule:
        """dYdX fee schedule."""
        return FeeSchedule(
            exchange_name="dydx",
            default_maker_fee=-0.0002,  # -0.02% (rebate)
            default_taker_fee=0.0005,  # 0.05%
            volume_tiers=[
                VolumeTier(0, -0.0002, 0.0005, "Tier 1"),
                VolumeTier(1000000, -0.0003, 0.0004, "Tier 2"),
                VolumeTier(5000000, -0.0004, 0.00035, "Tier 3"),
                VolumeTier(25000000, -0.0005, 0.0003, "Tier 4"),
            ],
            funding_config=FundingRateConfig(
                enabled=True,
                default_rate_annual=0.10,
                rate_update_interval_hours=8,
                max_rate_annual=0.50,
                min_rate_annual=-0.50
            )
        )
    
    @staticmethod
    def get_preset(name: str) -> FeeSchedule:
        """Get a preset fee schedule by name."""
        presets = {
            'delta': FeeSchedulePresets.delta_exchange(),
            'delta_exchange': FeeSchedulePresets.delta_exchange(),
            'binance': FeeSchedulePresets.binance(),
            'bybit': FeeSchedulePresets.bybit(),
            'dydx': FeeSchedulePresets.dydx(),
        }
        if name.lower() not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return presets[name.lower()]


class FundingRateModel:
    """Model for calculating funding rates."""
    
    def __init__(self, config: FundingRateConfig):
        """
        Initialize funding rate model.
        
        Args:
            config: Funding rate configuration
        """
        self.config = config
        self._current_rates: Dict[str, float] = {}
        self._rate_history: Dict[str, List[Dict[str, Any]]] = {}
        self._last_update: Optional[datetime] = None
        self._rate_generator: Optional[Callable[[str, datetime], float]] = None
    
    def set_rate_generator(self, generator: Callable[[str, datetime], float]) -> None:
        """
        Set a custom rate generator function.
        
        Args:
            generator: Function that takes (symbol, timestamp) and returns annual rate
        """
        self._rate_generator = generator
    
    def get_funding_rate(self, symbol: str, timestamp: datetime) -> float:
        """
        Get the funding rate for a symbol at a given time.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Current timestamp
            
        Returns:
            Annual funding rate (e.g., 0.10 for 10%)
        """
        if not self.config.enabled:
            return 0.0
        
        # Check if we need to update rates
        if self._should_update_rates(timestamp):
            self._update_rates(symbol, timestamp)
        
        return self._current_rates.get(symbol, self.config.default_rate_annual)
    
    def _should_update_rates(self, timestamp: datetime) -> bool:
        """Check if funding rates should be updated."""
        if self._last_update is None:
            return True
        
        elapsed = timestamp - self._last_update
        return elapsed >= timedelta(hours=self.config.rate_update_interval_hours)
    
    def _update_rates(self, symbol: str, timestamp: datetime) -> None:
        """Update funding rates for all symbols."""
        if self._rate_generator:
            rate = self._rate_generator(symbol, timestamp)
        else:
            rate = self._generate_synthetic_rate(symbol, timestamp)
        
        # Clamp to min/max
        rate = max(self.config.min_rate_annual, min(self.config.max_rate_annual, rate))
        
        self._current_rates[symbol] = rate
        self._last_update = timestamp
        
        # Record history
        if symbol not in self._rate_history:
            self._rate_history[symbol] = []
        
        self._rate_history[symbol].append({
            'timestamp': timestamp.isoformat(),
            'rate_annual': rate,
            'rate_8h': rate * (self.config.rate_update_interval_hours / (24 * 365))
        })
    
    def _generate_synthetic_rate(self, symbol: str, timestamp: datetime) -> float:
        """Generate a synthetic funding rate based on market conditions."""
        import math
        import random
        
        # Base rate with some randomness
        base_rate = self.config.default_rate_annual
        
        # Add cyclical component (funding rates tend to oscillate)
        day_of_year = timestamp.timetuple().tm_yday
        cyclical = 0.05 * math.sin(2 * math.pi * day_of_year / 30)
        
        # Add noise
        noise = random.gauss(0, 0.02)
        
        return base_rate + cyclical + noise
    
    def calculate_funding_payment(
        self,
        symbol: str,
        position_size: float,
        mark_price: float,
        timestamp: datetime
    ) -> float:
        """
        Calculate funding payment for a position.
        
        Args:
            symbol: Trading pair symbol
            position_size: Position size (positive for long, negative for short)
            mark_price: Current mark price
            timestamp: Current timestamp
            
        Returns:
            Funding payment amount (positive = receive, negative = pay)
        """
        if not self.config.enabled or position_size == 0:
            return 0.0
        
        rate = self.get_funding_rate(symbol, timestamp)
        
        # Calculate 8-hour rate from annual rate
        hours = self.config.rate_update_interval_hours
        period_rate = rate * (hours / (24 * 365))
        
        # Notional value
        notional = abs(position_size) * mark_price
        
        # Longs pay shorts when rate is positive, shorts pay longs when negative
        if position_size > 0:
            # Long position
            payment = -notional * period_rate
        else:
            # Short position
            payment = notional * period_rate
        
        return payment
    
    def get_rate_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get funding rate history for a symbol."""
        return self._rate_history.get(symbol, [])


class FeeCalculator:
    """
    Calculator for trading fees with support for maker/taker tiers and funding rates.
    
    This class provides:
    - Maker/taker fee calculation based on order type
    - Volume-based tier discounts
    - Funding rate calculations for leveraged positions
    - Fee history tracking
    - Exchange-specific fee schedules
    """
    
    def __init__(
        self,
        fee_schedule: Optional[FeeSchedule] = None,
        quote_currency: str = "USD"
    ):
        """
        Initialize fee calculator.
        
        Args:
            fee_schedule: Fee schedule configuration. Uses Delta Exchange defaults if None.
            quote_currency: Quote currency for fee calculations
        """
        self.fee_schedule = fee_schedule or FeeSchedulePresets.delta_exchange()
        self.quote_currency = quote_currency
        
        # Volume tracking for tier calculation
        self._volume_30d: Dict[str, float] = {}
        self._total_volume_30d: float = 0.0
        self._current_tier: Optional[VolumeTier] = None
        
        # Fee history
        self._fee_history: List[FeeRecord] = []
        self._cumulative_fees: Dict[FeeType, float] = {
            FeeType.MAKER: 0.0,
            FeeType.TAKER: 0.0,
            FeeType.FUNDING: 0.0,
            FeeType.WITHDRAWAL: 0.0,
            FeeType.DEPOSIT: 0.0
        }
        
        # Funding rate model
        self._funding_model = FundingRateModel(self.fee_schedule.funding_config)
        
        # Current fee rates (may be discounted based on volume)
        self._current_maker_rate: float = self.fee_schedule.default_maker_fee
        self._current_taker_rate: float = self.fee_schedule.default_taker_fee
        
        logger.info(
            f"FeeCalculator initialized for {self.fee_schedule.exchange_name}: "
            f"maker={self._current_maker_rate:.4%}, taker={self._current_taker_rate:.4%}"
        )
    
    def calculate_trade_fee(
        self,
        symbol: str,
        amount: float,
        price: float,
        order_type: OrderType,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate trading fee for an order fill.
        
        Args:
            symbol: Trading pair symbol
            amount: Fill amount
            price: Fill price
            order_type: Type of order (limit, market, post_only)
            timestamp: Transaction timestamp
            metadata: Additional metadata
            
        Returns:
            Dictionary with fee details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Determine if maker or taker
        if order_type == OrderType.MARKET:
            fee_type = FeeType.TAKER
            fee_rate = self._current_taker_rate
        elif order_type == OrderType.POST_ONLY:
            fee_type = FeeType.MAKER
            fee_rate = self._current_maker_rate
        else:  # LIMIT
            # For limit orders, assume maker if resting, taker if crossing spread
            # In backtest, we can determine this from metadata or assume maker
            is_maker = metadata.get('is_maker', True) if metadata else True
            fee_type = FeeType.MAKER if is_maker else FeeType.TAKER
            fee_rate = self._current_maker_rate if is_maker else self._current_taker_rate
        
        # Calculate notional value
        notional_value = amount * price
        
        # Calculate fee
        fee_paid = notional_value * fee_rate
        
        # Record fee
        record = FeeRecord(
            timestamp=timestamp,
            symbol=symbol,
            fee_type=fee_type,
            amount=amount,
            notional_value=notional_value,
            fee_rate=fee_rate,
            fee_paid=fee_paid,
            currency=self.quote_currency,
            metadata=metadata or {}
        )
        self._fee_history.append(record)
        self._cumulative_fees[fee_type] += fee_paid
        
        # Update volume tracking
        self._update_volume(symbol, notional_value)
        
        logger.debug(
            f"Trade fee: {symbol} {fee_type.value} {fee_paid:.4f} "
            f"({fee_rate:.4%} of {notional_value:.2f})"
        )
        
        return {
            'fee_paid': fee_paid,
            'fee_rate': fee_rate,
            'fee_type': fee_type.value,
            'notional_value': notional_value,
            'is_maker': fee_type == FeeType.MAKER
        }
    
    def calculate_funding_fee(
        self,
        symbol: str,
        position_size: float,
        mark_price: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate funding fee for a position.
        
        Args:
            symbol: Trading pair symbol
            position_size: Position size (positive for long, negative for short)
            mark_price: Current mark price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with funding fee details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if position_size == 0:
            return {
                'fee_paid': 0.0,
                'fee_rate': 0.0,
                'funding_rate_annual': 0.0,
                'notional_value': 0.0,
                'is_payment': False
            }
        
        # Calculate funding payment
        payment = self._funding_model.calculate_funding_payment(
            symbol, position_size, mark_price, timestamp
        )
        
        # Get current funding rate
        funding_rate = self._funding_model.get_funding_rate(symbol, timestamp)
        
        # Determine if this is a payment (negative) or receipt (positive)
        is_payment = payment < 0
        notional_value = abs(position_size) * mark_price
        
        # Record fee (only if there's a payment)
        if payment != 0:
            record = FeeRecord(
                timestamp=timestamp,
                symbol=symbol,
                fee_type=FeeType.FUNDING,
                amount=abs(position_size),
                notional_value=notional_value,
                fee_rate=funding_rate,
                fee_paid=abs(payment),
                currency=self.quote_currency,
                metadata={
                    'position_side': 'long' if position_size > 0 else 'short',
                    'funding_rate_8h': funding_rate * (8 / (24 * 365)),
                    'is_payment': is_payment
                }
            )
            self._fee_history.append(record)
            self._cumulative_fees[FeeType.FUNDING] += abs(payment)
        
        logger.debug(
            f"Funding fee: {symbol} {'paid' if is_payment else 'received'} "
            f"{abs(payment):.4f} (rate: {funding_rate:.4%})"
        )
        
        return {
            'fee_paid': abs(payment),
            'fee_rate': funding_rate,
            'funding_rate_annual': funding_rate,
            'notional_value': notional_value,
            'is_payment': is_payment,
            'net_payment': payment  # Negative if paying, positive if receiving
        }
    
    def _update_volume(self, symbol: str, notional_value: float) -> None:
        """Update 30-day volume tracking."""
        if symbol not in self._volume_30d:
            self._volume_30d[symbol] = 0.0
        
        self._volume_30d[symbol] += notional_value
        self._total_volume_30d += notional_value
        
        # Recalculate tier if volume tiers are enabled
        if self.fee_schedule.use_volume_tiers:
            self._recalculate_tier()
    
    def _recalculate_tier(self) -> None:
        """Recalculate fee tier based on volume."""
        if not self.fee_schedule.volume_tiers:
            return
        
        # Find applicable tier
        applicable_tier = None
        for tier in sorted(self.fee_schedule.volume_tiers, key=lambda t: t.min_volume_30d):
            if self._total_volume_30d >= tier.min_volume_30d:
                applicable_tier = tier
        
        if applicable_tier and applicable_tier != self._current_tier:
            self._current_tier = applicable_tier
            self._current_maker_rate = applicable_tier.maker_fee_rate
            self._current_taker_rate = applicable_tier.taker_fee_rate
            
            logger.info(
                f"Fee tier upgraded to {applicable_tier.name}: "
                f"maker={self._current_maker_rate:.4%}, "
                f"taker={self._current_taker_rate:.4%}"
            )
    
    def get_current_fee_rates(self) -> Dict[str, float]:
        """
        Get current fee rates.
        
        Returns:
            Dictionary with maker and taker rates
        """
        return {
            'maker_rate': self._current_maker_rate,
            'taker_rate': self._current_taker_rate,
            'tier_name': self._current_tier.name if self._current_tier else 'Default'
        }
    
    def get_cumulative_fees(self, fee_type: Optional[FeeType] = None) -> Dict[str, float]:
        """
        Get cumulative fees paid.
        
        Args:
            fee_type: Optional filter by fee type
            
        Returns:
            Dictionary with cumulative fees
        """
        if fee_type:
            return {fee_type.value: self._cumulative_fees[fee_type]}
        
        return {
            'total': sum(self._cumulative_fees.values()),
            'maker': self._cumulative_fees[FeeType.MAKER],
            'taker': self._cumulative_fees[FeeType.TAKER],
            'funding': self._cumulative_fees[FeeType.FUNDING],
            'withdrawal': self._cumulative_fees[FeeType.WITHDRAWAL],
            'deposit': self._cumulative_fees[FeeType.DEPOSIT]
        }
    
    def get_fee_history(
        self,
        symbol: Optional[str] = None,
        fee_type: Optional[FeeType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[FeeRecord]:
        """
        Get fee history with optional filters.
        
        Args:
            symbol: Filter by symbol
            fee_type: Filter by fee type
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of fee records
        """
        records = self._fee_history
        
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        
        if fee_type:
            records = [r for r in records if r.fee_type == fee_type]
        
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]
        
        return records
    
    def get_volume_stats(self) -> Dict[str, Any]:
        """
        Get volume statistics.
        
        Returns:
            Dictionary with volume statistics
        """
        return {
            'total_volume_30d': self._total_volume_30d,
            'volume_by_symbol': self._volume_30d.copy(),
            'current_tier': self._current_tier.name if self._current_tier else 'Default',
            'next_tier_volume': self._get_next_tier_volume()
        }
    
    def _get_next_tier_volume(self) -> Optional[float]:
        """Get volume required for next tier."""
        if not self.fee_schedule.volume_tiers:
            return None
        
        current_min = self._current_tier.min_volume_30d if self._current_tier else 0
        
        for tier in sorted(self.fee_schedule.volume_tiers, key=lambda t: t.min_volume_30d):
            if tier.min_volume_30d > current_min:
                return tier.min_volume_30d
        
        return None
    
    def reset(self) -> None:
        """Reset all fee tracking state."""
        self._volume_30d.clear()
        self._total_volume_30d = 0.0
        self._current_tier = None
        self._fee_history.clear()
        self._cumulative_fees = {ft: 0.0 for ft in FeeType}
        self._current_maker_rate = self.fee_schedule.default_maker_fee
        self._current_taker_rate = self.fee_schedule.default_taker_fee
        self._funding_model = FundingRateModel(self.fee_schedule.funding_config)
        
        logger.info("FeeCalculator reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'fee_schedule': self.fee_schedule.to_dict(),
            'quote_currency': self.quote_currency,
            'current_rates': self.get_current_fee_rates(),
            'cumulative_fees': self.get_cumulative_fees(),
            'volume_stats': self.get_volume_stats(),
            'total_fee_records': len(self._fee_history)
        }
