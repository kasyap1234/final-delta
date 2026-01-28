"""
Data models for the trading bot database.

This module defines dataclasses for all database entities including trades,
signals, positions, orders, hedges, market data, correlations, performance metrics,
and balance history.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import uuid


class TradeSide(Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class SignalType(Enum):
    """Signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    PO = "PO"    # Post Only


class HedgeStatus(Enum):
    """Hedge status enumeration."""
    ACTIVE = "active"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


class CorrelationType(Enum):
    """Correlation type enumeration."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class MetricType(Enum):
    """Performance metric type enumeration."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TRADE = "trade"
    OVERALL = "overall"


class LogLevel(Enum):
    """System log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{unique_id}" if prefix else unique_id


def to_json(data: Dict[str, Any]) -> Optional[str]:
    """Convert dictionary to JSON string."""
    if data is None:
        return None
    return json.dumps(data, default=str)


def from_json(json_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Convert JSON string to dictionary."""
    if json_str is None:
        return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


@dataclass
class Trade:
    """Trade execution record model."""
    
    symbol: str
    side: TradeSide
    entry_price: float
    quantity: float
    entry_time: datetime
    trade_id: str = field(default_factory=lambda: generate_id("trade"))
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    fees: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    strategy: Optional[str] = None
    signal_ids: Optional[List[str]] = None
    position_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for database storage."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, TradeSide) else self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else self.entry_time,
            'exit_time': self.exit_time.isoformat() if isinstance(self.exit_time, datetime) else self.exit_time,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'fees': self.fees,
            'status': self.status.value if isinstance(self.status, TradeStatus) else self.status,
            'strategy': self.strategy,
            'signal_ids': json.dumps(self.signal_ids) if self.signal_ids else None,
            'position_id': self.position_id,
            'metadata': to_json(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create Trade instance from dictionary."""
        return cls(
            trade_id=data.get('trade_id', generate_id("trade")),
            symbol=data['symbol'],
            side=TradeSide(data['side']) if isinstance(data['side'], str) else data['side'],
            entry_price=data['entry_price'],
            exit_price=data.get('exit_price'),
            quantity=data['quantity'],
            entry_time=datetime.fromisoformat(data['entry_time']) if isinstance(data['entry_time'], str) else data['entry_time'],
            exit_time=datetime.fromisoformat(data['exit_time']) if isinstance(data.get('exit_time'), str) else data.get('exit_time'),
            pnl=data.get('pnl'),
            pnl_percent=data.get('pnl_percent'),
            fees=data.get('fees', 0.0),
            status=TradeStatus(data['status']) if isinstance(data['status'], str) else data['status'],
            strategy=data.get('strategy'),
            signal_ids=json.loads(data['signal_ids']) if data.get('signal_ids') else None,
            position_id=data.get('position_id'),
            metadata=from_json(data.get('metadata'))
        )


@dataclass
class Signal:
    """Trading signal model."""
    
    symbol: str
    signal_type: SignalType
    strength: float
    price_at_signal: float
    generated_at: datetime = field(default_factory=datetime.now)
    signal_id: str = field(default_factory=lambda: generate_id("signal"))
    indicators: Optional[Dict[str, Any]] = None
    volume_at_signal: Optional[float] = None
    timeframe: Optional[str] = None
    strategy: Optional[str] = None
    confidence: Optional[float] = None
    executed: bool = False
    trade_id: Optional[str] = None
    outcome: Optional[str] = None
    pnl: Optional[float] = None
    executed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate signal strength is between 0 and 100."""
        if not 0 <= self.strength <= 100:
            raise ValueError("Signal strength must be between 0 and 100")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for database storage."""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value if isinstance(self.signal_type, SignalType) else self.signal_type,
            'strength': self.strength,
            'indicators': to_json(self.indicators),
            'price_at_signal': self.price_at_signal,
            'volume_at_signal': self.volume_at_signal,
            'timeframe': self.timeframe,
            'strategy': self.strategy,
            'confidence': self.confidence,
            'executed': self.executed,
            'trade_id': self.trade_id,
            'outcome': self.outcome,
            'pnl': self.pnl,
            'generated_at': self.generated_at.isoformat() if isinstance(self.generated_at, datetime) else self.generated_at,
            'executed_at': self.executed_at.isoformat() if isinstance(self.executed_at, datetime) else self.executed_at,
            'expires_at': self.expires_at.isoformat() if isinstance(self.expires_at, datetime) else self.expires_at,
            'metadata': to_json(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create Signal instance from dictionary."""
        return cls(
            signal_id=data.get('signal_id', generate_id("signal")),
            symbol=data['symbol'],
            signal_type=SignalType(data['signal_type']) if isinstance(data['signal_type'], str) else data['signal_type'],
            strength=data['strength'],
            indicators=from_json(data.get('indicators')),
            price_at_signal=data['price_at_signal'],
            volume_at_signal=data.get('volume_at_signal'),
            timeframe=data.get('timeframe'),
            strategy=data.get('strategy'),
            confidence=data.get('confidence'),
            executed=data.get('executed', False),
            trade_id=data.get('trade_id'),
            outcome=data.get('outcome'),
            pnl=data.get('pnl'),
            generated_at=datetime.fromisoformat(data['generated_at']) if isinstance(data['generated_at'], str) else data['generated_at'],
            executed_at=datetime.fromisoformat(data['executed_at']) if isinstance(data.get('executed_at'), str) else data.get('executed_at'),
            expires_at=datetime.fromisoformat(data['expires_at']) if isinstance(data.get('expires_at'), str) else data.get('expires_at'),
            metadata=from_json(data.get('metadata'))
        )


@dataclass
class Position:
    """Position tracking model."""
    
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    opened_at: datetime = field(default_factory=datetime.now)
    position_id: str = field(default_factory=lambda: generate_id("pos"))
    current_price: Optional[float] = None
    mark_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    liquidation_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_used: Optional[float] = None
    leverage: float = 1.0
    status: PositionStatus = PositionStatus.OPEN
    hedge_id: Optional[str] = None
    closed_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def calculate_unrealized_pnl(self, current_price: Optional[float] = None) -> float:
        """Calculate unrealized P&L based on current price."""
        price = current_price or self.current_price
        if price is None:
            return 0.0
        
        if self.side == PositionSide.LONG:
            return (price - self.entry_price) * self.size
        else:
            return (self.entry_price - price) * self.size
    
    def calculate_unrealized_pnl_percent(self, current_price: Optional[float] = None) -> float:
        """Calculate unrealized P&L percentage."""
        price = current_price or self.current_price
        if price is None or self.entry_price == 0:
            return 0.0
        
        if self.side == PositionSide.LONG:
            return ((price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for database storage."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, PositionSide) else self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'mark_price': self.mark_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'liquidation_price': self.liquidation_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'margin_used': self.margin_used,
            'leverage': self.leverage,
            'status': self.status.value if isinstance(self.status, PositionStatus) else self.status,
            'hedge_id': self.hedge_id,
            'opened_at': self.opened_at.isoformat() if isinstance(self.opened_at, datetime) else self.opened_at,
            'closed_at': self.closed_at.isoformat() if isinstance(self.closed_at, datetime) else self.closed_at,
            'last_updated': self.last_updated.isoformat() if isinstance(self.last_updated, datetime) else self.last_updated,
            'metadata': to_json(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position instance from dictionary."""
        return cls(
            position_id=data.get('position_id', generate_id("pos")),
            symbol=data['symbol'],
            side=PositionSide(data['side']) if isinstance(data['side'], str) else data['side'],
            size=data['size'],
            entry_price=data['entry_price'],
            current_price=data.get('current_price'),
            mark_price=data.get('mark_price'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            liquidation_price=data.get('liquidation_price'),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            realized_pnl=data.get('realized_pnl', 0.0),
            margin_used=data.get('margin_used'),
            leverage=data.get('leverage', 1.0),
            status=PositionStatus(data['status']) if isinstance(data['status'], str) else data['status'],
            hedge_id=data.get('hedge_id'),
            opened_at=datetime.fromisoformat(data['opened_at']) if isinstance(data['opened_at'], str) else data['opened_at'],
            closed_at=datetime.fromisoformat(data['closed_at']) if isinstance(data.get('closed_at'), str) else data.get('closed_at'),
            last_updated=datetime.fromisoformat(data['last_updated']) if isinstance(data.get('last_updated'), str) else data.get('last_updated'),
            metadata=from_json(data.get('metadata'))
        )


@dataclass
class Order:
    """Order execution details model."""
    
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    order_id: str = field(default_factory=lambda: generate_id("order"))
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    price: Optional[float] = None
    average_fill_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: Optional[TimeInForce] = None
    fees: float = 0.0
    fee_currency: Optional[str] = None
    trade_id: Optional[str] = None
    position_id: Optional[str] = None
    signal_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate remaining quantity if not provided."""
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity
    
    @property
    def fill_percent(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for database storage."""
        return {
            'order_id': self.order_id,
            'exchange_order_id': self.exchange_order_id,
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, OrderSide) else self.side,
            'order_type': self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'price': self.price,
            'average_fill_price': self.average_fill_price,
            'stop_price': self.stop_price,
            'status': self.status.value if isinstance(self.status, OrderStatus) else self.status,
            'time_in_force': self.time_in_force.value if isinstance(self.time_in_force, TimeInForce) else self.time_in_force,
            'fees': self.fees,
            'fee_currency': self.fee_currency,
            'trade_id': self.trade_id,
            'position_id': self.position_id,
            'signal_id': self.signal_id,
            'submitted_at': self.submitted_at.isoformat() if isinstance(self.submitted_at, datetime) else self.submitted_at,
            'filled_at': self.filled_at.isoformat() if isinstance(self.filled_at, datetime) else self.filled_at,
            'cancelled_at': self.cancelled_at.isoformat() if isinstance(self.cancelled_at, datetime) else self.cancelled_at,
            'metadata': to_json(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create Order instance from dictionary."""
        return cls(
            order_id=data.get('order_id', generate_id("order")),
            exchange_order_id=data.get('exchange_order_id'),
            symbol=data['symbol'],
            side=OrderSide(data['side']) if isinstance(data['side'], str) else data['side'],
            order_type=OrderType(data['order_type']) if isinstance(data['order_type'], str) else data['order_type'],
            quantity=data['quantity'],
            filled_quantity=data.get('filled_quantity', 0.0),
            remaining_quantity=data.get('remaining_quantity'),
            price=data.get('price'),
            average_fill_price=data.get('average_fill_price'),
            stop_price=data.get('stop_price'),
            status=OrderStatus(data['status']) if isinstance(data['status'], str) else data['status'],
            time_in_force=TimeInForce(data['time_in_force']) if isinstance(data.get('time_in_force'), str) else data.get('time_in_force'),
            fees=data.get('fees', 0.0),
            fee_currency=data.get('fee_currency'),
            trade_id=data.get('trade_id'),
            position_id=data.get('position_id'),
            signal_id=data.get('signal_id'),
            submitted_at=datetime.fromisoformat(data['submitted_at']) if isinstance(data.get('submitted_at'), str) else data.get('submitted_at'),
            filled_at=datetime.fromisoformat(data['filled_at']) if isinstance(data.get('filled_at'), str) else data.get('filled_at'),
            cancelled_at=datetime.fromisoformat(data['cancelled_at']) if isinstance(data.get('cancelled_at'), str) else data.get('cancelled_at'),
            metadata=from_json(data.get('metadata'))
        )


@dataclass
class Hedge:
    """Hedge position record model."""
    
    primary_position_id: str
    hedge_position_id: str
    primary_symbol: str
    hedge_symbol: str
    hedge_ratio: float
    primary_size: float
    hedge_size: float
    opened_at: datetime = field(default_factory=datetime.now)
    hedge_id: str = field(default_factory=lambda: generate_id("hedge"))
    correlation_at_hedge: Optional[float] = None
    status: HedgeStatus = HedgeStatus.ACTIVE
    pnl: float = 0.0
    primary_pnl: float = 0.0
    hedge_pnl: float = 0.0
    closed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def calculate_total_pnl(self) -> float:
        """Calculate total P&L from both positions."""
        return self.primary_pnl + self.hedge_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hedge to dictionary for database storage."""
        return {
            'hedge_id': self.hedge_id,
            'primary_position_id': self.primary_position_id,
            'hedge_position_id': self.hedge_position_id,
            'primary_symbol': self.primary_symbol,
            'hedge_symbol': self.hedge_symbol,
            'correlation_at_hedge': self.correlation_at_hedge,
            'hedge_ratio': self.hedge_ratio,
            'primary_size': self.primary_size,
            'hedge_size': self.hedge_size,
            'status': self.status.value if isinstance(self.status, HedgeStatus) else self.status,
            'pnl': self.pnl,
            'primary_pnl': self.primary_pnl,
            'hedge_pnl': self.hedge_pnl,
            'opened_at': self.opened_at.isoformat() if isinstance(self.opened_at, datetime) else self.opened_at,
            'closed_at': self.closed_at.isoformat() if isinstance(self.closed_at, datetime) else self.closed_at,
            'metadata': to_json(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hedge':
        """Create Hedge instance from dictionary."""
        return cls(
            hedge_id=data.get('hedge_id', generate_id("hedge")),
            primary_position_id=data['primary_position_id'],
            hedge_position_id=data['hedge_position_id'],
            primary_symbol=data['primary_symbol'],
            hedge_symbol=data['hedge_symbol'],
            correlation_at_hedge=data.get('correlation_at_hedge'),
            hedge_ratio=data['hedge_ratio'],
            primary_size=data['primary_size'],
            hedge_size=data['hedge_size'],
            status=HedgeStatus(data['status']) if isinstance(data['status'], str) else data['status'],
            pnl=data.get('pnl', 0.0),
            primary_pnl=data.get('primary_pnl', 0.0),
            hedge_pnl=data.get('hedge_pnl', 0.0),
            opened_at=datetime.fromisoformat(data['opened_at']) if isinstance(data['opened_at'], str) else data['opened_at'],
            closed_at=datetime.fromisoformat(data['closed_at']) if isinstance(data.get('closed_at'), str) else data.get('closed_at'),
            metadata=from_json(data.get('metadata'))
        )


@dataclass
class MarketData:
    """OHLCV market data model."""
    
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    taker_buy_volume: Optional[float] = None
    taker_buy_quote_volume: Optional[float] = None
    
    @property
    def range(self) -> float:
        """Calculate price range (high - low)."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Calculate candle body (close - open)."""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open)."""
        return self.close < self.open
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'trades_count': self.trades_count,
            'taker_buy_volume': self.taker_buy_volume,
            'taker_buy_quote_volume': self.taker_buy_quote_volume
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create MarketData instance from dictionary."""
        return cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            quote_volume=data.get('quote_volume'),
            trades_count=data.get('trades_count'),
            taker_buy_volume=data.get('taker_buy_volume'),
            taker_buy_quote_volume=data.get('taker_buy_quote_volume')
        )


@dataclass
class Correlation:
    """Correlation calculation model."""
    
    symbol_a: str
    symbol_b: str
    correlation: float
    correlation_type: CorrelationType
    timeframe: str
    period: int
    calculated_at: datetime = field(default_factory=datetime.now)
    price_data_range_start: Optional[datetime] = None
    price_data_range_end: Optional[datetime] = None
    p_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate correlation is between -1 and 1."""
        if not -1 <= self.correlation <= 1:
            raise ValueError("Correlation must be between -1 and 1")
    
    @property
    def is_strong_positive(self) -> bool:
        """Check if correlation is strongly positive (> 0.7)."""
        return self.correlation > 0.7
    
    @property
    def is_strong_negative(self) -> bool:
        """Check if correlation is strongly negative (< -0.7)."""
        return self.correlation < -0.7
    
    @property
    def is_weak(self) -> bool:
        """Check if correlation is weak (|correlation| < 0.3)."""
        return abs(self.correlation) < 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert correlation to dictionary for database storage."""
        return {
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'correlation': self.correlation,
            'correlation_type': self.correlation_type.value if isinstance(self.correlation_type, CorrelationType) else self.correlation_type,
            'timeframe': self.timeframe,
            'period': self.period,
            'calculated_at': self.calculated_at.isoformat() if isinstance(self.calculated_at, datetime) else self.calculated_at,
            'price_data_range_start': self.price_data_range_start.isoformat() if isinstance(self.price_data_range_start, datetime) else self.price_data_range_start,
            'price_data_range_end': self.price_data_range_end.isoformat() if isinstance(self.price_data_range_end, datetime) else self.price_data_range_end,
            'p_value': self.p_value,
            'metadata': to_json(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Correlation':
        """Create Correlation instance from dictionary."""
        return cls(
            symbol_a=data['symbol_a'],
            symbol_b=data['symbol_b'],
            correlation=data['correlation'],
            correlation_type=CorrelationType(data['correlation_type']) if isinstance(data['correlation_type'], str) else data['correlation_type'],
            timeframe=data['timeframe'],
            period=data['period'],
            calculated_at=datetime.fromisoformat(data['calculated_at']) if isinstance(data['calculated_at'], str) else data['calculated_at'],
            price_data_range_start=datetime.fromisoformat(data['price_data_range_start']) if isinstance(data.get('price_data_range_start'), str) else data.get('price_data_range_start'),
            price_data_range_end=datetime.fromisoformat(data['price_data_range_end']) if isinstance(data.get('price_data_range_end'), str) else data.get('price_data_range_end'),
            p_value=data.get('p_value'),
            metadata=from_json(data.get('metadata'))
        )


@dataclass
class Performance:
    """Performance metrics model."""
    
    metric_type: MetricType
    period_start: datetime
    period_end: datetime
    metric_id: str = field(default_factory=lambda: generate_id("perf"))
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    average_win: Optional[float] = None
    average_loss: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_percent: Optional[float] = None
    volatility: Optional[float] = None
    return_on_investment: Optional[float] = None
    return_on_investment_percent: Optional[float] = None
    starting_balance: Optional[float] = None
    ending_balance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived metrics if not provided."""
        if self.win_rate is None and self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        
        if self.profit_factor is None and self.gross_loss != 0:
            self.profit_factor = abs(self.gross_profit / self.gross_loss) if self.gross_loss != 0 else float('inf')
        
        if self.average_win is None and self.winning_trades > 0:
            self.average_win = self.gross_profit / self.winning_trades
        
        if self.average_loss is None and self.losing_trades > 0:
            self.average_loss = self.gross_loss / self.losing_trades
    
    @property
    def net_pnl(self) -> float:
        """Calculate net P&L (total P&L minus any fees if tracked separately)."""
        return self.total_pnl
    
    @property
    def expectancy(self) -> Optional[float]:
        """Calculate trading expectancy."""
        if self.win_rate is None or self.average_win is None or self.average_loss is None:
            return None
        win_rate_decimal = self.win_rate / 100
        return (win_rate_decimal * self.average_win) + ((1 - win_rate_decimal) * self.average_loss)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance to dictionary for database storage."""
        return {
            'metric_id': self.metric_id,
            'metric_type': self.metric_type.value if isinstance(self.metric_type, MetricType) else self.metric_type,
            'period_start': self.period_start.isoformat() if isinstance(self.period_start, datetime) else self.period_start,
            'period_end': self.period_end.isoformat() if isinstance(self.period_end, datetime) else self.period_end,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'volatility': self.volatility,
            'return_on_investment': self.return_on_investment,
            'return_on_investment_percent': self.return_on_investment_percent,
            'starting_balance': self.starting_balance,
            'ending_balance': self.ending_balance,
            'metadata': to_json(self.metadata),
            'calculated_at': self.calculated_at.isoformat() if isinstance(self.calculated_at, datetime) else self.calculated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Performance':
        """Create Performance instance from dictionary."""
        return cls(
            metric_id=data.get('metric_id', generate_id("perf")),
            metric_type=MetricType(data['metric_type']) if isinstance(data['metric_type'], str) else data['metric_type'],
            period_start=datetime.fromisoformat(data['period_start']) if isinstance(data['period_start'], str) else data['period_start'],
            period_end=datetime.fromisoformat(data['period_end']) if isinstance(data['period_end'], str) else data['period_end'],
            total_trades=data.get('total_trades', 0),
            winning_trades=data.get('winning_trades', 0),
            losing_trades=data.get('losing_trades', 0),
            total_pnl=data.get('total_pnl', 0.0),
            gross_profit=data.get('gross_profit', 0.0),
            gross_loss=data.get('gross_loss', 0.0),
            win_rate=data.get('win_rate'),
            profit_factor=data.get('profit_factor'),
            average_win=data.get('average_win'),
            average_loss=data.get('average_loss'),
            largest_win=data.get('largest_win'),
            largest_loss=data.get('largest_loss'),
            sharpe_ratio=data.get('sharpe_ratio'),
            sortino_ratio=data.get('sortino_ratio'),
            max_drawdown=data.get('max_drawdown'),
            max_drawdown_percent=data.get('max_drawdown_percent'),
            volatility=data.get('volatility'),
            return_on_investment=data.get('return_on_investment'),
            return_on_investment_percent=data.get('return_on_investment_percent'),
            starting_balance=data.get('starting_balance'),
            ending_balance=data.get('ending_balance'),
            metadata=from_json(data.get('metadata')),
            calculated_at=datetime.fromisoformat(data['calculated_at']) if isinstance(data.get('calculated_at'), str) else data.get('calculated_at', datetime.now())
        )


@dataclass
class BalanceHistory:
    """Account balance history model."""
    
    total_balance: float
    available_balance: float
    timestamp: datetime = field(default_factory=datetime.now)
    balance_id: str = field(default_factory=lambda: generate_id("bal"))
    margin_balance: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    realized_pnl_week: float = 0.0
    realized_pnl_month: float = 0.0
    currency: str = "USD"
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def in_use_balance(self) -> float:
        """Calculate balance in use (total - available)."""
        return self.total_balance - self.available_balance
    
    @property
    def margin_utilization(self) -> Optional[float]:
        """Calculate margin utilization percentage."""
        if self.margin_balance is None or self.margin_balance == 0:
            return None
        return ((self.margin_balance - self.available_balance) / self.margin_balance) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert balance history to dictionary for database storage."""
        return {
            'balance_id': self.balance_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'total_balance': self.total_balance,
            'available_balance': self.available_balance,
            'margin_balance': self.margin_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl_today': self.realized_pnl_today,
            'realized_pnl_week': self.realized_pnl_week,
            'realized_pnl_month': self.realized_pnl_month,
            'currency': self.currency,
            'source': self.source,
            'metadata': to_json(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BalanceHistory':
        """Create BalanceHistory instance from dictionary."""
        return cls(
            balance_id=data.get('balance_id', generate_id("bal")),
            total_balance=data['total_balance'],
            available_balance=data['available_balance'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            margin_balance=data.get('margin_balance'),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            realized_pnl_today=data.get('realized_pnl_today', 0.0),
            realized_pnl_week=data.get('realized_pnl_week', 0.0),
            realized_pnl_month=data.get('realized_pnl_month', 0.0),
            currency=data.get('currency', 'USD'),
            source=data.get('source'),
            metadata=from_json(data.get('metadata'))
        )


@dataclass
class TradeJournal:
    """Detailed trade journal entry model."""
    
    trade_id: str
    journal_id: str = field(default_factory=lambda: generate_id("journal"))
    entry_notes: Optional[str] = None
    exit_notes: Optional[str] = None
    lessons_learned: Optional[str] = None
    emotional_state: Optional[str] = None
    market_conditions: Optional[str] = None
    setup_quality: Optional[int] = None
    execution_quality: Optional[int] = None
    tags: Optional[List[str]] = None
    screenshots: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate quality ratings are between 1 and 10."""
        if self.setup_quality is not None and not 1 <= self.setup_quality <= 10:
            raise ValueError("Setup quality must be between 1 and 10")
        if self.execution_quality is not None and not 1 <= self.execution_quality <= 10:
            raise ValueError("Execution quality must be between 1 and 10")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade journal to dictionary for database storage."""
        return {
            'journal_id': self.journal_id,
            'trade_id': self.trade_id,
            'entry_notes': self.entry_notes,
            'exit_notes': self.exit_notes,
            'lessons_learned': self.lessons_learned,
            'emotional_state': self.emotional_state,
            'market_conditions': self.market_conditions,
            'setup_quality': self.setup_quality,
            'execution_quality': self.execution_quality,
            'tags': json.dumps(self.tags) if self.tags else None,
            'screenshots': json.dumps(self.screenshots) if self.screenshots else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeJournal':
        """Create TradeJournal instance from dictionary."""
        return cls(
            journal_id=data.get('journal_id', generate_id("journal")),
            trade_id=data['trade_id'],
            entry_notes=data.get('entry_notes'),
            exit_notes=data.get('exit_notes'),
            lessons_learned=data.get('lessons_learned'),
            emotional_state=data.get('emotional_state'),
            market_conditions=data.get('market_conditions'),
            setup_quality=data.get('setup_quality'),
            execution_quality=data.get('execution_quality'),
            tags=json.loads(data['tags']) if data.get('tags') else None,
            screenshots=json.loads(data['screenshots']) if data.get('screenshots') else None
        )


@dataclass
class SystemLog:
    """System log entry model."""
    
    level: LogLevel
    component: str
    message: str
    log_id: str = field(default_factory=lambda: generate_id("log"))
    details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system log to dictionary for database storage."""
        return {
            'log_id': self.log_id,
            'level': self.level.value if isinstance(self.level, LogLevel) else self.level,
            'component': self.component,
            'message': self.message,
            'details': to_json(self.details),
            'traceback': self.traceback,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemLog':
        """Create SystemLog instance from dictionary."""
        return cls(
            log_id=data.get('log_id', generate_id("log")),
            level=LogLevel(data['level']) if isinstance(data['level'], str) else data['level'],
            component=data['component'],
            message=data['message'],
            details=from_json(data.get('details')),
            traceback=data.get('traceback'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp']
        )