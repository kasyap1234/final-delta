"""
Strategy Engine for Backtesting.

This module integrates the trading bot's strategy logic with the backtest engine.
It processes indicators, generates signals, and executes trades during backtests.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.indicators.indicator_manager import IndicatorManager
from src.indicators.signal_detector import SignalDetector, Signal, SignalType
from src.risk.position_sizer import PositionSizer
from src.execution.price_calculator import PriceCalculator


class TradeDirection(Enum):
    """Trade direction for backtest strategy."""
    LONG = "long"
    SHORT = "short"


@dataclass
class StrategySignal:
    """Trading signal for backtest strategy."""
    symbol: str
    direction: TradeDirection
    timestamp: datetime
    price: float
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for the strategy engine."""
    ema_short: int = 9
    ema_medium: int = 21
    ema_long: int = 50
    ema_trend: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    
    # Risk parameters
    max_position_size_percent: float = 5.0
    max_risk_per_trade_percent: float = 2.0
    stop_loss_atr_multiplier: float = 2.0
    take_profit_rr_ratio: float = 2.0
    
    # Signal thresholds
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


class BacktestStrategyEngine:
    """
    Strategy engine that runs trading logic during backtests.
    
    This engine:
    1. Calculates technical indicators for each symbol
    2. Detects trading signals
    3. Calculates position sizes based on risk
    4. Generates orders for the backtest
    """
    
    def __init__(self, config: StrategyConfig, account_balance: float = 10000.0):
        """
        Initialize the strategy engine.
        
        Args:
            config: Strategy configuration
            account_balance: Initial account balance for position sizing
        """
        self.config = config
        self.account_balance = account_balance
        
        # Initialize indicator manager
        self.indicator_manager = IndicatorManager()
        
        # Initialize signal detector
        self.signal_detector = SignalDetector()
        
        # Initialize position sizer
        self.position_sizer = PositionSizer()
        
        # Initialize price calculator
        self.price_calculator = PriceCalculator()
        
        # Price history for each symbol
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Active signals
        self.active_signals: Dict[str, StrategySignal] = {}
        
        logger.info("BacktestStrategyEngine initialized")
    
    def update_price(self, symbol: str, candle: Dict[str, float], timestamp: datetime) -> None:
        """
        Update price history for a symbol.
        
        Args:
            symbol: Trading symbol
            candle: OHLCV candle data
            timestamp: Candle timestamp
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': timestamp,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        })
        
        # Keep only last 500 candles for efficiency
        if len(self.price_history[symbol]) > 500:
            self.price_history[symbol] = self.price_history[symbol][-500:]
    
    def calculate_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Calculate technical indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicator values or None if insufficient data
        """
        if symbol not in self.price_history:
            return None
        
        prices = self.price_history[symbol]
        
        # Need at least 200 candles for EMA200
        if len(prices) < self.config.ema_trend:
            return None
        
        closes = [p['close'] for p in prices]
        highs = [p['high'] for p in prices]
        lows = [p['low'] for p in prices]
        
        # Calculate EMAs
        ema_short = self._calculate_ema(closes, self.config.ema_short)
        ema_medium = self._calculate_ema(closes, self.config.ema_medium)
        ema_long = self._calculate_ema(closes, self.config.ema_long)
        ema_trend = self._calculate_ema(closes, self.config.ema_trend)
        
        # Calculate RSI
        rsi = self._calculate_rsi(closes, self.config.rsi_period)
        
        # Calculate ATR
        atr = self._calculate_atr(highs, lows, closes, self.config.atr_period)
        
        return {
            'ema_short': ema_short,
            'ema_medium': ema_medium,
            'ema_long': ema_long,
            'ema_trend': ema_trend,
            'rsi': rsi,
            'atr': atr,
            'close': closes[-1]
        }
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA for the given period."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # SMA for first value
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI for the given period."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return (highs[-1] - lows[-1]) if highs else 0.0
        
        true_ranges = []
        for i in range(1, len(highs)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
        
        return sum(true_ranges[-period:]) / period
    
    def generate_signal(self, symbol: str, indicators: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Generate trading signal based on indicators.
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary of indicator values
            
        Returns:
            Signal object or None
        """
        close = indicators['close']
        ema_short = indicators['ema_short']
        ema_medium = indicators['ema_medium']
        ema_long = indicators['ema_long']
        ema_trend = indicators['ema_trend']
        rsi = indicators['rsi']
        atr = indicators['atr']
        
        # Trend following strategy with mean reversion
        
        # LONG signal conditions:
        # 1. Price above EMA trend (bullish trend)
        # 2. EMA short crosses above EMA medium (momentum)
        # 3. RSI not overbought
        if close > ema_trend and ema_short > ema_medium and rsi < self.config.rsi_overbought:
            # Check if we already have an active long signal
            if symbol not in self.active_signals or self.active_signals[symbol].direction != TradeDirection.LONG:
                signal = StrategySignal(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    timestamp=datetime.now(),
                    price=close,
                    confidence=min(0.5 + (rsi / 200), 0.9),  # Higher confidence when RSI is higher but not overbought
                    metadata={
                        'ema_short': ema_short,
                        'ema_medium': ema_medium,
                        'rsi': rsi,
                        'atr': atr
                    }
                )
                self.active_signals[symbol] = signal
                logger.info(f"LONG signal generated for {symbol} at {close:.2f}")
                return signal
        
        # SHORT signal conditions:
        # 1. Price below EMA trend (bearish trend)
        # 2. EMA short crosses below EMA medium (momentum)
        # 3. RSI not oversold
        elif close < ema_trend and ema_short < ema_medium and rsi > self.config.rsi_oversold:
            # Check if we already have an active short signal
            if symbol not in self.active_signals or self.active_signals[symbol].direction != TradeDirection.SHORT:
                signal = StrategySignal(
                    symbol=symbol,
                    direction=TradeDirection.SHORT,
                    timestamp=datetime.now(),
                    price=close,
                    confidence=min(0.5 + ((100 - rsi) / 200), 0.9),
                    metadata={
                        'ema_short': ema_short,
                        'ema_medium': ema_medium,
                        'rsi': rsi,
                        'atr': atr
                    }
                )
                self.active_signals[symbol] = signal
                logger.info(f"SHORT signal generated for {symbol} at {close:.2f}")
                return signal
        
        # Clear signal if conditions no longer met
        if symbol in self.active_signals:
            del self.active_signals[symbol]
        
        return None
    
    def calculate_position_size(self, symbol: str, signal: StrategySignal, current_price: float) -> Dict[str, Any]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            Dictionary with position sizing information
        """
        # Get ATR for stop loss calculation
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return {'size': 0, 'stop_loss': 0, 'take_profit': 0}
        
        atr = indicators['atr']
        
        # Calculate stop loss distance
        stop_distance = atr * self.config.stop_loss_atr_multiplier
        
        # Calculate position size based on risk
        risk_amount = self.account_balance * (self.config.max_risk_per_trade_percent / 100)
        
        # Position size = Risk Amount / Stop Distance
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0
        
        # Limit by max position size
        max_position_value = self.account_balance * (self.config.max_position_size_percent / 100)
        max_position_size = max_position_value / current_price
        
        position_size = min(position_size, max_position_size)
        
        # Calculate stop loss and take profit prices
        if signal.direction == TradeDirection.LONG:
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * self.config.take_profit_rr_ratio)
        else:  # SHORT
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * self.config.take_profit_rr_ratio)
        
        return {
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'atr': atr
        }
    
    def process_candle(self, symbol: str, candle: Dict[str, float], timestamp: datetime, 
                       current_balance: float) -> Optional[Dict[str, Any]]:
        """
        Process a candle and generate trading decision.
        
        Args:
            symbol: Trading symbol
            candle: OHLCV candle data
            timestamp: Candle timestamp
            current_balance: Current account balance
            
        Returns:
            Trading decision dictionary or None
        """
        self.account_balance = current_balance
        
        # Update price history
        self.update_price(symbol, candle, timestamp)
        
        # Calculate indicators
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None
        
        # Generate signal
        signal = self.generate_signal(symbol, indicators)
        if not signal:
            return None
        
        # Calculate position size
        position_info = self.calculate_position_size(symbol, signal, candle['close'])
        
        if position_info['size'] <= 0:
            return None
        
        return {
            'signal': signal,
            'position_size': position_info['size'],
            'entry_price': candle['close'],
            'stop_loss': position_info['stop_loss'],
            'take_profit': position_info['take_profit'],
            'atr': position_info['atr']
        }
