"""
Signal Detector Module

Detects trading signals based on technical indicators including
EMA crossovers, trend direction, RSI conditions, and pivot point levels.
"""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

from .technical_indicators import (
    CrossoverType,
    is_near_resistance,
    is_near_support,
    detect_rsi_divergence,
    get_trend_direction,
    calculate_ema_crossover
)
from .indicator_manager import IndicatorValues

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    NONE = "none"
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class Signal:
    """Trading signal with metadata."""
    signal: SignalType
    reason: str
    strength: float  # 0.0 to 1.0
    symbol: str
    price: float
    timestamp: Optional[str] = None
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'signal': self.signal.value,
            'reason': self.reason,
            'strength': self.strength,
            'symbol': self.symbol,
            'price': self.price,
            'timestamp': self.timestamp,
            'details': self.details or {}
        }


class SignalDetector:
    """
    Detects trading signals based on technical indicators.
    
    Combines multiple indicator signals to generate buy/sell signals
    with strength ratings and detailed reasoning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SignalDetector.
        
        Args:
            config: Configuration dictionary with signal detection settings
        """
        self.config = config or {}
        
        # RSI thresholds
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        
        # Resistance/Support threshold
        self.resistance_threshold = self.config.get('resistance_threshold', 0.005)  # 0.5%
        
        # Signal strength thresholds
        self.strong_signal_threshold = self.config.get('strong_signal_threshold', 0.8)
        self.weak_signal_threshold = self.config.get('weak_signal_threshold', 0.3)
        
        # EMA crossover recency (number of candles)
        self.crossover_lookback = self.config.get('crossover_lookback', 3)
        
        logger.info(f"SignalDetector initialized with RSI({self.rsi_oversold}/{self.rsi_overbought}), "
                   f"threshold={self.resistance_threshold}")
    
    def check_entry_signal(
        self,
        symbol: str,
        indicators: IndicatorValues,
        current_price: float,
        price_history: Optional[np.ndarray] = None
    ) -> Signal:
        """
        Check for entry signals based on all indicators.
        
        Args:
            symbol: Trading pair symbol
            indicators: Current indicator values
            current_price: Current market price
            price_history: Optional price history for divergence detection
            
        Returns:
            Signal object with signal type, reason, and strength
        """
        signals = []
        details = {
            'ema_analysis': {},
            'rsi_analysis': {},
            'pivot_analysis': {},
            'trend': indicators.trend
        }
        
        # 1. Check EMA Crossover
        crossover_signal = self._check_crossover_signal(indicators)
        signals.append(crossover_signal)
        details['ema_analysis'] = {
            'crossover': indicators.last_crossover.value,
            'ema_9': indicators.ema_9,
            'ema_21': indicators.ema_21,
            'ema_50': indicators.ema_50,
            'ema_200': indicators.ema_200
        }
        
        # 2. Check Trend Direction
        trend_signal = self._check_trend_signal(indicators)
        signals.append(trend_signal)
        
        # 3. Check RSI Conditions
        rsi_signal = self._check_rsi_signal(indicators, price_history)
        signals.append(rsi_signal)
        details['rsi_analysis'] = {
            'rsi': indicators.rsi,
            'period': indicators.rsi_period,
            'overbought_threshold': self.rsi_overbought,
            'oversold_threshold': self.rsi_oversold
        }
        
        # 4. Check Pivot Point Levels
        pivot_signal = self._check_pivot_signal(indicators, current_price)
        signals.append(pivot_signal)
        details['pivot_analysis'] = {
            'pivot': indicators.pivot,
            'r1': indicators.r1,
            's1': indicators.s1,
            'r2': indicators.r2,
            's2': indicators.s2,
            'near_resistance': False,
            'near_support': False
        }
        
        # Combine signals
        final_signal = self._combine_signals(
            signals, symbol, current_price, details
        )
        
        logger.debug(f"Signal for {symbol} at {current_price}: {final_signal.signal.value} "
                    f"(strength={final_signal.strength:.2f})")
        
        return final_signal
    
    def _check_crossover_signal(self, indicators: IndicatorValues) -> Tuple[SignalType, float, str]:
        """
        Check for EMA crossover signals.
        
        Returns:
            Tuple of (signal_type, strength, reason)
        """
        # Check for recent crossover
        if indicators.last_crossover == CrossoverType.BULLISH:
            # Recent bullish crossover
            if indicators.trend == 'uptrend':
                return (SignalType.BUY, 0.8, "Bullish EMA crossover in uptrend")
            else:
                return (SignalType.BUY, 0.5, "Bullish EMA crossover")
        
        elif indicators.last_crossover == CrossoverType.BEARISH:
            # Recent bearish crossover
            if indicators.trend == 'downtrend':
                return (SignalType.SELL, 0.8, "Bearish EMA crossover in downtrend")
            else:
                return (SignalType.SELL, 0.5, "Bearish EMA crossover")
        
        # No recent crossover - check EMA alignment
        if indicators.ema_9 and indicators.ema_21 and indicators.ema_50:
            if indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
                return (SignalType.BUY, 0.3, "Bullish EMA alignment")
            elif indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
                return (SignalType.SELL, 0.3, "Bearish EMA alignment")
        
        return (SignalType.NONE, 0.0, "No EMA signal")
    
    def _check_trend_signal(self, indicators: IndicatorValues) -> Tuple[SignalType, float, str]:
        """
        Check trend direction signal.
        
        Returns:
            Tuple of (signal_type, strength, reason)
        """
        if indicators.trend == 'uptrend':
            return (SignalType.BUY, 0.4, "Uptrend (EMA50 > EMA200)")
        elif indicators.trend == 'downtrend':
            return (SignalType.SELL, 0.4, "Downtrend (EMA50 < EMA200)")
        else:
            return (SignalType.NONE, 0.0, "Neutral trend")
    
    def _check_rsi_signal(
        self,
        indicators: IndicatorValues,
        price_history: Optional[np.ndarray] = None
    ) -> Tuple[SignalType, float, str]:
        """
        Check RSI conditions for signals.
        
        Returns:
            Tuple of (signal_type, strength, reason)
        """
        if indicators.rsi is None:
            return (SignalType.NONE, 0.0, "No RSI data")
        
        rsi = indicators.rsi
        
        # Check for overbought/oversold
        if rsi <= self.rsi_oversold:
            # Oversold - potential buy signal
            divergence = None
            if price_history is not None:
                rsi_history = np.array([indicators.rsi])  # Simplified
                divergence = detect_rsi_divergence(price_history, rsi_history)
            
            if divergence == 'bullish':
                return (SignalType.BUY, 0.9, f"RSI oversold ({rsi:.1f}) with bullish divergence")
            else:
                return (SignalType.BUY, 0.6, f"RSI oversold ({rsi:.1f})")
        
        elif rsi >= self.rsi_overbought:
            # Overbought - potential sell signal
            divergence = None
            if price_history is not None:
                rsi_history = np.array([indicators.rsi])  # Simplified
                divergence = detect_rsi_divergence(price_history, rsi_history)
            
            if divergence == 'bearish':
                return (SignalType.SELL, 0.9, f"RSI overbought ({rsi:.1f}) with bearish divergence")
            else:
                return (SignalType.SELL, 0.6, f"RSI overbought ({rsi:.1f})")
        
        # RSI in middle range - no strong signal
        return (SignalType.NONE, 0.0, f"RSI neutral ({rsi:.1f})")
    
    def _check_pivot_signal(
        self,
        indicators: IndicatorValues,
        current_price: float
    ) -> Tuple[SignalType, float, str]:
        """
        Check pivot point levels for signals.
        
        Returns:
            Tuple of (signal_type, strength, reason)
        """
        signals = []
        
        # Check resistance levels
        if indicators.r1 is not None and indicators.r2 is not None:
            resistance_levels = np.array([indicators.r1, indicators.r2])
            near_res, nearest_r = is_near_resistance(
                current_price, resistance_levels, self.resistance_threshold
            )
            
            if near_res:
                signals.append((SignalType.SELL, 0.5, f"Price near resistance ({nearest_r:.2f})"))
        
        # Check support levels
        if indicators.s1 is not None and indicators.s2 is not None:
            support_levels = np.array([indicators.s1, indicators.s2])
            near_sup, nearest_s = is_near_support(
                current_price, support_levels, self.resistance_threshold
            )
            
            if near_sup:
                signals.append((SignalType.BUY, 0.5, f"Price near support ({nearest_s:.2f})"))
        
        # Return strongest signal or none
        if signals:
            # Sort by strength and return strongest
            signals.sort(key=lambda x: x[1], reverse=True)
            return signals[0]
        
        return (SignalType.NONE, 0.0, "No pivot point signal")
    
    def _combine_signals(
        self,
        signals: List[Tuple[SignalType, float, str]],
        symbol: str,
        price: float,
        details: Dict[str, Any]
    ) -> Signal:
        """
        Combine multiple indicator signals into final signal.
        
        Args:
            signals: List of (signal_type, strength, reason) tuples
            symbol: Trading pair symbol
            price: Current price
            details: Signal details dictionary
            
        Returns:
            Combined Signal object
        """
        # Separate buy and sell signals
        buy_signals = [s for s in signals if s[0] in (SignalType.BUY, SignalType.STRONG_BUY)]
        sell_signals = [s for s in signals if s[0] in (SignalType.SELL, SignalType.STRONG_SELL)]
        
        # Calculate total strength for each side
        buy_strength = sum(s[1] for s in buy_signals)
        sell_strength = sum(s[1] for s in sell_signals)
        
        # Determine final signal
        if buy_strength > sell_strength and buy_strength >= self.weak_signal_threshold:
            # Buy signal
            if buy_strength >= self.strong_signal_threshold:
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY
            
            reasons = [s[2] for s in buy_signals if s[1] > 0]
            reason = "; ".join(reasons) if reasons else "Buy signal"
            strength = min(buy_strength, 1.0)
            
        elif sell_strength > buy_strength and sell_strength >= self.weak_signal_threshold:
            # Sell signal
            if sell_strength >= self.strong_signal_threshold:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
            
            reasons = [s[2] for s in sell_signals if s[1] > 0]
            reason = "; ".join(reasons) if reasons else "Sell signal"
            strength = min(sell_strength, 1.0)
            
        else:
            # No clear signal
            signal_type = SignalType.NONE
            reason = "No clear signal"
            strength = 0.0
        
        return Signal(
            signal=signal_type,
            reason=reason,
            strength=strength,
            symbol=symbol,
            price=price,
            details=details
        )
    
    def check_exit_signal(
        self,
        symbol: str,
        indicators: IndicatorValues,
        current_price: float,
        entry_price: float,
        position_type: str  # 'long' or 'short'
    ) -> Signal:
        """
        Check for exit signals for an existing position.
        Exits immediately on trend reversal, RSI extremes, or crossover.
        
        Args:
            symbol: Trading pair symbol
            indicators: Current indicator values
            current_price: Current market price
            entry_price: Entry price of the position
            position_type: 'long' or 'short'
            
        Returns:
            Signal object
        """
        details = {
            'position_type': position_type,
            'entry_price': entry_price,
            'current_price': current_price,
            'unrealized_pnl': (current_price - entry_price) / entry_price
        }
        
        if position_type == 'long':
            # Exit immediately on any exit signal for long positions
            if indicators.trend == 'downtrend':
                return Signal(
                    signal=SignalType.SELL,
                    reason="Trend reversal to downtrend",
                    strength=0.8,
                    symbol=symbol,
                    price=current_price,
                    details=details
                )
            
            if indicators.rsi and indicators.rsi >= self.rsi_overbought:
                return Signal(
                    signal=SignalType.SELL,
                    reason=f"RSI overbought ({indicators.rsi:.1f})",
                    strength=0.7,
                    symbol=symbol,
                    price=current_price,
                    details=details
                )
            
            if indicators.last_crossover == CrossoverType.BEARISH:
                return Signal(
                    signal=SignalType.SELL,
                    reason="Bearish EMA crossover",
                    strength=0.6,
                    symbol=symbol,
                    price=current_price,
                    details=details
                )
        
        else:  # short position
            # Exit immediately on any exit signal for short positions
            if indicators.trend == 'uptrend':
                return Signal(
                    signal=SignalType.BUY,
                    reason="Trend reversal to uptrend",
                    strength=0.8,
                    symbol=symbol,
                    price=current_price,
                    details=details
                )
            
            if indicators.rsi and indicators.rsi <= self.rsi_oversold:
                return Signal(
                    signal=SignalType.BUY,
                    reason=f"RSI oversold ({indicators.rsi:.1f})",
                    strength=0.7,
                    symbol=symbol,
                    price=current_price,
                    details=details
                )
            
            if indicators.last_crossover == CrossoverType.BULLISH:
                return Signal(
                    signal=SignalType.BUY,
                    reason="Bullish EMA crossover",
                    strength=0.6,
                    symbol=symbol,
                    price=current_price,
                    details=details
                )
        
        # No exit signal
        return Signal(
            signal=SignalType.NONE,
            reason="No exit signal",
            strength=0.0,
            symbol=symbol,
            price=current_price,
            details=details
        )
    
    def get_stop_loss_price(
        self,
        indicators: IndicatorValues,
        entry_price: float,
        position_type: str,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            indicators: Current indicator values
            entry_price: Entry price
            position_type: 'long' or 'short'
            multiplier: ATR multiplier for stop distance
            
        Returns:
            Stop-loss price
        """
        if indicators.atr is None:
            # Fallback to percentage-based stop
            stop_pct = 0.02  # 2%
            if position_type == 'long':
                return entry_price * (1 - stop_pct)
            else:
                return entry_price * (1 + stop_pct)
        
        atr_distance = indicators.atr * multiplier
        
        if position_type == 'long':
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance
    
    def get_take_profit_price(
        self,
        indicators: IndicatorValues,
        entry_price: float,
        position_type: str,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take-profit price based on risk/reward ratio.
        
        Args:
            indicators: Current indicator values
            entry_price: Entry price
            position_type: 'long' or 'short'
            risk_reward_ratio: Risk to reward ratio
            
        Returns:
            Take-profit price
        """
        stop_loss = self.get_stop_loss_price(indicators, entry_price, position_type)
        
        if position_type == 'long':
            risk = entry_price - stop_loss
            return entry_price + (risk * risk_reward_ratio)
        else:
            risk = stop_loss - entry_price
            return entry_price - (risk * risk_reward_ratio)
