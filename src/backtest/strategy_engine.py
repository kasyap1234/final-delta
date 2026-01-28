"""
Strategy Engine for Backtesting.

This module integrates the trading bot's strategy logic with the backtest engine.
It processes indicators, generates signals, and executes trades during backtests.
Uses the actual SignalDetector and IndicatorManager from the live trading bot
for complete parity with live trading.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from src.indicators.signal_detector import SignalDetector, Signal, SignalType
from src.indicators.technical_indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_pivot_points_from_ohlcv,
    calculate_all_emas,
    CrossoverType,
    calculate_ema_crossover,
    get_trend_direction,
    is_near_resistance,
    is_near_support,
    detect_rsi_divergence
)
from src.indicators.indicator_manager import IndicatorManager, IndicatorValues
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


@dataclass
class ExitSignal:
    """Exit signal for closing positions."""
    symbol: str
    direction: TradeDirection  # Direction to exit (opposite of position)
    timestamp: datetime
    price: float
    reason: str
    strength: float


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
    pivot_lookback: int = 10

    # Risk parameters
    max_position_size_percent: float = 5.0
    max_risk_per_trade_percent: float = 2.0
    stop_loss_atr_multiplier: float = 2.0
    take_profit_rr_ratio: float = 2.0

    # Signal thresholds (matching live trading bot)
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    resistance_threshold: float = 0.005  # 0.5%
    strong_signal_threshold: float = 0.8
    weak_signal_threshold: float = 0.3
    crossover_lookback: int = 3
    
    # Adaptive risk controls - prevents overfitting by using normalized volatility
    atr_baseline_lookback: int = 100  # baseline for ATR% median
    sl_mult_min_factor: float = 0.75  # min multiplier for SL in low vol
    sl_mult_max_factor: float = 1.50  # max multiplier for SL in high vol
    
    rr_min_factor: float = 0.60  # min RR multiplier in weak trends
    rr_max_factor: float = 1.10  # max RR multiplier in strong trends
    trend_strength_ref: float = 2.0  # "2 ATRs" separation is decent trend
    
    # Regime filter - skip trading in choppy/ranging markets
    min_trend_strength_atr: float = 1.0  # abs(EMA50-EMA200)/ATR must exceed this
    
    # Cooldown after losses to reduce churn
    cooldown_after_loss_candles: int = 5


class BacktestStrategyEngine:
    """
    Strategy engine that runs trading logic during backtests.

    This engine uses the actual SignalDetector and IndicatorManager from
    the live trading bot to ensure complete parity between backtest and live trading:
    1. Uses IndicatorManager for all technical indicator calculations
    2. Uses SignalDetector for signal generation (entry and exit)
    3. Supports all signal types: crossover, trend, RSI divergence, pivot points
    4. Maintains identical signal strength thresholds as live trading
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

        # Initialize indicator manager with config
        indicator_config = {
            'rsi_period': config.rsi_period,
            'atr_period': config.atr_period,
            'pivot_lookback': config.pivot_lookback,
            'ema_periods': [config.ema_short, config.ema_medium, config.ema_long, config.ema_trend]
        }
        self.indicator_manager = IndicatorManager(indicator_config)

        # Initialize signal detector with config (matching live trading)
        signal_config = {
            'rsi_overbought': config.rsi_overbought,
            'rsi_oversold': config.rsi_oversold,
            'resistance_threshold': config.resistance_threshold,
            'strong_signal_threshold': config.strong_signal_threshold,
            'weak_signal_threshold': config.weak_signal_threshold,
            'crossover_lookback': config.crossover_lookback
        }
        self.signal_detector = SignalDetector(signal_config)

        # Initialize position sizer
        self.position_sizer = PositionSizer()

        # Initialize price calculator
        self.price_calculator = PriceCalculator()

        # Price history for each symbol (OHLCV format for IndicatorManager)
        self.price_history: Dict[str, List[List[float]]] = {}

        # Active entry signals
        self.active_signals: Dict[str, StrategySignal] = {}

        # Track open positions for exit signal detection
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        
        # Cooldown tracking per symbol (candle index when cooldown expires)
        self.cooldown_until_index: Dict[str, int] = {}

        logger.info("BacktestStrategyEngine initialized with live trading parity")

    def update_price(self, symbol: str, candle: Dict[str, float], timestamp: datetime) -> None:
        """
        Update price history for a symbol.

        Args:
            symbol: Trading symbol
            candle: OHLCV candle data
            timestamp: Candle timestamp
        """
        # Convert to OHLCV format for IndicatorManager: [timestamp, open, high, low, close, volume]
        ohlcv_candle = [
            timestamp.timestamp() * 1000,  # timestamp in milliseconds
            candle['open'],
            candle['high'],
            candle['low'],
            candle['close'],
            candle['volume']
        ]

        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(ohlcv_candle)

        # Keep only last 500 candles for efficiency
        if len(self.price_history[symbol]) > 500:
            self.price_history[symbol] = self.price_history[symbol][-500:]

        # Update IndicatorManager
        self.indicator_manager.update_ohlcv(symbol, self.price_history[symbol])

    def calculate_indicators(self, symbol: str) -> Optional[IndicatorValues]:
        """
        Calculate technical indicators for a symbol using IndicatorManager.

        Args:
            symbol: Trading symbol

        Returns:
            IndicatorValues object or None if insufficient data
        """
        if symbol not in self.price_history:
            return None

        # Use IndicatorManager to calculate all indicators
        indicators = self.indicator_manager.calculate_all(symbol)

        if not indicators or not indicators.ema_200:
            logger.debug(f"Insufficient data for {symbol} to calculate indicators")
            return None

        return indicators
    
    def _clamp(self, x: float, lo: float, hi: float) -> float:
        """Clamp value x between lo and hi."""
        return max(lo, min(hi, x))
    
    def _vol_context(self, symbol: str, indicators: IndicatorValues) -> Dict[str, float]:
        """
        Calculate volatility context for adaptive SL/TP.
        
        Returns:
            Dictionary with atr_ratio (current vol vs baseline) and trend_strength
        """
        price_arrays = self.indicator_manager.get_price_arrays(symbol)
        if not price_arrays or indicators.atr is None:
            return {'atr_ratio': 1.0, 'trend_strength': 0.0}
        
        closes = price_arrays['closes']
        highs = price_arrays['highs']
        lows = price_arrays['lows']
        
        # Recompute ATR series for baseline
        from src.indicators.technical_indicators import calculate_atr
        atr_series = calculate_atr(highs, lows, closes, self.config.atr_period)
        atr_pct_series = atr_series / closes
        
        lookback = min(self.config.atr_baseline_lookback, len(closes))
        recent = atr_pct_series[-lookback:]
        recent = recent[~np.isnan(recent)]
        baseline = float(np.median(recent)) if len(recent) else (indicators.atr / closes[-1])
        
        atr_pct_now = indicators.atr / closes[-1]
        atr_ratio = atr_pct_now / baseline if baseline > 0 else 1.0
        
        # Trend strength in ATR units (dimensionless)
        trend_strength = 0.0
        if indicators.ema_50 is not None and indicators.ema_200 is not None and indicators.atr > 0:
            trend_strength = abs(indicators.ema_50 - indicators.ema_200) / indicators.atr
        
        return {'atr_ratio': atr_ratio, 'trend_strength': trend_strength}

    def generate_signal(self, symbol: str, indicators: IndicatorValues) -> Optional[StrategySignal]:
        """
        Generate trading signal using the actual SignalDetector from live trading.

        Args:
            symbol: Trading symbol
            indicators: IndicatorValues object from IndicatorManager

        Returns:
            StrategySignal object or None
        """
        current_price = indicators.ema_9  # Use any available price
        if current_price is None:
            return None

        # Get price history for divergence detection
        price_arrays = self.indicator_manager.get_price_arrays(symbol)
        price_history = None
        if price_arrays:
            price_history = price_arrays['closes']

        # Use SignalDetector for entry signal detection (same as live trading)
        signal = self.signal_detector.check_entry_signal(
            symbol=symbol,
            indicators=indicators,
            current_price=current_price,
            price_history=price_history
        )

        # Convert Signal to StrategySignal if we have a valid entry signal
        if signal.signal in (SignalType.BUY, SignalType.STRONG_BUY):
            # Check if we already have an active long signal
            if symbol not in self.active_signals or self.active_signals[symbol].direction != TradeDirection.LONG:
                strategy_signal = StrategySignal(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=signal.strength,
                    metadata={
                        'signal_type': signal.signal.value,
                        'reason': signal.reason,
                        'details': signal.details,
                        'ema_9': indicators.ema_9,
                        'ema_21': indicators.ema_21,
                        'ema_50': indicators.ema_50,
                        'ema_200': indicators.ema_200,
                        'rsi': indicators.rsi,
                        'atr': indicators.atr,
                        'trend': indicators.trend,
                        'last_crossover': indicators.last_crossover.value if indicators.last_crossover else None
                    }
                )
                self.active_signals[symbol] = strategy_signal
                logger.info(f"LONG signal generated for {symbol} at {current_price:.2f} "
                           f"(strength={signal.strength:.2f}, reason={signal.reason})")
                return strategy_signal

        elif signal.signal in (SignalType.SELL, SignalType.STRONG_SELL):
            # Check if we already have an active short signal
            if symbol not in self.active_signals or self.active_signals[symbol].direction != TradeDirection.SHORT:
                strategy_signal = StrategySignal(
                    symbol=symbol,
                    direction=TradeDirection.SHORT,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=signal.strength,
                    metadata={
                        'signal_type': signal.signal.value,
                        'reason': signal.reason,
                        'details': signal.details,
                        'ema_9': indicators.ema_9,
                        'ema_21': indicators.ema_21,
                        'ema_50': indicators.ema_50,
                        'ema_200': indicators.ema_200,
                        'rsi': indicators.rsi,
                        'atr': indicators.atr,
                        'trend': indicators.trend,
                        'last_crossover': indicators.last_crossover.value if indicators.last_crossover else None
                    }
                )
                self.active_signals[symbol] = strategy_signal
                logger.info(f"SHORT signal generated for {symbol} at {current_price:.2f} "
                           f"(strength={signal.strength:.2f}, reason={signal.reason})")
                return strategy_signal

        # Clear signal if conditions no longer met
        if symbol in self.active_signals:
            del self.active_signals[symbol]

        return None

    def check_exit_signal(self, symbol: str, indicators: IndicatorValues,
                          entry_price: float, position_type: str) -> Optional[ExitSignal]:
        """
        Check for exit signals using the SignalDetector's exit logic.

        Args:
            symbol: Trading symbol
            indicators: IndicatorValues object
            entry_price: Entry price of the position
            position_type: 'long' or 'short'

        Returns:
            ExitSignal object or None
        """
        current_price = indicators.ema_9
        if current_price is None:
            return None

        # Use SignalDetector for exit signal detection (same as live trading)
        signal = self.signal_detector.check_exit_signal(
            symbol=symbol,
            indicators=indicators,
            current_price=current_price,
            entry_price=entry_price,
            position_type=position_type
        )

        # Convert to ExitSignal if we have a valid exit signal
        if signal.signal != SignalType.NONE:
            exit_direction = TradeDirection.SHORT if position_type == 'long' else TradeDirection.LONG

            return ExitSignal(
                symbol=symbol,
                direction=exit_direction,
                timestamp=datetime.now(),
                price=current_price,
                reason=signal.reason,
                strength=signal.strength
            )

        return None

    def calculate_position_size(self, symbol: str, signal: StrategySignal,
                                current_price: float) -> Dict[str, Any]:
        """
        Calculate position size based on risk parameters using adaptive SL/TP.

        Uses volatility context to adapt stop-loss multiplier (wider in high vol)
        and risk-reward ratio (tighter in weak trends) without overfitting.

        Args:
            symbol: Trading symbol
            signal: Trading signal
            current_price: Current market price

        Returns:
            Dictionary with position sizing information
        """
        # Get indicators for stop loss calculation
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return {'size': 0, 'stop_loss': 0, 'take_profit': 0}

        position_type = signal.direction.value
        
        # Get volatility context for adaptive parameters
        ctx = self._vol_context(symbol, indicators)
        
        # Adaptive stop-loss multiplier: wider in high volatility, tighter in low vol
        sl_mult = self.config.stop_loss_atr_multiplier * self._clamp(
            ctx['atr_ratio'],
            self.config.sl_mult_min_factor,
            self.config.sl_mult_max_factor
        )
        
        # Adaptive RR based on trend strength: tighter in weak trends/ranges
        trend_factor = self._clamp(
            ctx['trend_strength'] / self.config.trend_strength_ref,
            self.config.rr_min_factor,
            self.config.rr_max_factor
        )
        rr = self.config.take_profit_rr_ratio * trend_factor

        stop_loss = self.signal_detector.get_stop_loss_price(
            indicators=indicators,
            entry_price=current_price,
            position_type=position_type,
            multiplier=sl_mult
        )

        take_profit = self.signal_detector.get_take_profit_price(
            indicators=indicators,
            entry_price=current_price,
            position_type=position_type,
            risk_reward_ratio=rr
        )
        
        # Optional: Cap TP at R2/S2 pivot levels (wider than R1/S1) in weak trend conditions
        # Only apply if trend_strength is low (ranging market) and the pivot is reasonable
        if ctx['trend_strength'] < self.config.trend_strength_ref:
            min_rr_distance = abs(current_price - stop_loss) * 0.8  # Ensure at least 0.8:1 RR
            if position_type == 'long' and indicators.r2 is not None:
                pivot_cap = indicators.r2 * (1 - self.config.resistance_threshold)
                if pivot_cap > current_price + min_rr_distance:
                    take_profit = min(take_profit, pivot_cap)
            elif position_type == 'short' and indicators.s2 is not None:
                pivot_cap = indicators.s2 * (1 + self.config.resistance_threshold)
                if pivot_cap < current_price - min_rr_distance:
                    take_profit = max(take_profit, pivot_cap)

        # Calculate position size based on risk (uses adaptive SL multiplier)
        atr = indicators.atr if indicators.atr else current_price * 0.02  # Default 2% if no ATR
        stop_distance = atr * sl_mult

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

        return {
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'atr': atr,
            'sl_mult': sl_mult,
            'rr': rr,
            'trend_strength': ctx['trend_strength']
        }

    def process_candle(self, symbol: str, candle: Dict[str, float], timestamp: datetime,
                       current_balance: float) -> Optional[Dict[str, Any]]:
        """
        Process a candle and generate trading decision.
        
        Includes regime filtering (skip choppy markets) and cooldown after losses.

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
        
        # Check cooldown - skip if still in cooldown period after a loss
        current_idx = len(self.price_history.get(symbol, [])) - 1
        if symbol in self.cooldown_until_index and current_idx <= self.cooldown_until_index[symbol]:
            return None

        # Calculate indicators using IndicatorManager
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None
        
        # Regime filter - skip choppy/ranging markets
        ctx = self._vol_context(symbol, indicators)
        if ctx['trend_strength'] < self.config.min_trend_strength_atr:
            return None  # Skip this candle - not enough trend

        # Generate entry signal using SignalDetector
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
            'atr': position_info['atr'],
            'trend_strength': ctx['trend_strength']
        }

    def check_position_exit(self, symbol: str, candle: Dict[str, float],
                           timestamp: datetime, entry_price: float,
                           position_type: str) -> Optional[ExitSignal]:
        """
        Check if an open position should be exited.

        Args:
            symbol: Trading symbol
            candle: OHLCV candle data
            timestamp: Candle timestamp
            entry_price: Entry price of the position
            position_type: 'long' or 'short'

        Returns:
            ExitSignal object or None
        """
        # Update price history
        self.update_price(symbol, candle, timestamp)

        # Calculate indicators
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None

        # Check for exit signal using SignalDetector
        exit_signal = self.check_exit_signal(symbol, indicators, entry_price, position_type)

        return exit_signal

    def register_position(self, symbol: str, entry_price: float,
                         position_type: str, position_size: float) -> None:
        """
        Register an open position for exit signal tracking.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            position_size: Position size
        """
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'position_type': position_type,
            'position_size': position_size,
            'entry_time': datetime.now()
        }

    def close_position(self, symbol: str) -> None:
        """
        Remove a position from tracking when closed.

        Args:
            symbol: Trading symbol
        """
        if symbol in self.open_positions:
            del self.open_positions[symbol]
    
    def record_trade_outcome(self, symbol: str, realized_pnl: float) -> None:
        """
        Record trade outcome for cooldown management.
        
        Triggers cooldown period after losing trades to reduce churn
        during unfavorable market conditions.

        Args:
            symbol: Trading symbol
            realized_pnl: Realized profit/loss (positive or negative)
        """
        current_idx = len(self.price_history.get(symbol, [])) - 1
        if realized_pnl < 0:
            self.cooldown_until_index[symbol] = current_idx + self.config.cooldown_after_loss_candles
            logger.debug(f"Cooldown activated for {symbol} until candle {self.cooldown_until_index[symbol]}")

    def get_indicator_values(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current indicator values for a symbol as a dictionary.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of indicator values or None
        """
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None

        return indicators.to_dict()
