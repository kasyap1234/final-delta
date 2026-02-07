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

from src.indicators.signal_detector import SignalDetector, Signal
from src.indicators.enhanced_signal_detector import EnhancedSignalDetector, SignalType
from src.indicators.market_regime import AdaptiveMarketRegimeDetector, MarketRegime
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
    detect_rsi_divergence,
)
from src.indicators.indicator_manager import IndicatorManager, IndicatorValues
from src.risk.position_sizer import PositionSizer
from src.risk.exit_manager import AllWeatherExitManager
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
    max_position_size_percent: float = 20.0
    max_risk_per_trade_percent: float = 2.0
    stop_loss_atr_multiplier: float = 2.0
    take_profit_rr_ratio: float = 2.0

    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    resistance_threshold: float = 0.005
    strong_signal_threshold: float = 0.80
    weak_signal_threshold: float = 0.35
    crossover_lookback: int = 3
    min_signal_confidence: float = 0.30
    min_adx_for_entry: float = 14.0
    min_ema_spread_for_entry: float = 0.003
    min_regime_confidence: float = 0.35
    max_atr_percent_for_entry: float = 0.05
    atr_percent_lookback: int = 3

    # Enhanced strategy settings
    use_enhanced_strategy: bool = True  # Enable new regime-based strategy

    # Kelly criterion settings
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.5  # Half-Kelly

    # Volatility targeting
    use_volatility_targeting: bool = True
    target_volatility: float = 0.15  # 15% annualized

    # Mean reversion settings
    mr_rsi_threshold: float = 20
    mr_bb_threshold: float = 0.02

    # Regime-based weights
    trending_trend_weight: float = 0.9
    trending_mr_weight: float = 0.1
    ranging_trend_weight: float = 0.2
    ranging_mr_weight: float = 0.8


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
            "rsi_period": config.rsi_period,
            "atr_period": config.atr_period,
            "pivot_lookback": config.pivot_lookback,
            "ema_periods": [
                config.ema_short,
                config.ema_medium,
                config.ema_long,
                config.ema_trend,
            ],
        }
        self.indicator_manager = IndicatorManager(indicator_config)

        # Initialize signal detector with config (matching live trading)
        signal_config = {
            "rsi_overbought": config.rsi_overbought,
            "rsi_oversold": config.rsi_oversold,
            "resistance_threshold": config.resistance_threshold,
            "strong_signal_threshold": config.strong_signal_threshold,
            "weak_signal_threshold": config.weak_signal_threshold,
            "crossover_lookback": config.crossover_lookback,
            "mr_rsi_threshold": config.mr_rsi_threshold,
            "mr_bb_threshold": config.mr_bb_threshold,
        }

        # Use enhanced strategy with regime detection if enabled
        self.use_enhanced_strategy = config.use_enhanced_strategy
        if self.use_enhanced_strategy:
            self.signal_detector = EnhancedSignalDetector(signal_config)
            self.regime_detector = AdaptiveMarketRegimeDetector(
                {
                    "adx_strong_trend": 25,
                    "adx_weak_trend": 20,
                    "bb_squeeze_threshold": 0.06,
                    "bb_volatile_threshold": 0.10,
                    "min_confidence": 0.6,
                }
            )
            logger.info("Using EnhancedSignalDetector with MarketRegimeDetector")
        else:
            self.signal_detector = SignalDetector(signal_config)
            self.regime_detector = None
            logger.info("Using standard SignalDetector")

        # Initialize position sizer with enhanced config
        position_sizer_config = {
            "default_risk_percent": config.max_risk_per_trade_percent,
            "default_atr_multiplier": config.stop_loss_atr_multiplier,
            "default_risk_reward_ratio": config.take_profit_rr_ratio,
            "use_kelly": config.use_kelly_sizing,
            "kelly_fraction": config.kelly_fraction,
            "use_volatility_targeting": config.use_volatility_targeting,
            "target_volatility": config.target_volatility,
        }
        self.position_sizer = PositionSizer(position_sizer_config)

        # Initialize price calculator
        self.price_calculator = PriceCalculator()

        # Price history for each symbol (OHLCV format for IndicatorManager)
        self.price_history: Dict[str, List[List[float]]] = {}

        # Active entry signals
        self.active_signals: Dict[str, StrategySignal] = {}

        # Track open positions for exit signal detection
        self.open_positions: Dict[str, Dict[str, Any]] = {}

        # Track performance for Kelly criterion
        self.trade_history: List[Dict[str, Any]] = []

        # Cache regime metrics from signal generation to avoid double-counting
        self._last_regime_metrics: Dict[str, Any] = {}

        # Initialize exit manager for all-weather exits
        self.exit_manager = AllWeatherExitManager()

        logger.info("BacktestStrategyEngine initialized with all-weather strategy")

    def update_price(
        self, symbol: str, candle: Dict[str, float], timestamp: datetime
    ) -> None:
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
            candle["open"],
            candle["high"],
            candle["low"],
            candle["close"],
            candle["volume"],
        ]

        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(ohlcv_candle)

        # Keep only last 500 candles — pop from front to keep same list object
        if len(self.price_history[symbol]) > 500:
            del self.price_history[symbol][:len(self.price_history[symbol]) - 500]

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

    def _calculate_atr_percent(self, symbol: str, indicators: IndicatorValues) -> Optional[float]:
        """Calculate ATR as a percent of price using recent candle data."""
        if indicators.atr is None:
            return None

        price_arrays = self.indicator_manager.get_price_arrays(symbol)
        if not price_arrays:
            return None

        closes = price_arrays.get("closes")
        if closes is not None and len(closes) >= self.config.atr_percent_lookback:
            window = closes[-self.config.atr_percent_lookback :]
            avg_price = float(np.mean(window))
            if avg_price > 0:
                return float(indicators.atr / avg_price)

        reference_price = indicators.ema_50 or indicators.ema_9
        if reference_price and reference_price > 0:
            return float(indicators.atr / reference_price)

        return None

    def generate_signal(
        self, symbol: str, indicators: IndicatorValues
    ) -> Optional[StrategySignal]:
        """
        Generate trading signal using the actual SignalDetector from live trading.

        Args:
            symbol: Trading symbol
            indicators: IndicatorValues object from IndicatorManager

        Returns:
            StrategySignal object or None
        """
        # Use last close price (matching live bot's use of actual market price)
        if symbol in self.price_history and self.price_history[symbol]:
            current_price = self.price_history[symbol][-1][4]  # close price from OHLCV
        else:
            current_price = indicators.ema_9
        if current_price is None:
            return None

        # Get price history for divergence detection
        price_arrays = self.indicator_manager.get_price_arrays(symbol)
        price_history = None
        if price_arrays:
            price_history = price_arrays["closes"]

        # Detect market regime if using enhanced strategy
        regime_metrics = None
        if self.use_enhanced_strategy and self.regime_detector:
            regime_metrics = self.regime_detector.detect_regime(
                prices=price_history
                if price_history is not None
                else np.array([current_price]),
                ema_fast=indicators.ema_9,
                ema_slow=indicators.ema_50,
                adx=indicators.adx,
                atr=indicators.atr,
            )
            # Check if trading is allowed in this regime
            should_trade, _ = self.regime_detector.should_trade_in_regime(
                regime_metrics.regime
            )
            if not should_trade:
                logger.debug(
                    f"Trading not allowed in regime: {regime_metrics.regime.value}"
                )
                return None

        self._last_regime_metrics[symbol] = regime_metrics

        signal = self.signal_detector.check_entry_signal(
            symbol=symbol,
            indicators=indicators,
            current_price=current_price,
            price_history=price_history,
            regime_metrics=regime_metrics,
        )

        # Apply regime confidence as a scaler to signal strength
        adjusted_strength = signal.strength
        if regime_metrics:
            regime_conf_scale = min(1.0, max(0.5, regime_metrics.confidence / 0.60))
            adjusted_strength *= regime_conf_scale

        if adjusted_strength < self.config.min_signal_confidence:
            return None

        long_signals = (
            SignalType.BUY,
            SignalType.STRONG_BUY,
            SignalType.MEAN_REVERSION_LONG,
        )
        short_signals = (
            SignalType.SELL,
            SignalType.STRONG_SELL,
            SignalType.MEAN_REVERSION_SHORT,
        )

        # Determine direction and check regime suitability
        if signal.signal in long_signals:
            direction = "long"
        elif signal.signal in short_signals:
            direction = "short"
        else:
            if symbol in self.active_signals:
                del self.active_signals[symbol]
            return None

        if regime_metrics and self.regime_detector:
            suitability = self.regime_detector.get_regime_suitability(regime_metrics.regime, direction)
            if suitability < 0.3:
                logger.debug(f"Trade direction {direction} not suitable for regime {regime_metrics.regime.value}")
                return None
            # Scale confidence by suitability
            adjusted_strength *= suitability

        if signal.signal in long_signals:
            # Check if we already have an active long signal
            if (
                symbol not in self.active_signals
                or self.active_signals[symbol].direction != TradeDirection.LONG
            ):
                strategy_signal = StrategySignal(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=adjusted_strength,
                    metadata={
                        "signal_type": signal.signal.value,
                        "reason": signal.reason,
                        "details": signal.details,
                        "ema_9": indicators.ema_9,
                        "ema_21": indicators.ema_21,
                        "ema_50": indicators.ema_50,
                        "ema_200": indicators.ema_200,
                        "rsi": indicators.rsi,
                        "atr": indicators.atr,
                        "trend": indicators.trend,
                        "last_crossover": indicators.last_crossover.value
                        if indicators.last_crossover
                        else None,
                    },
                )
                self.active_signals[symbol] = strategy_signal
                logger.info(
                    f"LONG signal generated for {symbol} at {current_price:.2f} "
                    f"(strength={signal.strength:.2f}, reason={signal.reason})"
                )
                return strategy_signal

        elif signal.signal in short_signals:
            # Check if we already have an active short signal
            if (
                symbol not in self.active_signals
                or self.active_signals[symbol].direction != TradeDirection.SHORT
            ):
                strategy_signal = StrategySignal(
                    symbol=symbol,
                    direction=TradeDirection.SHORT,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=adjusted_strength,
                    metadata={
                        "signal_type": signal.signal.value,
                        "reason": signal.reason,
                        "details": signal.details,
                        "ema_9": indicators.ema_9,
                        "ema_21": indicators.ema_21,
                        "ema_50": indicators.ema_50,
                        "ema_200": indicators.ema_200,
                        "rsi": indicators.rsi,
                        "atr": indicators.atr,
                        "trend": indicators.trend,
                        "last_crossover": indicators.last_crossover.value
                        if indicators.last_crossover
                        else None,
                    },
                )
                self.active_signals[symbol] = strategy_signal
                logger.info(
                    f"SHORT signal generated for {symbol} at {current_price:.2f} "
                    f"(strength={signal.strength:.2f}, reason={signal.reason})"
                )
                return strategy_signal

        # Clear signal if conditions no longer met
        if symbol in self.active_signals:
            del self.active_signals[symbol]

        return None

    def check_exit_signal(
        self,
        symbol: str,
        indicators: IndicatorValues,
        entry_price: float,
        position_type: str,
    ) -> Optional[ExitSignal]:
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
        # Use last close price (matching live bot's use of actual market price)
        if symbol in self.price_history and self.price_history[symbol]:
            current_price = self.price_history[symbol][-1][4]  # close price from OHLCV
        else:
            current_price = indicators.ema_9
        if current_price is None:
            return None

        # Use SignalDetector for exit signal detection (same as live trading)
        signal = self.signal_detector.check_exit_signal(
            symbol=symbol,
            indicators=indicators,
            current_price=current_price,
            entry_price=entry_price,
            position_type=position_type,
        )

        # Convert to ExitSignal if we have a valid exit signal
        if signal.signal != SignalType.NONE:
            exit_direction = (
                TradeDirection.SHORT if position_type == "long" else TradeDirection.LONG
            )

            return ExitSignal(
                symbol=symbol,
                direction=exit_direction,
                timestamp=datetime.now(),
                price=current_price,
                reason=signal.reason,
                strength=signal.strength,
            )

        return None

    def calculate_dynamic_atr_multiplier(self, indicators: IndicatorValues) -> float:
        """
        Calculate dynamic ATR multiplier based on market volatility.
        Increases multiplier in high volatility conditions to avoid premature stop-outs.

        Args:
            indicators: IndicatorValues object with ATR and price data

        Returns:
            Adjusted ATR multiplier
        """
        base_multiplier = self.config.stop_loss_atr_multiplier  # 2.0

        if indicators.atr and indicators.ema_50:
            # Calculate ATR as percentage of price
            atr_percent = indicators.atr / indicators.ema_50

            # High volatility regime (choppy market) - widen stops
            if atr_percent > 0.03:  # 3% ATR
                return base_multiplier * 1.5  # 3.0
            # Medium-high volatility
            elif atr_percent > 0.025:  # 2.5% ATR
                return base_multiplier * 1.25  # 2.5
            # Medium volatility
            elif atr_percent > 0.02:  # 2% ATR
                return base_multiplier * 1.1  # 2.2

        return base_multiplier

    def calculate_adaptive_rr_ratio(self, indicators: IndicatorValues) -> float:
        """Reduce profit targets in ranging markets."""
        base_rr = self.config.take_profit_rr_ratio  # 2.0

        # Detect ranging market (EMAs close together)
        if indicators.ema_9 and indicators.ema_21 and indicators.ema_50:
            ema_range = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50

            if ema_range < 0.01:  # Tight EMAs = ranging
                return 1.5  # Lower target
            elif ema_range < 0.02:  # Medium range
                return 1.75

        return base_rr

    def calculate_signal_strength_multiplier(self, signal_confidence: float) -> float:
        """
        Calculate position size multiplier based on signal strength.

        Position sizing based on signal quality:
        - strength >= 0.8: 100% position (full size)
        - strength 0.6-0.8: 75% position
        - strength 0.4-0.6: 50% position
        - strength < 0.4: 25% position

        This preserves bull market gains (strong signals = full size)
        while reducing choppy market losses (weak signals = smaller size).

        Args:
            signal_confidence: Signal strength from 0.0 to 1.0

        Returns:
            Position size multiplier (0.25 to 1.0)
        """
        if signal_confidence >= 0.8:
            return 1.0  # Full position
        elif signal_confidence >= 0.6:
            return 0.75  # 75% position
        elif signal_confidence >= 0.4:
            return 0.5  # 50% position
        else:
            return 0.25  # 25% position (minimum)

    def calculate_position_size(
        self, symbol: str, signal: StrategySignal, current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate position size using all-weather multi-factor formula.

        Args:
            symbol: Trading symbol
            signal: Trading signal with confidence (strength)
            current_price: Current market price

        Returns:
            Dictionary with position sizing information
        """
        # Reuse cached indicators (already calculated in process_candle)
        indicators = self.indicator_manager.get_latest(symbol)
        if not indicators or not indicators.ema_200:
            return {"size": 0, "stop_loss": 0, "take_profit": 0}

        # Use SignalDetector's methods for stop loss and take profit
        position_type = signal.direction.value

        # Use dynamic ATR multiplier based on volatility
        atr_multiplier = self.calculate_dynamic_atr_multiplier(indicators)

        stop_loss = self.signal_detector.get_stop_loss_price(
            indicators=indicators,
            entry_price=current_price,
            position_type=position_type,
            multiplier=atr_multiplier,
        )

        # Use adaptive R:R ratio based on market conditions
        adaptive_rr = self.calculate_adaptive_rr_ratio(indicators)

        take_profit = self.signal_detector.get_take_profit_price(
            indicators=indicators,
            entry_price=current_price,
            position_type=position_type,
            risk_reward_ratio=adaptive_rr,
        )

        # Reuse regime metrics from signal generation (avoid double-counting stateful detector)
        regime_metrics = self._last_regime_metrics.get(symbol)

        # Regime-based risk adjustment
        regime_risk_mult = 1.0
        if regime_metrics:
            regime_val = regime_metrics.regime.value if hasattr(regime_metrics.regime, 'value') else str(regime_metrics.regime)
            risk_multipliers = {
                'trending_up': 1.5,
                'trending_down': 1.5,
                'ranging': 0.8,
                'volatile': 0.5,
                'quiet': 0.7,
                'unknown': 0.5,
            }
            regime_risk_mult = risk_multipliers.get(regime_val, 1.0)
            # Scale by confidence
            regime_risk_mult = 1.0 + (regime_risk_mult - 1.0) * min(1.0, regime_metrics.confidence / 0.7)

        adjusted_risk = self.config.max_risk_per_trade_percent * regime_risk_mult

        # Calculate recent performance metrics
        recent_performance = self._calculate_recent_performance()

        # Calculate current drawdown
        current_drawdown = self._calculate_current_drawdown()

        # Use all-weather position sizing
        position_result = self.position_sizer.calculate_all_weather_position_size(
            account_balance=self.account_balance,
            risk_percent=adjusted_risk,
            entry_price=current_price,
            stop_loss_price=stop_loss,
            symbol=symbol,
            signal_strength=signal.confidence,
            regime_metrics=regime_metrics,
            recent_performance=recent_performance,
            current_drawdown=current_drawdown,
        )

        atr = indicators.atr if indicators.atr else current_price * 0.02

        return {
            "size": position_result.position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_amount": position_result.risk_amount,
            "atr": atr,
            "atr_multiplier": atr_multiplier,
            "signal_strength": signal.confidence,
            "regime_metrics": regime_metrics.to_dict() if regime_metrics else None,
        }

    def _calculate_recent_performance(self) -> Dict[str, Any]:
        """Calculate recent performance metrics for position sizing."""
        if len(self.trade_history) < 10:
            return {"win_rate": 0.5}

        recent_trades = self.trade_history[-20:]
        wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(recent_trades)

        return {"win_rate": win_rate}

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from trade history."""
        if not self.trade_history:
            return 0.0

        # Use initial account balance
        balance = self.account_balance
        peak_balance = balance
        current_balance = balance

        for trade in self.trade_history:
            pnl = trade.get("pnl", 0)
            current_balance += pnl
            peak_balance = max(peak_balance, current_balance)

        if peak_balance > 0:
            drawdown = (peak_balance - current_balance) / peak_balance
            return max(0.0, drawdown)

        return 0.0

    def process_candle(
        self,
        symbol: str,
        candle: Dict[str, float],
        timestamp: datetime,
        current_balance: float,
    ) -> Optional[Dict[str, Any]]:
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

        # Calculate indicators using IndicatorManager
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None

        # Soft regime scaling (replaces hard regime gate)
        regime_multiplier = self._calculate_regime_multiplier(indicators)

        # ATR percent filter: prevent trading during extreme volatility
        atr_percent = self._calculate_atr_percent(symbol, indicators)
        if atr_percent is not None and atr_percent > self.config.max_atr_percent_for_entry:
            logger.debug(
                f"Skipping entry for {symbol}: ATR% {atr_percent:.3f} exceeds "
                f"limit {self.config.max_atr_percent_for_entry:.3f}"
            )
            return None

        # Generate entry signal using SignalDetector
        signal = self.generate_signal(symbol, indicators)
        if not signal:
            return None

        # Calculate position size with regime-adjusted risk
        position_info = self.calculate_position_size(symbol, signal, candle["close"])

        if position_info["size"] <= 0:
            return None

        # Apply regime multiplier to final position size
        adjusted_size = position_info["size"] * regime_multiplier

        return {
            "signal": signal,
            "position_size": adjusted_size,
            "entry_price": candle["close"],
            "stop_loss": position_info["stop_loss"],
            "take_profit": position_info["take_profit"],
            "atr": position_info["atr"],
        }

    def _calculate_regime_multiplier(self, indicators: IndicatorValues) -> float:
        """Calculate a soft regime quality multiplier (0.3 to 1.0) instead of hard gate."""
        multiplier = 1.0

        # ADX contribution: scale from 0.5 at ADX=0 to 1.0 at ADX >= min_adx
        if indicators.adx is not None:
            adx_scale = min(1.0, max(0.5, indicators.adx / max(self.config.min_adx_for_entry, 1.0)))
            multiplier *= adx_scale

        # EMA spread contribution: scale from 0.6 at 0 spread to 1.0 at min_spread
        if indicators.ema_9 and indicators.ema_50 and indicators.ema_50 > 0:
            ema_spread = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50
            spread_scale = min(1.0, max(0.6, ema_spread / max(self.config.min_ema_spread_for_entry, 0.001)))
            multiplier *= spread_scale

        return max(0.3, multiplier)

    def check_position_exit(
        self,
        symbol: str,
        candle: Dict[str, float],
        timestamp: datetime,
        entry_price: float,
        position_type: str,
    ) -> Optional[ExitSignal]:
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
        exit_signal = self.check_exit_signal(
            symbol, indicators, entry_price, position_type
        )

        return exit_signal

    def register_position(
        self, symbol: str, entry_price: float, position_type: str, position_size: float
    ) -> None:
        """
        Register an open position for exit signal tracking.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            position_size: Position size
        """
        self.open_positions[symbol] = {
            "entry_price": entry_price,
            "position_type": position_type,
            "position_size": position_size,
            "entry_time": datetime.now(),
        }

    def close_position(self, symbol: str) -> None:
        """
        Remove a position from tracking when closed.

        Args:
            symbol: Trading symbol
        """
        if symbol in self.open_positions:
            del self.open_positions[symbol]

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
