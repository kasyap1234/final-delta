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
from src.indicators.enhanced_signal_detector import (
    EnhancedSignalDetector,
    SignalType,
    check_regime_alpha,
    RegimeAlphaResult,
)
from src.indicators.market_regime import (
    AdaptiveMarketRegimeDetector,
    MarketRegime,
    get_regime_profile,
    V3OperatingMode,
)
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
from src.indicators.signal_quality import (
    SignalQualityScorer,
    calculate_confirmation_score,
    check_multi_confirmation_gate,
    check_volatility_sanity_filter,
    check_activation_gate,
    check_top1_lead_margin,
    rank_symbols_by_edge,
    EdgeScore,
)
from src.risk.position_sizer import PositionSizer
from src.risk.exit_manager import AllWeatherExitManager
from src.risk.regime_risk_router import get_risk_parameters, get_drawdown_size_multiplier
from src.risk.capital_protection import (
    CapitalProtectionManager,
    CapitalProtectionConfig,
    DeploymentState,
)
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
    min_signal_confidence: float = 0.40
    min_adx_for_entry: float = 18.0
    min_ema_spread_for_entry: float = 0.004
    min_regime_confidence: float = 0.35
    max_atr_percent_for_entry: float = 0.04
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

    # Regime detector config (must match live trading bot)
    adx_strong_trend: float = 25.0
    adx_weak_trend: float = 20.0
    bb_squeeze_threshold: float = 0.06
    bb_volatile_threshold: float = 0.10
    min_regime_confidence: float = 0.6


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
            regime_config = {
                "adx_strong_trend": config.adx_strong_trend,
                "adx_weak_trend": config.adx_weak_trend,
                "bb_squeeze_threshold": config.bb_squeeze_threshold,
                "bb_volatile_threshold": config.bb_volatile_threshold,
                "min_confidence": config.min_regime_confidence,
            }
            self.regime_detector = AdaptiveMarketRegimeDetector(regime_config)
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
        
        # Initialize signal quality scorer for directional composite scoring
        self.signal_quality_scorer = SignalQualityScorer()

        # ─── Phase 4: Capital Protection Manager ───────────────────────────────
        # Initialize capital protection manager for deployment state machine
        capital_protection_config = CapitalProtectionConfig()
        self.capital_protection = CapitalProtectionManager(capital_protection_config)

        # ─── Regime-based trade frequency controls ───
        self._candle_count: int = 0
        self._last_trade_candle: Dict[
            str, int
        ] = {}  # symbol -> candle index of last trade
        self._daily_trade_count: Dict[str, int] = {}  # symbol -> trades today
        self._current_day: Optional[str] = (
            None  # track day boundary for resetting counts
        )

        logger.info(
            "BacktestStrategyEngine initialized with regime-based strategy switcher, "
            "signal quality scoring, and capital protection manager"
        )

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
            del self.price_history[symbol][: len(self.price_history[symbol]) - 500]

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

    def _calculate_atr_percent(
        self, symbol: str, indicators: IndicatorValues
    ) -> Optional[float]:
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

    def _volatility_risk_modifier(self, atr_percent: Optional[float]) -> float:
        """Scale risk down as volatility approaches entry cut-off."""
        if atr_percent is None:
            return 1.0
        limit = max(1e-6, float(self.config.max_atr_percent_for_entry))
        if atr_percent >= limit:
            return 0.0
        if atr_percent >= limit * 0.90:
            return 0.45
        if atr_percent >= limit * 0.75:
            return 0.65
        if atr_percent >= limit * 0.60:
            return 0.80
        return 1.0

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorValues,
        regime_metrics: Optional[Any] = None,
        regime_profile: Optional[Dict[str, Any]] = None,
        current_volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> Optional[StrategySignal]:
        """
        Generate trading signal using the actual SignalDetector from live trading.

        Args:
            symbol: Trading symbol
            indicators: IndicatorValues object from IndicatorManager
            regime_metrics: Optional pre-computed regime metrics
            regime_profile: Regime profile from REGIME_PROFILES
            current_volume: Current candle volume for volume confirmation
            avg_volume: Average volume for volume confirmation

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
        if (
            regime_metrics is None
            and self.use_enhanced_strategy
            and self.regime_detector
        ):
            regime_metrics = self.regime_detector.detect_regime(
                prices=price_history
                if price_history is not None
                else np.array([current_price]),
                ema_fast=indicators.ema_9,
                ema_slow=indicators.ema_50,
                adx=indicators.adx,
                atr=indicators.atr,
            )

        self._last_regime_metrics[symbol] = regime_metrics

        signal = self.signal_detector.check_entry_signal(
            symbol=symbol,
            indicators=indicators,
            current_price=current_price,
            price_history=price_history,
            regime_metrics=regime_metrics,
            current_volume=current_volume,
            avg_volume=avg_volume,
            regime_profile=regime_profile,
        )

        # Use regime profile's signal_threshold if available, otherwise use config
        signal_threshold = self.config.min_signal_confidence
        if regime_profile is not None:
            signal_threshold = regime_profile["signal_threshold"]

        # Apply regime confidence as a scaler to signal strength
        adjusted_strength = signal.strength
        if regime_metrics:
            regime_conf_scale = min(1.0, max(0.5, regime_metrics.confidence / 0.60))
            adjusted_strength *= regime_conf_scale

        if adjusted_strength < signal_threshold:
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
            suitability = self.regime_detector.get_regime_suitability(
                regime_metrics.regime, direction
            )
            if suitability < 0.3:
                logger.debug(
                    f"Trade direction {direction} not suitable for regime {regime_metrics.regime.value}"
                )
                return None
            # Scale confidence by suitability
            adjusted_strength *= suitability

        if signal.signal in long_signals:
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
                    f"(strength={adjusted_strength:.2f}, reason={signal.reason})"
                )
                return strategy_signal

        elif signal.signal in short_signals:
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
                    f"(strength={adjusted_strength:.2f}, reason={signal.reason})"
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
        strong_threshold = self.config.strong_signal_threshold
        weak_threshold = self.config.weak_signal_threshold

        if signal_confidence >= strong_threshold:
            return 1.0  # Full position
        elif signal_confidence >= 0.6:
            return 0.75  # 75% position
        elif signal_confidence >= weak_threshold:
            return 0.5  # 50% position
        else:
            return 0.25  # 25% position (minimum)

    def calculate_position_size(
        self,
        symbol: str,
        signal: StrategySignal,
        current_price: float,
        regime_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate position size using regime profile parameters.

        Args:
            symbol: Trading symbol
            signal: Trading signal with confidence (strength)
            current_price: Current market price
            regime_profile: Regime profile from REGIME_PROFILES

        Returns:
            Dictionary with position sizing information
        """
        # Reuse cached indicators (already calculated in process_candle)
        indicators = self.indicator_manager.get_latest(symbol)
        if not indicators or not indicators.ema_200:
            return {"size": 0, "stop_loss": 0, "take_profit": 0}

        position_type = signal.direction.value

        # Use profile's ATR multiplier for stop loss
        atr_multiplier = self.config.stop_loss_atr_multiplier
        stop_pct_floor = None
        stop_pct_cap = None
        if regime_profile is not None:
            atr_multiplier = regime_profile["atr_multiplier"]
            stop_pct_floor = regime_profile.get("stop_pct_floor")
            stop_pct_cap = regime_profile.get("stop_pct_cap")

        # Use regime risk router for stop loss with floor/cap enforcement
        from src.risk.regime_risk_router import calculate_stop_loss_price
        atr = indicators.atr if indicators.atr else current_price * 0.02
        stop_loss = calculate_stop_loss_price(
            entry_price=current_price,
            atr=atr,
            atr_multiplier=atr_multiplier,
            position_type=position_type,
            stop_pct_floor=stop_pct_floor,
            stop_pct_cap=stop_pct_cap,
        )

        # Use profile's R:R ratio for take profit
        rr_ratio = self.config.take_profit_rr_ratio
        if regime_profile is not None:
            rr_ratio = regime_profile["rr_ratio"]

        # rr_ratio == 0 means trailing-only (set TP very far away so it never hits)
        if rr_ratio == 0.0:
            atr = indicators.atr if indicators.atr else current_price * 0.02
            if position_type == "long":
                take_profit = (
                    current_price + atr * 20.0
                )  # Far away — trailing stop does the work
            else:
                take_profit = current_price - atr * 20.0
        else:
            take_profit = self.signal_detector.get_take_profit_price(
                indicators=indicators,
                entry_price=current_price,
                position_type=position_type,
                risk_reward_ratio=rr_ratio,
            )

        # Reuse regime metrics from signal generation
        regime_metrics = self._last_regime_metrics.get(symbol)

        # Apply regime profile's position_size_mod
        position_size_mod = 1.0
        if regime_profile is not None:
            position_size_mod = regime_profile["position_size_mod"]

        adjusted_risk = self.config.max_risk_per_trade_percent * position_size_mod
        atr_percent = self._calculate_atr_percent(symbol, indicators)
        adjusted_risk *= self._volatility_risk_modifier(atr_percent)
        if adjusted_risk <= 0:
            return {"size": 0, "stop_loss": stop_loss, "take_profit": take_profit}

        # Calculate recent performance metrics
        recent_performance = self._calculate_recent_performance()

        # Calculate current drawdown
        current_drawdown = self._calculate_current_drawdown()
        
        # Apply stricter drawdown ladder from regime risk router
        drawdown_multiplier = get_drawdown_size_multiplier(current_drawdown)
        adjusted_risk *= drawdown_multiplier
        if adjusted_risk <= 0:
            return {"size": 0, "stop_loss": stop_loss, "take_profit": take_profit}

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
            "rr_ratio": rr_ratio,
            "signal_strength": signal.confidence,
            "regime_metrics": regime_metrics.to_dict() if regime_metrics else None,
            "use_trailing_tp": regime_profile.get("use_trailing_tp", False)
            if regime_profile
            else False,
            "trailing_activation_atr": regime_profile.get(
                "trailing_activation_atr", 1.5
            )
            if regime_profile
            else 1.5,
            "trailing_distance_atr": regime_profile.get("trailing_distance_atr", 2.0)
            if regime_profile
            else 2.0,
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

    def _check_expected_net_edge(
        self,
        position_size: float,
        entry_price: float,
        stop_distance: float,
        rr_ratio: float,
        regime_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Check if expected net edge is positive and above minimum threshold.

        Implements cost-aware entry rejection from profitability improvement plan:
        - estimated_cost = fees + slippage_buffer + spread_buffer + latency_buffer
        - expected_gross = size * stop_distance * blended_reward_r
        - expected_net_edge = expected_gross - estimated_cost
        - enter only if expected_net_edge > 0 and expected_gross / estimated_cost >= regime_min_ratio

        Args:
            position_size: Position size in units
            entry_price: Entry price
            stop_distance: Stop loss distance in price units
            rr_ratio: Risk-reward ratio
            regime_profile: Regime profile for min_reward_cost_ratio

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        # Cost components (conservative estimates)
        maker_fee = 0.0002  # 0.02%
        taker_fee = 0.0006  # 0.06%
        slippage_buffer = 0.0001  # 0.01%
        spread_buffer = 0.0001  # 0.01%
        latency_buffer = 0.00005  # 0.005%

        # Total estimated cost as percentage of position value
        total_cost_pct = maker_fee + taker_fee + slippage_buffer + spread_buffer + latency_buffer

        # Position value
        position_value = position_size * entry_price

        # Estimated cost in USD
        estimated_cost = position_value * total_cost_pct

        # Expected gross reward (using blended R multiple from TP ladder)
        # Use a conservative estimate: average of TP1 and TP2 if available, else use rr_ratio
        if regime_profile:
            tp1_r = regime_profile.get("tp1_r", rr_ratio)
            tp2_r = regime_profile.get("tp2_r", rr_ratio * 2)
            blended_reward_r = (tp1_r + tp2_r) / 2.0
        else:
            blended_reward_r = rr_ratio

        expected_gross = position_size * stop_distance * blended_reward_r

        # Expected net edge
        expected_net_edge = expected_gross - estimated_cost

        # Get minimum reward/cost ratio from regime profile
        min_reward_cost_ratio = 2.5  # Default
        if regime_profile:
            min_reward_cost_ratio = regime_profile.get("min_reward_cost_ratio", 2.5)

        # Calculate reward/cost ratio
        reward_cost_ratio = expected_gross / estimated_cost if estimated_cost > 0 else 0.0

        # Check conditions
        if expected_net_edge <= 0:
            return False, f"Expected net edge negative: ${expected_net_edge:.2f} (gross ${expected_gross:.2f} - cost ${estimated_cost:.2f})"

        if reward_cost_ratio < min_reward_cost_ratio:
            return False, f"Reward/cost ratio {reward_cost_ratio:.2f} below minimum {min_reward_cost_ratio:.2f}"

        return True, f"Expected net edge positive: ${expected_net_edge:.2f} (ratio {reward_cost_ratio:.2f})"

    def process_candle(
        self,
        symbol: str,
        candle: Dict[str, float],
        timestamp: datetime,
        current_balance: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a candle and generate trading decision.

        Uses centralized REGIME_PROFILES for all regime-specific parameters:
        cooldown, daily limits, signal threshold, volume confirmation,
        position sizing, and R:R ratio.

        Args:
            symbol: Trading symbol
            candle: OHLCV candle data
            timestamp: Candle timestamp
            current_balance: Current account balance

        Returns:
            Trading decision dictionary or None
        """
        self.account_balance = current_balance
        self._candle_count += 1

        # Increment candle count on exit manager for minimum hold time tracking
        self.exit_manager.increment_candle_count()

        # Increment candle count on capital protection manager for cooldown tracking
        self.capital_protection.increment_candle_count()

        # Reset daily trade count on day boundary
        day_key = (
            timestamp.strftime("%Y-%m-%d")
            if hasattr(timestamp, "strftime")
            else str(timestamp)[:10]
        )
        if day_key != self._current_day:
            self._current_day = day_key
            self._daily_trade_count.clear()

        # Update price history
        self.update_price(symbol, candle, timestamp)

        # ─── Phase 4: Update Capital Protection Manager ───────────────────────
        # Update balance for drawdown tracking
        self.capital_protection.update_balance(current_balance)

        # Calculate indicators using IndicatorManager
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None

        # Detect regime first; all subsequent switcher decisions must use this.
        regime_metrics = None
        regime_profile = None
        if self.use_enhanced_strategy and self.regime_detector:
            price_arrays = self.indicator_manager.get_price_arrays(symbol)
            price_history = (
                price_arrays["closes"] if price_arrays else np.array([candle["close"]])
            )
            regime_metrics = self.regime_detector.detect_regime(
                prices=price_history,
                ema_fast=indicators.ema_9,
                ema_slow=indicators.ema_50,
                adx=indicators.adx,
                atr=indicators.atr,
            )
            regime_profile = get_regime_profile(regime_metrics.regime.value)

            should_trade, _ = self.regime_detector.should_trade_in_regime(
                regime_metrics.regime
            )
            if not should_trade:
                logger.debug(
                    f"Trading not allowed in regime: {regime_metrics.regime.value}"
                )
                return None

            # Check position_size_mod: 0 means no trading in this regime
            if regime_profile["position_size_mod"] == 0.0:
                logger.debug(
                    f"No trading in {regime_metrics.regime.value} regime for {symbol}"
                )
                return None

            # Check transition buffer: skip entries if regime just changed
            if self.regime_detector.is_in_transition():
                logger.debug(f"Regime in transition for {symbol}, skipping entry")
                return None

            # Check cooldown: must wait N candles between trades per symbol
            cooldown = regime_profile["cooldown_candles"]
            last_trade = self._last_trade_candle.get(symbol, -cooldown - 1)
            if self._candle_count - last_trade < cooldown:
                return None

            # Check daily trade limit per symbol
            max_daily = regime_profile["max_trades_per_day"]
            daily_count = self._daily_trade_count.get(symbol, 0)
            if daily_count >= max_daily:
                return None

        # ATR percent filter: prevent trading during extreme volatility
        atr_percent = self._calculate_atr_percent(symbol, indicators)
        if (
            atr_percent is not None
            and atr_percent > self.config.max_atr_percent_for_entry
        ):
            logger.debug(
                f"Skipping entry for {symbol}: ATR% {atr_percent:.3f} exceeds "
                f"limit {self.config.max_atr_percent_for_entry:.3f}"
            )
            return None

        # Get volume data for volume confirmation
        current_volume = candle.get("volume")
        avg_volume = None
        price_arrays = self.indicator_manager.get_price_arrays(symbol)
        if price_arrays and "volumes" in price_arrays:
            volumes = price_arrays["volumes"]
            if len(volumes) >= 20:
                avg_volume = float(np.mean(volumes[-20:]))

        # ─── Phase 1: Layer A Activation Gate (expectancy-positive redesign) ───
        if regime_metrics and regime_profile:
            # Calculate directional scores for activation gate
            # Use the signal quality scorer to get long/short scores
            scorer = SignalQualityScorer()
            
            # Get primary signal scores from indicators
            long_primary = 0.5  # Default neutral
            short_primary = 0.5
            long_confirm = 0.5
            short_confirm = 0.5
            long_regime = 0.5
            short_regime = 0.5
            long_structure = 0.5
            short_structure = 0.5
            
            # Calculate regime alignment
            if regime_metrics.regime.value in ("trending_up", "trending_down"):
                long_regime = 0.8 if regime_metrics.regime.value == "trending_up" else 0.2
                short_regime = 0.2 if regime_metrics.regime.value == "trending_up" else 0.8
            elif regime_metrics.regime.value == "ranging":
                long_regime = 0.5
                short_regime = 0.5
            
            # Calculate structure score based on EMA spread
            if indicators.ema_9 and indicators.ema_50:
                ema_spread = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50
                long_structure = 0.5 + (0.5 * min(1.0, ema_spread / 0.02))
                short_structure = 0.5 + (0.5 * min(1.0, ema_spread / 0.02))
            
            # Calculate directional scores
            directional_score = scorer.calculate_directional_score(
                regime=regime_metrics.regime,
                long_primary=long_primary,
                short_primary=short_primary,
                long_confirm=long_confirm,
                short_confirm=short_confirm,
                long_regime=long_regime,
                short_regime=short_regime,
                long_structure=long_structure,
                short_structure=short_structure,
            )
            
            # Calculate volume ratio for liquidity quality
            volume_ratio = None
            if current_volume and avg_volume and avg_volume > 0:
                volume_multiplier = regime_profile.get("volume_multiplier", 1.0)
                volume_ratio = current_volume / (avg_volume * volume_multiplier)
            
            # V2 Fix C1: Detect trend continuation scenario for relaxed entry thresholds
            # Trend continuation: high regime confidence + strong directional alignment
            trend_continuation_relaxed = False
            if regime_metrics.regime.value in ("trending_up", "trending_down"):
                # Check for strong trend continuation conditions
                if regime_metrics.confidence >= 0.75:  # High confidence in trend
                    # Check if EMAs are aligned with trend direction
                    ema_aligned = False
                    if indicators.ema_9 and indicators.ema_50:
                        if regime_metrics.regime.value == "trending_up":
                            ema_aligned = indicators.ema_9 > indicators.ema_50
                        else:
                            ema_aligned = indicators.ema_9 < indicators.ema_50
                    
                    # Check if ADX confirms strong trend
                    adx_confirms = indicators.adx is not None and indicators.adx >= 25.0
                    
                    if ema_aligned and adx_confirms:
                        trend_continuation_relaxed = True
                        logger.debug(
                            f"Trend continuation detected for {symbol}: "
                            f"regime={regime_metrics.regime.value}, confidence={regime_metrics.confidence:.2f}, "
                            f"ADX={indicators.adx:.1f}"
                        )
            
            # Check activation gate
            activation_passed, activation_reason, edge_score = check_activation_gate(
                regime=regime_metrics.regime,
                regime_confidence=regime_metrics.confidence,
                long_score=directional_score.long_score,
                short_score=directional_score.short_score,
                volume_ratio=volume_ratio,
                spread_pct=None,  # Not available in backtest
                min_regime_confidence=0.6,
                trend_continuation_relaxed=trend_continuation_relaxed,
            )
            
            if not activation_passed:
                logger.debug(f"Activation gate blocked for {symbol}: {activation_reason}")
                return None
            
            logger.debug(f"Activation gate passed for {symbol}: {activation_reason}")

            # ─── Phase 4: Deployment State Evaluation ───────────────────────────
            # Evaluate deployment state based on current conditions
            current_drawdown = self.capital_protection.get_current_drawdown()
            active_edge = edge_score.edge_score if edge_score else 0.0
            
            # Simple benchmark trend strength estimate (can be enhanced with actual benchmark data)
            benchmark_trend_strength = 0.0
            if regime_metrics and regime_metrics.regime.value in ("trending_up", "trending_down"):
                benchmark_trend_strength = 0.8  # Strong trend
            elif regime_metrics and regime_metrics.regime.value == "ranging":
                benchmark_trend_strength = 0.3  # Weak trend
            
            # Get regime value for trend quality assessment
            regime_value = regime_metrics.regime.value if regime_metrics else "unknown"
            
            deployment_state = self.capital_protection.evaluate_deployment_state(
                current_drawdown=current_drawdown,
                active_edge=active_edge,
                benchmark_trend_strength=benchmark_trend_strength,
                regime_value=regime_value,
            )
            
            logger.debug(f"Deployment state: {deployment_state.value} (drawdown: {current_drawdown:.2%}, edge: {active_edge:.3f})")

            # ─── Phase 4: Entry Gating Based on Deployment State ───────────────
            # Check if entry should be allowed based on deployment state
            edge_floor = regime_profile.get("edge_floor", 0.08)
            ambiguity_threshold = regime_profile.get("ambiguity_veto_threshold", 0.05)
            
            entry_allowed, entry_reason = self.capital_protection.should_allow_entry(
                regime=regime_metrics.regime.value,
                edge_score=active_edge,
                direction_margin=directional_score.decision_margin,
                edge_floor=edge_floor,
                ambiguity_threshold=ambiguity_threshold,
            )
            
            if not entry_allowed:
                logger.debug(f"Deployment state blocked entry for {symbol}: {entry_reason}")
                return None
            
            logger.debug(f"Deployment state allowed entry for {symbol}: {entry_reason}")

        # ─── Phase 2: Layer B Regime Alpha Confirmation ───────────────────────
        if regime_metrics and regime_profile:
            # Determine breakout condition and directional momentum for volatile regime
            breakout_condition = False
            directional_momentum = False
            if regime_metrics.regime.value == "volatile":
                price_arrays = self.indicator_manager.get_price_arrays(symbol)
                if price_arrays and "highs" in price_arrays and "lows" in price_arrays:
                    highs = price_arrays["highs"]
                    lows = price_arrays["lows"]
                    if len(highs) >= 20:
                        recent_high = max(highs[-20:])
                        recent_low = min(lows[-20:])
                        current_price = candle["close"]
                        # Breakout if price is near recent high/low
                        breakout_condition = current_price >= recent_high * 0.995 or current_price <= recent_low * 1.005
                
                # Directional momentum: check if EMAs are aligned
                if indicators.ema_9 and indicators.ema_50:
                    directional_momentum = indicators.ema_9 != indicators.ema_50
            
            # Check regime-specific alpha confirmation
            alpha_result = check_regime_alpha(
                regime=regime_metrics.regime,
                indicators=indicators,
                direction="long",  # Will be updated after signal generation
                price_history=price_arrays["closes"] if price_arrays else None,
                breakout_condition=breakout_condition,
                directional_momentum=directional_momentum,
            )
            
            if not alpha_result.passed:
                logger.debug(f"Regime alpha confirmation blocked for {symbol}: {alpha_result.reason}")
                return None
            
            logger.debug(f"Regime alpha confirmation passed for {symbol}: {alpha_result.reason}")

        # ─── V2 Fix A1: Strong-Uptrend Directional Bias and Short Suppression ───
        # In high-confidence trending_up regime, suppress short entries unless exceptional edge
        # This addresses severe underperformance in bull years (2023/2024)
        if regime_metrics and regime_profile and regime_metrics.regime.value == "trending_up":
            # Check if we have high confidence in the uptrend
            if regime_metrics.confidence >= 0.75:
                # For short signals, require exceptional edge to override suppression
                # Exceptional edge: edge score at least 2x the normal floor
                exceptional_edge_floor = regime_profile.get("edge_floor", 0.08) * 2.0
                
                # Calculate directional scores to check if short signal is being generated
                scorer = SignalQualityScorer()
                directional_score = scorer.calculate_directional_score(
                    regime=regime_metrics.regime,
                    long_primary=0.5,  # Will be recalculated below
                    short_primary=0.5,
                    long_confirm=0.5,
                    short_confirm=0.5,
                    long_regime=0.8,  # Strong long bias in uptrend
                    short_regime=0.2,  # Weak short bias in uptrend
                    long_structure=0.5,
                    short_structure=0.5,
                )
                
                # If short score is higher than long score, this is a short signal
                if directional_score.short_score > directional_score.long_score:
                    # Check if we have exceptional edge to allow short
                    edge_score = edge_score.edge_score if edge_score else 0.0
                    if edge_score < exceptional_edge_floor:
                        logger.debug(
                            f"Short signal suppressed in strong uptrend {symbol}: "
                            f"edge {edge_score:.3f} < exceptional floor {exceptional_edge_floor:.3f}"
                        )
                        return None
                    else:
                        logger.info(
                            f"Short signal allowed in strong uptrend {symbol} with exceptional edge: "
                            f"edge {edge_score:.3f} >= exceptional floor {exceptional_edge_floor:.3f}"
                        )

        # ─── V3 Pivot: Operating Mode Evaluation and Tactical Short Exception ───
        # Evaluate V3 operating mode for risk-on/risk-off behavior
        v3_operating_mode = None
        if regime_metrics and regime_profile:
            # Get current drawdown and active edge for V3 evaluation
            current_drawdown = self.capital_protection.get_current_drawdown()
            active_edge = edge_score.edge_score if edge_score else 0.0
            
            # Calculate benchmark trend strength (simplified for backtest)
            benchmark_trend_strength = 0.0
            if regime_metrics.regime.value in ("trending_up", "trending_down"):
                benchmark_trend_strength = 0.8  # Strong trend
            elif regime_metrics.regime.value == "ranging":
                benchmark_trend_strength = 0.3  # Weak trend
            elif regime_metrics.regime.value == "volatile":
                benchmark_trend_strength = 0.5  # Moderate trend
            
            # Get regime value and confidence
            regime_value = regime_metrics.regime.value if regime_metrics else "unknown"
            regime_confidence = regime_metrics.confidence if regime_metrics else 0.0
            
            # Evaluate V3 operating mode
            v3_operating_mode = self.capital_protection.evaluate_v3_operating_mode(
                current_drawdown=current_drawdown,
                active_edge=active_edge,
                benchmark_trend_strength=benchmark_trend_strength,
                regime_value=regime_value,
                regime_confidence=regime_confidence,
            )
            
            logger.debug(
                f"V3 operating mode: {v3_operating_mode.value} "
                f"(drawdown: {current_drawdown:.2%}, edge: {active_edge:.3f}, "
                f"regime: {regime_value}, confidence: {regime_confidence:.2f})"
            )
            
            # ─── V3: RISK_OFF_NO_TRADE Mode ───
            # Block all entries in RISK_OFF_NO_TRADE mode
            if v3_operating_mode == V3OperatingMode.RISK_OFF_NO_TRADE:
                logger.debug(f"V3 RISK_OFF_NO_TRADE mode: blocking all entries for {symbol}")
                return None
            
            # ─── V3: PASSIVE_DEFER Mode ───
            # Block all entries in PASSIVE_DEFER mode
            if v3_operating_mode == V3OperatingMode.PASSIVE_DEFER:
                logger.debug(f"V3 PASSIVE_DEFER mode: blocking all entries for {symbol}")
                return None
            
            # ─── V3: RISK_ON_BULL_CAPTURE Mode ───
            # Long-biased participation with bounded short exceptions
            if v3_operating_mode == V3OperatingMode.RISK_ON_BULL_CAPTURE:
                # Check if regime profile allows long bias
                if regime_profile.get("v3_long_bias", False):
                    # For long signals, allow normal processing
                    pass  # Continue to signal generation
                else:
                    # For short signals, check tactical short exception policy
                    if regime_profile.get("v3_tactical_short_allowed", False):
                        # Check if we meet tactical short conditions
                        tactical_short_edge_mult = regime_profile.get("v3_tactical_short_edge_mult", 1.5)
                        tactical_short_edge_floor = regime_profile.get("edge_floor", 0.08) * tactical_short_edge_mult
                        
                        # Check if edge meets tactical short floor
                        if active_edge < tactical_short_edge_floor:
                            logger.debug(
                                f"V3 RISK_ON_BULL_CAPTURE: short signal blocked for {symbol}, "
                                f"edge {active_edge:.3f} < tactical floor {tactical_short_edge_floor:.3f}"
                            )
                            return None
                        else:
                            logger.info(
                                f"V3 RISK_ON_BULL_CAPTURE: tactical short allowed for {symbol}, "
                                f"edge {active_edge:.3f} >= tactical floor {tactical_short_edge_floor:.3f}"
                            )
                    else:
                        # Tactical shorts not allowed in this regime
                        logger.debug(
                            f"V3 RISK_ON_BULL_CAPTURE: short signal blocked for {symbol}, "
                            f"tactical shorts not allowed in regime {regime_value}"
                        )
                        return None
            
            # ─── V3: RISK_ON_SELECTIVE Mode ───
            # Selective participation based on edge quality
            if v3_operating_mode == V3OperatingMode.RISK_ON_SELECTIVE:
                # Apply higher edge threshold for selective mode
                selective_edge_threshold = regime_profile.get("edge_floor", 0.08) * 1.2
                if active_edge < selective_edge_threshold:
                    logger.debug(
                        f"V3 RISK_ON_SELECTIVE: signal blocked for {symbol}, "
                        f"edge {active_edge:.3f} < selective threshold {selective_edge_threshold:.3f}"
                    )
                    return None

        # Generate entry signal using SignalDetector with regime profile
        signal = self.generate_signal(
            symbol,
            indicators,
            regime_metrics=regime_metrics,
            regime_profile=regime_profile,
            current_volume=current_volume,
            avg_volume=avg_volume,
        )
        if not signal:
            return None

        # ─── Multi-confirmation gate (new from profitability improvement plan) ───
        if regime_metrics and regime_profile:
            # Calculate EMA spread for structure check
            ema_spread = None
            if indicators.ema_9 and indicators.ema_50:
                ema_spread = abs(indicators.ema_9 - indicators.ema_50) / indicators.ema_50

            # Calculate volume ratio
            volume_ratio = None
            if current_volume and avg_volume and avg_volume > 0:
                volume_multiplier = regime_profile.get("volume_multiplier", 1.0)
                volume_ratio = current_volume / (avg_volume * volume_multiplier)

            # Determine breakout condition for volatile regime
            breakout_condition = False
            directional_momentum = False
            if regime_metrics.regime.value == "volatile":
                # Simple breakout detection: price near recent high/low
                price_arrays = self.indicator_manager.get_price_arrays(symbol)
                if price_arrays and "highs" in price_arrays and "lows" in price_arrays:
                    highs = price_arrays["highs"]
                    lows = price_arrays["lows"]
                    if len(highs) >= 20:
                        recent_high = max(highs[-20:])
                        recent_low = min(lows[-20:])
                        current_price = candle["close"]
                        # Breakout if price is near recent high (for long) or low (for short)
                        if signal.direction == TradeDirection.LONG:
                            breakout_condition = current_price >= recent_high * 0.995
                        else:
                            breakout_condition = current_price <= recent_low * 1.005

                # Directional momentum: check if EMAs are aligned with signal direction
                if signal.direction == TradeDirection.LONG:
                    directional_momentum = (
                        indicators.ema_9 and indicators.ema_50 and
                        indicators.ema_9 > indicators.ema_50
                    )
                else:
                    directional_momentum = (
                        indicators.ema_9 and indicators.ema_50 and
                        indicators.ema_9 < indicators.ema_50
                    )

            # Check multi-confirmation gate
            mc_passed, mc_reason = check_multi_confirmation_gate(
                regime=regime_metrics.regime,
                adx=indicators.adx,
                ema_spread=ema_spread,
                volume_ratio=volume_ratio,
                rsi=indicators.rsi,
                breakout_condition=breakout_condition,
                directional_momentum=directional_momentum,
            )
            if not mc_passed:
                logger.debug(f"Multi-confirmation gate blocked for {symbol}: {mc_reason}")
                return None

            # ─── Volatility sanity filter (new from profitability improvement plan) ───
            atr_percent = self._calculate_atr_percent(symbol, indicators)
            atr_cap = regime_profile.get("stop_pct_cap", 0.06)  # Use stop cap as ATR cap
            vs_passed, vs_reason = check_volatility_sanity_filter(
                regime=regime_metrics.regime,
                atr_percent=atr_percent,
                atr_cap=atr_cap,
                extreme_cap=0.08,  # 8% absolute maximum
            )
            if not vs_passed:
                logger.debug(f"Volatility sanity filter blocked for {symbol}: {vs_reason}")
                return None

        # Calculate position size with regime profile
        position_info = self.calculate_position_size(
            symbol, signal, candle["close"], regime_profile=regime_profile
        )

        if position_info["size"] <= 0:
            return None

        # ─── Cost-aware expected-edge gate (new from profitability improvement plan) ───
        # Replace simple fee check with comprehensive net-edge calculation
        atr = position_info["atr"]
        rr = position_info.get("rr_ratio", self.config.take_profit_rr_ratio)
        stop_distance = atr * position_info["atr_multiplier"]

        edge_passed, edge_reason = self._check_expected_net_edge(
            position_size=position_info["size"],
            entry_price=candle["close"],
            stop_distance=stop_distance,
            rr_ratio=rr,
            regime_profile=regime_profile,
        )
        if not edge_passed:
            logger.debug(f"Expected net edge gate blocked for {symbol}: {edge_reason}")
            return None

        # Record trade for cooldown and daily limit tracking
        self._last_trade_candle[symbol] = self._candle_count
        self._daily_trade_count[symbol] = self._daily_trade_count.get(symbol, 0) + 1

        return {
            "signal": signal,
            "position_size": position_info["size"],
            "entry_price": candle["close"],
            "stop_loss": position_info["stop_loss"],
            "take_profit": position_info["take_profit"],
            "atr": position_info["atr"],
            "use_trailing_tp": position_info.get("use_trailing_tp", False),
            "trailing_activation_atr": position_info.get(
                "trailing_activation_atr", 1.5
            ),
            "trailing_distance_atr": position_info.get("trailing_distance_atr", 2.0),
            "regime_value": regime_metrics.regime.value if regime_metrics else "unknown",
        }

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
        self, 
        symbol: str, 
        entry_price: float, 
        position_type: str, 
        position_size: float,
        stop_loss_price: Optional[float] = None,
        atr: Optional[float] = None,
        regime_value: str = "unknown",
    ) -> None:
        """
        Register an open position for exit signal tracking.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            position_size: Position size
            stop_loss_price: Stop loss price
            atr: ATR value
            regime_value: Regime value for TP level calculation
        """
        position_id = f"{symbol}_{position_type}"
        self.open_positions[symbol] = {
            "id": position_id,
            "entry_price": entry_price,
            "position_type": position_type,
            "position_size": position_size,
            "entry_time": datetime.now(),
        }
        
        # Register with exit manager for TP level tracking
        self.exit_manager.register_position(
            position_id=position_id,
            symbol=symbol,
            side=position_type,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss_price=stop_loss_price,
            atr=atr,
            regime_value=regime_value,
        )

    def close_position(self, symbol: str) -> None:
        """
        Remove a position from tracking when closed.

        Args:
            symbol: Trading symbol
        """
        if symbol in self.open_positions:
            position_id = self.open_positions[symbol].get("id", f"{symbol}_long")
            self.exit_manager.close_position(position_id)
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

    # ─── Phase 4: Trade Result Recording for Capital Protection ───────────────

    def record_trade_result(
        self,
        symbol: str,
        regime: str,
        pnl_r: float,
        is_win: bool,
        pnl_usd: float = 0.0,
    ) -> None:
        """
        Record a trade result for capital protection tracking.

        This method updates:
        - Trade history for performance calculation
        - Regime performance metrics for kill switch evaluation
        - Daily PnL for daily loss cap
        - Consecutive loss counter

        Args:
            symbol: Trading symbol
            regime: Regime identifier
            pnl_r: PnL in R units
            is_win: True if the trade was a win
            pnl_usd: PnL in USD (optional)
        """
        # Record in trade history
        self.trade_history.append({
            "symbol": symbol,
            "regime": regime,
            "pnl_r": pnl_r,
            "pnl_usd": pnl_usd,
            "is_win": is_win,
            "timestamp": datetime.now(),
        })

        # Update capital protection manager
        self.capital_protection.record_trade_result(regime, pnl_r, is_win)

        # Update consecutive losses
        self.capital_protection.update_consecutive_losses(not is_win)

        # Update daily PnL
        current_day = datetime.now().strftime("%Y-%m-%d")
        self.capital_protection.update_daily_pnl(pnl_usd, current_day)

        # Trigger cooldown if consecutive loss throttle is hit
        if self.capital_protection.is_consecutive_loss_throttle_active():
            self.capital_protection.trigger_consecutive_loss_cooldown()

        logger.debug(
            f"Trade result recorded: {symbol} in {regime} regime, "
            f"PnL: {pnl_r:.2f}R (${pnl_usd:.2f}), Win: {is_win}"
        )

    def get_deployment_state(self) -> DeploymentState:
        """
        Get the current deployment state.

        Returns:
            Current deployment state (ACTIVE/PASSIVE/DEFENSIVE)
        """
        return self.capital_protection.get_state()

    def get_capital_protection_metrics(self) -> Dict[str, Any]:
        """
        Get capital protection metrics for monitoring.

        Returns:
            Dictionary with capital protection metrics
        """
        return {
            "deployment_state": self.capital_protection.get_state().value,
            "current_drawdown": self.capital_protection.get_current_drawdown(),
            "drawdown_size_multiplier": self.capital_protection.get_drawdown_size_multiplier(),
            "daily_pnl": self.capital_protection._daily_pnl,
            "consecutive_losses": self.capital_protection._consecutive_losses,
            "regime_metrics": {
                regime: metrics.to_dict()
                for regime, metrics in self.capital_protection.get_all_regime_metrics().items()
            },
            "state_history": self.capital_protection.get_state_history(),
        }
