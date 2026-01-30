"""
Main Trading Bot Orchestrator for Delta Exchange India.

This module provides the TradingBot class that coordinates all modules
and implements the main trading loop, signal processing, position management,
and hedge management.
"""

import asyncio
import logging
import signal
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from pathlib import Path

# Import configuration
from ..config import TradingBotConfig, load_config

# Import exchange
from ..exchange import ExchangeClient

# Import data modules
from ..data import DataCache, StreamManager, StreamConfig

# Import indicators
from ..indicators import (
    IndicatorManager, IndicatorValues,
    SignalDetector, Signal, SignalType
)

# Import correlation
from ..correlation import CorrelationCalculator, PriceHistory

# Import risk management
from ..risk import (
    RiskManager, RiskCheckResult,
    PositionSizer, PositionSizeResult,
    PortfolioTracker, Position
)

# Import execution
from ..execution import (
    OrderExecutor, OrderManager,
    PriceCalculator, OrderResult, OrderStatus
)

# Import hedge management
from ..hedge import (
    HedgeManager, HedgeExecutor,
    HedgeManagerConfig, HedgeExecutorConfig,
    HedgeTriggerResult
)

# Import database
from ..database import DatabaseManager, Trade, TradeSide, TradeStatus

# Import utilities
from ..utils import get_logger, LogCategory

# Import state manager
from .state_manager import StateManager, BotState

logger = get_logger(__name__)


@dataclass
class BotRuntimeConfig:
    """Runtime configuration for the trading bot."""
    check_interval: float = 5.0  # Seconds between trading cycles
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_hedging: bool = True
    enable_signal_logging: bool = True


class CircuitBreaker:
    """Circuit breaker for handling repeated failures."""
    
    STATE_CLOSED = 'closed'
    STATE_OPEN = 'open'
    STATE_HALF_OPEN = 'half_open'
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> str:
        return self._state
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self._lock:
            if self._state == self.STATE_OPEN:
                if self._should_attempt_reset():
                    self._state = self.STATE_HALF_OPEN
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        async with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._state = self.STATE_CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed after successful call")
            else:
                self._failure_count = 0
    
    async def _on_failure(self):
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now().timestamp()
            
            if self._failure_count >= self.failure_threshold:
                self._state = self.STATE_OPEN
                logger.warning(f"Circuit breaker OPENED after {self._failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        elapsed = datetime.now().timestamp() - self._last_failure_time
        return elapsed >= self.reset_timeout


class TradingBot:
    """
    Main trading bot orchestrator.
    
    This class coordinates all modules and implements the main trading loop:
    1. Fetch market data
    2. Calculate indicators
    3. Generate signals
    4. Execute trades
    5. Manage hedges
    6. Log everything
    
    Attributes:
        config: TradingBotConfig instance
        running: Whether the bot is currently running
        initialized: Whether the bot has been initialized
    """
    
    def __init__(self, config: TradingBotConfig):
        """
        Initialize the trading bot.
        
        Args:
            config: TradingBotConfig with all settings
        """
        self.config = config
        self.bot_config = BotRuntimeConfig()
        
        # State
        self.running = False
        self.initialized = False
        self._shutdown_event = asyncio.Event()
        self._trading_task: Optional[asyncio.Task] = None
        
        # Components (initialized in initialize())
        self.db_manager: Optional[DatabaseManager] = None
        self.state_manager: Optional[StateManager] = None
        self.exchange: Optional[ExchangeClient] = None
        self.data_cache: Optional[DataCache] = None
        self.stream_manager: Optional[StreamManager] = None
        self.indicator_manager: Optional[IndicatorManager] = None
        self.signal_detector: Optional[SignalDetector] = None
        self.correlation_calc: Optional[CorrelationCalculator] = None
        self.risk_manager: Optional[RiskManager] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.portfolio_tracker: Optional[PortfolioTracker] = None
        self.price_calculator: Optional[PriceCalculator] = None
        self.order_executor: Optional[OrderExecutor] = None
        self.order_manager: Optional[OrderManager] = None
        self.hedge_executor: Optional[HedgeExecutor] = None
        self.hedge_manager: Optional[HedgeManager] = None
        
        # Tracking
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.symbols: List[str] = []
        self.indicators: Dict[str, IndicatorValues] = {}
        self.signals: List[Signal] = []
        
        # Circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Signal handlers
        self._setup_signal_handlers()
        
        logger.info("TradingBot initialized")
    
    def _setup_signal_handlers(self):
        """Setup handlers for shutdown signals."""
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    def _signal_handler(self):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        asyncio.create_task(self.stop())
    
    async def initialize(self) -> bool:
        """
        Initialize all bot components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing trading bot components...")
            
            # 1. Initialize database
            db_path = self.config.database.db_path
            self.db_manager = DatabaseManager(db_path)
            self.db_manager.initialize_database()
            logger.info(f"Database initialized at {db_path}")
            
            # 2. Initialize state manager
            state_file = Path(db_path).parent / "bot_state.json"
            self.state_manager = StateManager(
                self.db_manager,
                str(state_file)
            )
            logger.info("State manager initialized")
            
            # 3. Initialize exchange client
            exchange_config = {
                'api_key': self.config.exchange.api_key,
                'api_secret': self.config.exchange.api_secret,
                'sandbox': self.config.exchange.sandbox,
                'testnet': self.config.exchange.testnet,
            }
            self.exchange = ExchangeClient(exchange_config)
            await self.exchange.initialize()
            logger.info(f"Exchange client initialized ({self.config.exchange.exchange_id})")
            
            # 4. Initialize data cache and stream manager
            self.data_cache = DataCache()
            stream_config = StreamConfig(
                default_timeframe=self.config.trading.timeframe
            )
            self.stream_manager = StreamManager(
                self.data_cache,
                stream_config,
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret
            )
            logger.info("Data components initialized")
            
            # 5. Initialize indicators and signal detector
            strategy_config = self.config.strategy.dict()
            self.indicator_manager = IndicatorManager(strategy_config)
            
            # Use RSI thresholds from config (now aligned with backtest: 70/30)
            self.signal_detector = SignalDetector({
                'rsi_overbought': strategy_config.get('rsi_short_threshold', 30),
                'rsi_oversold': strategy_config.get('rsi_long_threshold', 70),
            })
            logger.info("Indicator components initialized")
            
            # 6. Initialize correlation calculator
            self.correlation_calc = CorrelationCalculator()
            logger.info("Correlation calculator initialized")
            
            # 7. Initialize risk management
            risk_config = {
                'max_total_exposure_percent': 80.0,
                'max_total_risk_percent': self.config.risk_management.max_risk_per_trade_percent * 3,
                'max_positions': self.config.trading.max_positions,
                'daily_loss_limit_percent': 3.0,
                'weekly_loss_limit_percent': 10.0,
            }
            self.risk_manager = RiskManager(risk_config)
            
            position_sizer_config = {
                'default_risk_percent': self.config.risk_management.risk_per_trade_percent,
                'default_atr_multiplier': strategy_config.get('atr_multiplier', 2.0),
                'default_risk_reward_ratio': self.config.risk_management.take_profit_r_ratio,
            }
            self.position_sizer = PositionSizer(position_sizer_config)
            
            self.portfolio_tracker = PortfolioTracker(
                initial_balance=self.config.risk_management.account_balance
            )
            logger.info("Risk management components initialized")
            
            # 8. Initialize execution components
            self.price_calculator = PriceCalculator()
            self.order_executor = OrderExecutor(
                self.exchange,
                self.price_calculator
            )
            self.order_manager = OrderManager(
                self.order_executor,
                self.db_manager
            )
            logger.info("Execution components initialized")
            
            # 9. Initialize hedge management
            if self.bot_config.enable_hedging:
                hedge_exec_config = HedgeExecutorConfig(
                    num_chunks=self.config.hedge.hedge_chunks
                )
                self.hedge_executor = HedgeExecutor(
                    self.order_executor,
                    hedge_exec_config
                )
                
                hedge_mgr_config = HedgeManagerConfig(
                    hedge_trigger_threshold=self.config.hedge.hedge_trigger_percent / 100.0,
                    enable_rehedging=True,
                    max_hedges_per_position=self.config.hedge.max_hedges_per_position
                )
                self.hedge_manager = HedgeManager(
                    self.hedge_executor,
                    self.correlation_calc,
                    hedge_mgr_config
                )
                logger.info("Hedge management components initialized")
            
            # 10. Setup symbols
            self.symbols = self.config.trading.symbols
            
            # 11. Recover previous state
            await self._recover_state()
            
            self.initialized = True
            logger.info("Trading bot initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}", exc_info=True)
            return False
    
    async def _recover_state(self):
        """Recover state from previous session."""
        try:
            recovered_positions = self.state_manager.recover_positions()
            
            for pos_data in recovered_positions:
                self.open_positions[pos_data['id']] = pos_data
                
                # Register with portfolio tracker
                self.portfolio_tracker.add_position(
                    symbol=pos_data['symbol'],
                    side=pos_data['side'],
                    size=pos_data['size'],
                    entry_price=pos_data['entry_price'],
                    stop_loss=pos_data.get('stop_loss'),
                    take_profit=pos_data.get('take_profit'),
                    risk_amount=pos_data.get('risk_amount', 0)
                )
                
                # Register with hedge manager if enabled
                if self.hedge_manager:
                    self.hedge_manager.register_position(pos_data)
            
            if recovered_positions:
                logger.info(f"Recovered {len(recovered_positions)} positions from previous session")
                
        except Exception as e:
            logger.error(f"Error recovering state: {e}")
    
    async def start(self) -> bool:
        """
        Start the trading bot.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.initialized:
            logger.error("Cannot start: bot not initialized")
            return False
        
        if self.running:
            logger.warning("Bot is already running")
            return True
        
        try:
            logger.info("Starting trading bot...")
            
            # Start stream manager
            await self.stream_manager.start()
            
            # Subscribe to symbols
            for symbol in self.symbols:
                await self.stream_manager.subscribe_symbol(symbol)
            
            # Wait a moment for initial data
            await asyncio.sleep(2)
            
            # Start trading loop
            self.running = True
            self._trading_task = asyncio.create_task(self._trading_loop())
            
            logger.info("Trading bot started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}", exc_info=True)
            return False
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.running:
            return
        
        logger.info("Stopping trading bot...")
        self.running = False
        self._shutdown_event.set()
        
        # Cancel trading loop
        if self._trading_task:
            self._trading_task.cancel()
            try:
                await self._trading_task
            except asyncio.CancelledError:
                pass
        
        # Stop stream manager
        if self.stream_manager:
            await self.stream_manager.stop()
        
        # Save state
        if self.state_manager:
            self.state_manager.save_state()
        
        # Close exchange connection
        if self.exchange:
            await self.exchange.close()
        
        logger.info("Trading bot stopped")
    
    async def _trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # Wait for shutdown event with timeout
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.bot_config.check_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Continue trading
                
                # Update indicators
                await self._update_indicators()
                
                # Process entry signals
                await self._process_entry_signals()
                
                # Manage open positions
                await self._manage_positions()
                
                # Handle hedges
                await self._handle_hedges()
                
                # Save state periodically
                if self.state_manager:
                    self.state_manager.save_state()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info("Trading loop ended")
    
    async def _update_indicators(self):
        """Update technical indicators for all symbols."""
        try:
            for symbol in self.symbols:
                # Get OHLCV data from cache
                ohlcv = self.data_cache.get_ohlcv(symbol)
                
                if ohlcv and len(ohlcv) >= 50:  # Need enough data
                    # Calculate indicators
                    indicators = self.indicator_manager.calculate(ohlcv)
                    self.indicators[symbol] = indicators
                    
        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
    
    async def _process_entry_signals(self):
        """Process entry signals for all symbols."""
        try:
            for symbol in self.symbols:
                # Skip if already at max positions
                if len(self.open_positions) >= self.config.trading.max_positions:
                    break
                
                # Skip if already have position for this symbol
                if any(p['symbol'] == symbol for p in self.open_positions.values()):
                    continue
                
                indicators = self.indicators.get(symbol)
                if not indicators:
                    continue
                
                # Get current price
                current_price = self.state_manager.get_last_price(symbol)
                if not current_price:
                    continue
                
                # Check for signal
                signal = self.signal_detector.check_entry_signal(
                    symbol=symbol,
                    indicators=indicators,
                    current_price=current_price
                )
                
                # Log signal if detected
                if signal.signal != SignalType.NONE:
                    logger.info(f"Signal detected: {symbol} - {signal.signal.value} "
                               f"(strength: {signal.strength:.2f})")
                    
                    if self.bot_config.enable_signal_logging:
                        self._log_signal(signal)
                    
                    # Check if we should execute
                    if signal.strength >= 0.6:  # Minimum strength threshold
                        await self._execute_entry(signal, indicators)
        
        except Exception as e:
            logger.error(f"Error processing entry signals: {e}")
    
    def _passes_regime_filter(self, indicators: IndicatorValues) -> bool:
        """
        Filter out low-trend conditions using ADX and EMA spread.
        Matches backtest regime filtering.
        
        Args:
            indicators: IndicatorValues with ADX and EMA data
            
        Returns:
            True if market regime is suitable for entry
        """
        # Check if regime filtering is enabled
        if not self.config.strategy.min_adx_for_entry and not self.config.strategy.min_ema_spread_for_entry:
            return True
        
        adx_ok = True
        if indicators.adx is not None and self.config.strategy.min_adx_for_entry > 0:
            adx_ok = indicators.adx >= self.config.strategy.min_adx_for_entry
        
        ema_spread_ok = True
        if (indicators.ema_fast and indicators.ema_slow and indicators.ema_slow > 0 and 
            self.config.strategy.min_ema_spread_for_entry > 0):
            ema_spread = abs(indicators.ema_fast - indicators.ema_slow) / indicators.ema_slow
            ema_spread_ok = ema_spread >= self.config.strategy.min_ema_spread_for_entry
        
        return adx_ok and ema_spread_ok
    
    def calculate_dynamic_atr_multiplier(self, indicators: IndicatorValues) -> float:
        """
        Calculate dynamic ATR multiplier based on market volatility.
        Increases multiplier in high volatility conditions to avoid premature stop-outs.
        Matches backtest implementation.
        
        Args:
            indicators: IndicatorValues object with ATR and price data
            
        Returns:
            Adjusted ATR multiplier
        """
        # Check if dynamic ATR is enabled
        if not self.config.strategy.enable_dynamic_atr:
            return self.config.strategy.atr_multiplier
        
        base_multiplier = self.config.strategy.atr_multiplier  # 2.0
        
        if indicators.atr and indicators.ema_slow:
            # Calculate ATR as percentage of price
            atr_percent = indicators.atr / indicators.ema_slow
            
            high_vol_threshold = self.config.strategy.high_volatility_atr_threshold
            
            # High volatility regime (choppy market) - widen stops
            if atr_percent > high_vol_threshold:  # 3% ATR
                return base_multiplier * 1.5  # 3.0
            # Medium-high volatility
            elif atr_percent > high_vol_threshold * 0.83:  # 2.5% ATR
                return base_multiplier * 1.25  # 2.5
            # Medium volatility
            elif atr_percent > high_vol_threshold * 0.67:  # 2% ATR
                return base_multiplier * 1.1  # 2.2
        
        return base_multiplier
    
    def calculate_adaptive_rr_ratio(self, indicators: IndicatorValues) -> float:
        """
        Reduce profit targets in ranging markets.
        Matches backtest implementation.
        
        Args:
            indicators: IndicatorValues with EMA data
            
        Returns:
            Adjusted risk:reward ratio
        """
        # Check if adaptive R:R is enabled
        if not self.config.strategy.enable_adaptive_rr:
            return self.config.risk_management.take_profit_r_ratio
        
        base_rr = self.config.risk_management.take_profit_r_ratio  # 2.0
        
        # Detect ranging market (EMAs close together)
        if indicators.ema_fast and indicators.ema_medium and indicators.ema_slow:
            ema_range = abs(indicators.ema_fast - indicators.ema_slow) / indicators.ema_slow
            
            ranging_threshold = self.config.strategy.ranging_ema_threshold
            
            if ema_range < ranging_threshold:  # Tight EMAs = ranging
                return 1.5  # Lower target
            elif ema_range < ranging_threshold * 2:  # Medium range
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
        
        Matches backtest implementation.
        
        Args:
            signal_confidence: Signal strength from 0.0 to 1.0
            
        Returns:
            Position size multiplier (0.25 to 1.0)
        """
        # Check if signal strength sizing is enabled
        if not self.config.strategy.enable_signal_strength_sizing:
            return 1.0
        
        strong_threshold = self.config.strategy.strong_signal_threshold
        weak_threshold = self.config.strategy.weak_signal_threshold
        
        if signal_confidence >= strong_threshold:
            return 1.0  # Full position
        elif signal_confidence >= 0.6:
            return 0.75  # 75% position
        elif signal_confidence >= weak_threshold:
            return 0.5  # 50% position
        else:
            return 0.25  # 25% position (minimum)
    
    async def _execute_entry(self, signal: Signal, indicators: IndicatorValues):
        """Execute entry order for a signal."""
        try:
            symbol = signal.symbol
            current_price = signal.price
            
            # Apply regime filter (matching backtest)
            if not self._passes_regime_filter(indicators):
                logger.info(f"Signal rejected due to regime filter: {symbol}")
                return
            
            # Determine side
            side = 'long' if signal.signal in (SignalType.BUY, SignalType.STRONG_BUY) else 'short'
            
            # Calculate dynamic ATR multiplier
            atr_multiplier = self.calculate_dynamic_atr_multiplier(indicators)
            
            # Calculate stop loss
            atr = indicators.atr or (current_price * 0.02)  # Default 2% if no ATR
            stop_loss_result = self.position_sizer.calculate_stop_loss(
                entry_price=current_price,
                atr_value=atr,
                atr_multiplier=atr_multiplier,
                position_type=side
            )
            
            # Calculate adaptive R:R ratio
            adaptive_rr = self.calculate_adaptive_rr_ratio(indicators)
            
            # Calculate take profit
            take_profit_result = self.position_sizer.calculate_take_profit(
                entry_price=current_price,
                stop_loss_price=stop_loss_result.stop_loss_price,
                risk_reward_ratio=adaptive_rr,
                position_type=side
            )
            
            # Calculate position size
            account_balance = self.portfolio_tracker.get_total_value()
            position_result = self.position_sizer.calculate_position_size(
                account_balance=account_balance,
                risk_percent=self.config.risk_management.risk_per_trade_percent,
                entry_price=current_price,
                stop_loss_price=stop_loss_result.stop_loss_price,
                symbol=symbol
            )
            
            if not position_result.is_valid:
                logger.warning(f"Invalid position size for {symbol}: {position_result.error_message}")
                return
            
            # Apply signal strength-based position sizing
            signal_multiplier = self.calculate_signal_strength_multiplier(signal.strength)
            adjusted_position_size = position_result.position_size * signal_multiplier
            
            # Log position sizing details
            logger.info(f"Position sizing for {symbol}: base={position_result.position_size:.4f}, "
                       f"signal_strength={signal.strength:.2f}, multiplier={signal_multiplier:.2f}, "
                       f"final={adjusted_position_size:.4f}")
            
            # Check risk limits
            risk_check = self.risk_manager.can_open_position(
                symbol=symbol,
                position_size=adjusted_position_size,
                stop_loss_price=stop_loss_result.stop_loss_price,
                entry_price=current_price,
                current_positions=self._get_position_risks()
            )
            
            if not risk_check.can_trade:
                logger.warning(f"Risk check failed for {symbol}: {risk_check.reason}")
                return
            
            # Execute order
            order_result = await self.order_manager.place_entry_order(
                symbol=symbol,
                side='buy' if side == 'long' else 'sell',
                amount=adjusted_position_size,
                price=current_price,
                stop_loss=stop_loss_result.stop_loss_price,
                take_profit=take_profit_result.take_profit_price
            )
            
            if order_result.success:
                logger.info(f"Entry order placed for {symbol}: {order_result.order_id}")
                
                # Track position
                position_data = {
                    'id': order_result.order_id or f"pos_{symbol}_{datetime.now().timestamp()}",
                    'symbol': symbol,
                    'side': side,
                    'entry_price': current_price,
                    'current_price': current_price,
                    'size': adjusted_position_size,
                    'stop_loss': stop_loss_result.stop_loss_price,
                    'take_profit': take_profit_result.take_profit_price,
                    'entry_time': datetime.now(),
                    'risk_amount': account_balance * (self.config.risk_management.risk_per_trade_percent / 100),
                    'signal_strength': signal.strength,
                    'atr_multiplier': atr_multiplier,
                    'rr_ratio': adaptive_rr,
                }
                
                self.open_positions[position_data['id']] = position_data
                self.state_manager.add_position(position_data)
                
                # Track with portfolio tracker
                self.portfolio_tracker.add_position(
                    symbol=symbol,
                    side=side,
                    size=adjusted_position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss_result.stop_loss_price,
                    take_profit=take_profit_result.take_profit_price,
                    risk_amount=position_data['risk_amount']
                )
                
                # Register with hedge manager
                if self.hedge_manager:
                    self.hedge_manager.register_position(position_data)
                
                # Log to database
                self._log_entry_to_db(position_data, signal)
                
            else:
                logger.error(f"Failed to place entry order for {symbol}: {order_result.error_message}")
        
        except Exception as e:
            logger.error(f"Error executing entry for {signal.symbol}: {e}", exc_info=True)
    
    def _get_position_risks(self) -> List[Dict[str, Any]]:
        """Get current position risks for risk manager."""
        risks = []
        for pos_id, position in self.open_positions.items():
            risks.append({
                'symbol': position['symbol'],
                'size': position['size'],
                'entry_price': position['entry_price'],
                'stop_loss': position.get('stop_loss'),
                'side': position['side'],
            })
        return risks
    
    async def _manage_positions(self):
        """Manage open positions (check stops, take profits, exit signals)."""
        try:
            positions_to_close = []
            
            for pos_id, position in list(self.open_positions.items()):
                symbol = position['symbol']
                current_price = self.state_manager.get_last_price(symbol)
                
                if not current_price:
                    continue
                
                # Update current price
                position['current_price'] = current_price
                
                # Check stop loss
                if position['side'] == 'long':
                    if current_price <= position['stop_loss']:
                        logger.info(f"Stop loss hit for {symbol} at {current_price}")
                        positions_to_close.append((pos_id, 'stop_loss'))
                        continue
                    
                    # Check take profit
                    if current_price >= position['take_profit']:
                        logger.info(f"Take profit hit for {symbol} at {current_price}")
                        positions_to_close.append((pos_id, 'take_profit'))
                        continue
                
                else:  # short
                    if current_price >= position['stop_loss']:
                        logger.info(f"Stop loss hit for {symbol} at {current_price}")
                        positions_to_close.append((pos_id, 'stop_loss'))
                        continue
                    
                    # Check take profit
                    if current_price <= position['take_profit']:
                        logger.info(f"Take profit hit for {symbol} at {current_price}")
                        positions_to_close.append((pos_id, 'take_profit'))
                        continue
                
                # Check for exit signals
                indicators = self.indicators.get(symbol)
                if indicators:
                    exit_signal = self.signal_detector.check_exit_signal(
                        position_side=position['side'],
                        indicators=indicators,
                        current_price=current_price
                    )
                    
                    if exit_signal:
                        logger.info(f"Exit signal for {symbol}: {exit_signal.reason}")
                        positions_to_close.append((pos_id, 'signal'))
            
            # Close positions
            for pos_id, reason in positions_to_close:
                await self._close_position(pos_id, reason)
        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    async def _close_position(self, position_id: str, reason: str):
        """Close a position."""
        try:
            position = self.open_positions.get(position_id)
            if not position:
                return
            
            symbol = position['symbol']
            current_price = self.state_manager.get_last_price(symbol)
            
            # Execute close order
            close_side = 'sell' if position['side'] == 'long' else 'buy'
            
            order_result = await self.order_manager.close_position(
                symbol=symbol,
                side=close_side,
                amount=position['size'],
                price=current_price
            )
            
            if order_result.success:
                # Calculate realized P&L
                if position['side'] == 'long':
                    realized_pnl = (current_price - position['entry_price']) * position['size']
                else:
                    realized_pnl = (position['entry_price'] - current_price) * position['size']
                
                logger.info(f"Position closed: {symbol} ({reason}), P&L: {realized_pnl:.2f}")
                
                # Update tracking
                del self.open_positions[position_id]
                self.state_manager.remove_position(position_id)
                self.state_manager.update_daily_metrics(realized_pnl)
                
                # Update portfolio tracker
                self.portfolio_tracker.close_position(
                    symbol=symbol,
                    exit_price=current_price
                )
                
                # Close any associated hedges
                if self.hedge_manager:
                    await self.hedge_manager.close_all_positions(position_id)
                
                # Update database
                self._update_trade_in_db(position_id, current_price, realized_pnl, reason)
                
            else:
                logger.error(f"Failed to close position {position_id}: {order_result.error_message}")
        
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    async def _handle_hedges(self):
        """Check and manage hedge positions."""
        try:
            if not self.hedge_manager:
                return
            
            # Check each open position for hedge triggers
            for pos_id, position in self.open_positions.items():
                # Get current price
                current_price = self.state_manager.get_last_price(position['symbol'])
                if not current_price:
                    continue
                
                # Check if hedge should be triggered
                trigger_result = self.hedge_manager.check_hedge_trigger(
                    position=position,
                    current_price=current_price
                )
                
                if trigger_result.should_hedge:
                    logger.info(f"Hedge triggered for {position['symbol']}: {trigger_result.reason}")
                    
                    # Execute hedge
                    hedge_result = await self.hedge_manager.execute_hedge(
                        position=position,
                        trigger_result=trigger_result
                    )
                    
                    if hedge_result.success:
                        logger.info(f"Hedge executed for {position['symbol']}")
                    else:
                        logger.warning(f"Hedge execution failed: {hedge_result.error_message}")
        
        except Exception as e:
            logger.error(f"Error handling hedges: {e}")
    
    def _log_signal(self, signal: Signal):
        """Log signal to database."""
        try:
            if self.db_manager:
                self.db_manager.log_signal(
                    symbol=signal.symbol,
                    signal_type=signal.signal.value,
                    strength=signal.strength,
                    price=signal.price,
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
    
    def _log_entry_to_db(self, position_data: Dict[str, Any], signal: Signal):
        """Log entry to database."""
        try:
            if self.db_manager:
                trade = Trade(
                    id=position_data['id'],
                    symbol=position_data['symbol'],
                    side=TradeSide.LONG if position_data['side'] == 'long' else TradeSide.SHORT,
                    entry_price=position_data['entry_price'],
                    size=position_data['size'],
                    stop_loss=position_data.get('stop_loss'),
                    take_profit=position_data.get('take_profit'),
                    entry_time=position_data['entry_time'],
                    status=TradeStatus.OPEN,
                    signal_strength=signal.strength,
                )
                self.db_manager.save_trade(trade)
        except Exception as e:
            logger.error(f"Error logging entry: {e}")
    
    def _update_trade_in_db(self, position_id: str, exit_price: float, 
                           realized_pnl: float, reason: str):
        """Update trade in database when closed."""
        try:
            if self.db_manager:
                self.db_manager.close_trade(
                    trade_id=position_id,
                    exit_price=exit_price,
                    realized_pnl=realized_pnl,
                    exit_reason=reason,
                    exit_time=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            'running': self.running,
            'initialized': self.initialized,
            'open_positions': len(self.open_positions),
            'active_orders': len(self.active_orders),
            'portfolio_value': self.portfolio_tracker.get_total_value() if self.portfolio_tracker else 0,
            'unrealized_pnl': self.portfolio_tracker.get_unrealized_pnl() if self.portfolio_tracker else 0,
        }
