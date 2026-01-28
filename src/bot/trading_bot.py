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
class TradingBotConfig:
    """Configuration for the trading bot."""
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
        self.bot_config = TradingBotConfig()
        
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
            self.signal_detector = SignalDetector({
                'rsi_overbought': strategy_config.get('rsi_long_threshold', 60) + 10,
                'rsi_oversold': strategy_config.get('rsi_short_threshold', 40) - 10,
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
            await self.stream_manager.subscribe_symbols(self.symbols)
            logger.info(f"Subscribed to {len(self.symbols)} symbols")
            
            # Start order manager
            await self.order_manager.start()
            
            # Start hedge manager if enabled
            if self.hedge_manager:
                await self.hedge_manager.start()
            
            self.running = True
            self._shutdown_event.clear()
            
            # Start trading loop
            self._trading_task = asyncio.create_task(self._trading_loop())
            
            logger.info("Trading bot started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}", exc_info=True)
            return False
    
    async def stop(self) -> bool:
        """
        Stop the trading bot gracefully.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Bot is not running")
            return True
        
        try:
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
            
            # Save state
            if self.state_manager:
                self._update_state_manager()
                self.state_manager.save_state()
            
            # Stop hedge manager
            if self.hedge_manager:
                await self.hedge_manager.stop()
            
            # Stop order manager
            if self.order_manager:
                await self.order_manager.stop()
            
            # Stop stream manager
            if self.stream_manager:
                await self.stream_manager.stop()
            
            # Close exchange
            if self.exchange:
                await self.exchange.close()
            
            # Close database
            if self.db_manager:
                self.db_manager.close()
            
            logger.info("Trading bot stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}", exc_info=True)
            return False
    
    async def _trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")
        
        while self.running and not self._shutdown_event.is_set():
            try:
                cycle_start = datetime.now()
                
                # 1. Update market data
                await self._update_market_data()
                
                # 2. Calculate indicators
                self._calculate_indicators()
                
                # 3. Process signals (entry)
                await self._process_entry_signals()
                
                # 4. Manage open positions (exit, hedges)
                await self._manage_positions()
                
                # 5. Update hedge status
                if self.hedge_manager:
                    await self._handle_hedges()
                
                # 6. Save state periodically
                self._update_state_manager()
                if len(self.open_positions) > 0 or len(self.active_orders) > 0:
                    self.state_manager.save_state()
                
                # Calculate sleep time
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, self.bot_config.check_interval - cycle_duration)
                
                logger.debug(f"Trading cycle completed in {cycle_duration:.2f}s, "
                            f"sleeping for {sleep_time:.2f}s")
                
                # Wait for next cycle or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time
                    )
                except asyncio.TimeoutError:
                    pass
                    
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(self.bot_config.check_interval)
        
        logger.info("Trading loop ended")
    
    async def _update_market_data(self):
        """Update market data from stream manager."""
        try:
            for symbol in self.symbols:
                price = self.stream_manager.get_last_price(symbol)
                if price:
                    self.state_manager.update_last_price(symbol, price)
                    
                    # Update portfolio tracker
                    self.portfolio_tracker.update_price(symbol, price)
                    
                    # Update open positions
                    for pos_id, pos in self.open_positions.items():
                        if pos['symbol'] == symbol:
                            pos['current_price'] = price
                            
                            # Calculate unrealized P&L
                            if pos['side'] == 'long':
                                pos['unrealized_pnl'] = (price - pos['entry_price']) * pos['size']
                            else:
                                pos['unrealized_pnl'] = (pos['entry_price'] - price) * pos['size']
        
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _calculate_indicators(self):
        """Calculate technical indicators for all symbols."""
        try:
            for symbol in self.symbols:
                # Get OHLCV data from cache
                ohlcv = self.data_cache.get_ohlcv(symbol)
                
                if ohlcv and len(ohlcv) > 50:
                    # Calculate indicators
                    indicators = self.indicator_manager.calculate_all(symbol, ohlcv)
                    self.indicators[symbol] = indicators
                    
                    logger.debug(f"Indicators calculated for {symbol}")
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
    
    async def _process_entry_signals(self):
        """Check for and process entry signals."""
        try:
            # Check if we can open new positions
            if len(self.open_positions) >= self.config.trading.max_positions:
                logger.debug("Max positions reached, skipping entry signals")
                return
            
            for symbol in self.symbols:
                # Skip if already have position in this symbol
                has_position = any(
                    p['symbol'] == symbol 
                    for p in self.open_positions.values()
                )
                if has_position:
                    continue
                
                # Get indicators
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
    
    async def _execute_entry(self, signal: Signal, indicators: IndicatorValues):
        """Execute entry order for a signal."""
        try:
            symbol = signal.symbol
            current_price = signal.price
            
            # Determine side
            side = 'long' if signal.signal in (SignalType.BUY, SignalType.STRONG_BUY) else 'short'
            
            # Calculate stop loss
            atr = indicators.atr or (current_price * 0.02)  # Default 2% if no ATR
            stop_loss_result = self.position_sizer.calculate_stop_loss(
                entry_price=current_price,
                atr_value=atr,
                atr_multiplier=self.config.strategy.atr_multiplier,
                position_type=side
            )
            
            # Calculate take profit
            take_profit_result = self.position_sizer.calculate_take_profit(
                entry_price=current_price,
                stop_loss_price=stop_loss_result.stop_loss_price,
                risk_reward_ratio=self.config.risk_management.take_profit_r_ratio,
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
            
            # Check risk limits
            risk_check = self.risk_manager.can_open_position(
                symbol=symbol,
                position_size=position_result.position_size,
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
                amount=position_result.position_size,
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
                    'size': position_result.position_size,
                    'unrealized_pnl': 0.0,
                    'stop_loss': stop_loss_result.stop_loss_price,
                    'take_profit': take_profit_result.take_profit_price,
                    'risk_amount': position_result.risk_amount,
                    'entry_time': datetime.now().isoformat(),
                    'order_id': order_result.order_id
                }
                
                self.open_positions[position_data['id']] = position_data
                self.state_manager.add_position(position_data)
                
                # Add to portfolio tracker
                self.portfolio_tracker.add_position(
                    symbol=symbol,
                    side=side,
                    size=position_result.position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss_result.stop_loss_price,
                    take_profit=take_profit_result.take_profit_price,
                    risk_amount=position_result.risk_amount
                )
                
                # Register with hedge manager
                if self.hedge_manager:
                    self.hedge_manager.register_position(position_data)
                
                # Save to database
                self._save_trade_to_db(position_data, signal)
                
            else:
                logger.error(f"Failed to place entry order for {symbol}: {order_result.error_message}")
        
        except Exception as e:
            logger.error(f"Error executing entry: {e}", exc_info=True)
    
    async def _manage_positions(self):
        """Manage open positions (check exits, update status)."""
        try:
            positions_to_close = []
            
            for pos_id, position in list(self.open_positions.items()):
                symbol = position['symbol']
                current_price = self.state_manager.get_last_price(symbol)
                
                if not current_price:
                    continue
                
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
            
            # Check hedge triggers for open positions
            for pos_id, position in self.open_positions.items():
                trigger_result = self.hedge_manager.check_hedge_trigger(position)
                
                if trigger_result.should_hedge:
                    logger.info(f"Hedge trigger for {position['symbol']}: {trigger_result.message}")
                    
                    # Open hedge
                    hedge_result = await self.hedge_manager.open_hedge(position)
                    
                    if hedge_result and hedge_result.success:
                        logger.info(f"Hedge opened for {position['symbol']}: {hedge_result.hedge_id}")
                    else:
                        logger.error(f"Failed to open hedge for {position['symbol']}")
            
            # Update hedge status
            await self.hedge_manager.update_hedge_status()
        
        except Exception as e:
            logger.error(f"Error handling hedges: {e}")
    
    def _update_state_manager(self):
        """Update state manager with current positions."""
        try:
            # State manager already tracks positions via add/remove/update methods
            # Just update prices
            for symbol in self.symbols:
                price = self.state_manager.get_last_price(symbol)
                if price:
                    for pos_id, position in self.open_positions.items():
                        if position['symbol'] == symbol:
                            self.state_manager.update_position(pos_id, {
                                'current_price': price,
                                'unrealized_pnl': position.get('unrealized_pnl', 0)
                            })
        
        except Exception as e:
            logger.error(f"Error updating state manager: {e}")
    
    def _get_position_risks(self) -> List[Dict[str, Any]]:
        """Get current positions in risk manager format."""
        risks = []
        for position in self.open_positions.values():
            risks.append({
                'symbol': position['symbol'],
                'size': position['size'],
                'entry_price': position['entry_price'],
                'stop_loss_price': position.get('stop_loss'),
                'side': position['side']
            })
        return risks
    
    def _log_signal(self, signal: Signal):
        """Log signal to database."""
        try:
            from ..database import Signal as SignalModel
            
            signal_model = SignalModel(
                symbol=signal.symbol,
                signal_type=signal.signal.value,
                price=signal.price,
                strength=signal.strength,
                reason=signal.reason,
                details=signal.details
            )
            self.db_manager.save_signal(signal_model)
        
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
    
    def _save_trade_to_db(self, position_data: Dict[str, Any], signal: Signal):
        """Save trade entry to database."""
        try:
            trade = Trade(
                id=position_data['id'],
                symbol=position_data['symbol'],
                side=TradeSide.LONG if position_data['side'] == 'long' else TradeSide.SHORT,
                entry_price=position_data['entry_price'],
                quantity=position_data['size'],
                entry_time=datetime.fromisoformat(position_data['entry_time']),
                status=TradeStatus.OPEN,
                stop_loss_price=position_data.get('stop_loss'),
                take_profit_price=position_data.get('take_profit'),
                metadata={
                    'signal_strength': signal.strength,
                    'signal_reason': signal.reason
                }
            )
            self.db_manager.save_trade(trade)
        
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    def _update_trade_in_db(self, position_id: str, exit_price: float, realized_pnl: float, reason: str):
        """Update trade in database on close."""
        try:
            trade = self.db_manager.get_trade(position_id)
            if trade:
                trade.status = TradeStatus.CLOSED
                trade.exit_price = exit_price
                trade.realized_pnl = realized_pnl
                trade.exit_time = datetime.now()
                trade.close_reason = reason
                self.db_manager.update_trade(trade)
        
        except Exception as e:
            logger.error(f"Error updating trade in database: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current bot status.
        
        Returns:
            Dictionary with bot status information
        """
        return {
            'running': self.running,
            'initialized': self.initialized,
            'open_positions': len(self.open_positions),
            'active_orders': len(self.active_orders),
            'portfolio_value': self.portfolio_tracker.get_total_value() if self.portfolio_tracker else 0,
            'unrealized_pnl': sum(p.get('unrealized_pnl', 0) for p in self.open_positions.values()),
            'state': self.state_manager.get_state_summary() if self.state_manager else {}
        }
