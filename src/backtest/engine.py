"""
Backtest engine module for backtesting.

This module provides the main backtesting engine that orchestrates
the entire backtesting process.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from src.backtest.config import BacktestConfig
from src.backtest.time_controller import TimeController
from src.backtest.account_state import AccountState
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.mock.order_simulator import BacktestOrderSimulator, OHLCV as BacktestOHLCV
from src.backtest.mock.data_cache import BacktestDataCache
from src.backtest.mock.stream_manager import BacktestStreamManager
from src.backtest.mock.exchange_client import BacktestExchangeClient
from src.backtest.strategy_engine import BacktestStrategyEngine, StrategyConfig, TradeDirection
from src.backtest.risk import RiskManager, PortfolioTracker, PositionSizer
from src.backtest.state import StateManager
from src.data.data_cache import OHLCV

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_fees: float
    final_equity: float
    initial_balance: float
    start_date: datetime
    end_date: datetime


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the backtest.
    
    Responsibilities:
    - Load historical data for 2025
    - Initialize all components (mocked and real)
    - Control time progression through historical data
    - Execute trading cycles at each candle
    - Track all state and metrics
    - Generate final reports
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        
        # Core components
        self.time_controller: Optional[TimeController] = None
        self.account_state: Optional[AccountState] = None
        self.data_loader: Optional[HistoricalDataLoader] = None
        
        # Mock components
        self.order_simulator: Optional[BacktestOrderSimulator] = None
        self.data_cache: Optional[BacktestDataCache] = None
        self.stream_manager: Optional[BacktestStreamManager] = None
        self.exchange_client: Optional[BacktestExchangeClient] = None
        
        # Risk management components
        self.risk_manager: Optional[RiskManager] = None
        self.portfolio_tracker: Optional[PortfolioTracker] = None
        self.position_sizer: Optional[PositionSizer] = None
        
        # State management
        self.state_manager: Optional[StateManager] = None
        
        # Historical data
        self.historical_data: Dict[str, List[OHLCV]] = {}
        
        # State tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Strategy engine
        self.strategy_engine: Optional[BacktestStrategyEngine] = None
        
        # Track active positions (symbol -> position_id)
        self.active_positions: Dict[str, str] = {}
        
        logger.info("BacktestEngine initialized")
    
    async def run(self) -> Dict[str, Any]:
        """
        Execute the complete backtest.
        
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        # Phase 1: Initialize components
        await self._initialize()
        
        # Phase 2: Load historical data
        await self._load_data()
        
        # Phase 3: Run backtest loop
        await self._run_backtest_loop()
        
        # Phase 4: Generate results
        results = self._generate_results()
        
        logger.info("Backtest completed successfully")
        
        return results
    
    async def _initialize(self) -> None:
        """Initialize all backtest components."""
        logger.info("Initializing backtest components...")
        
        # Initialize time controller
        self.time_controller = TimeController(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            timeframe=self.config.timeframe
        )
        
        # Initialize account state
        self.account_state = AccountState(
            initial_balance=self.config.initial_balance,
            currency=self.config.initial_currency
        )

        # Initialize fee calculator and attach to account state
        fee_calculator = self.config.create_fee_calculator()
        self.account_state.set_fee_calculator(fee_calculator)
        
        # Initialize order simulator
        from src.backtest.mock.order_simulator import SimulatorConfig
        simulator_config = SimulatorConfig()
        self.order_simulator = BacktestOrderSimulator(
            config=simulator_config,
            account_state=self.account_state
        )
        
        # Initialize data loader
        self.data_loader = HistoricalDataLoader(self.config)
        
        # Initialize risk management components
        risk_config = self._get_risk_config()
        self.risk_manager = RiskManager(config=risk_config)
        self.risk_manager.set_account_balance(self.config.initial_balance)
        
        self.portfolio_tracker = PortfolioTracker(
            initial_balance=self.config.initial_balance
        )
        
        self.position_sizer = PositionSizer(config=risk_config)
        
        # Initialize state manager
        self.state_manager = StateManager(
            bot_id=f"backtest_{int(datetime.now().timestamp())}",
            initial_balance=self.config.initial_balance
        )
        
        # Set correlation groups if available in config
        if hasattr(self.config, 'correlation_groups') and self.config.correlation_groups:
            self.risk_manager.set_correlation_groups(self.config.correlation_groups)
        
        # Initialize strategy engine
        strategy_config = StrategyConfig()
        if self.config.trading_bot_config and 'indicators' in self.config.trading_bot_config:
            indicators = self.config.trading_bot_config['indicators']
            strategy_config.ema_short = indicators.get('ema_short', 9)
            strategy_config.ema_medium = indicators.get('ema_medium', 21)
            strategy_config.ema_long = indicators.get('ema_long', 50)
            strategy_config.ema_trend = indicators.get('ema_trend', 200)
            strategy_config.rsi_period = indicators.get('rsi_period', 14)
            strategy_config.atr_period = indicators.get('atr_period', 14)
        
        if self.config.trading_bot_config and 'risk' in self.config.trading_bot_config:
            risk = self.config.trading_bot_config['risk']
            strategy_config.max_position_size_percent = risk.get('max_position_size_percent', 5.0)
            strategy_config.max_risk_per_trade_percent = risk.get('max_risk_per_trade_percent', 2.0)
            strategy_config.stop_loss_atr_multiplier = risk.get('stop_loss_atr_multiplier', 2.0)
            strategy_config.take_profit_rr_ratio = risk.get('take_profit_rr_ratio', 2.0)
        
        self.strategy_engine = BacktestStrategyEngine(
            config=strategy_config,
            account_balance=self.config.initial_balance
        )
        
        logger.info("Backtest components initialized")
    
    def _get_risk_config(self) -> Dict[str, Any]:
        """Extract risk configuration from backtest config."""
        risk_config = {
            'max_total_exposure_percent': 80.0,
            'max_total_risk_percent': 5.0,
            'max_positions': 10,
            'daily_loss_limit_percent': 3.0,
            'weekly_loss_limit_percent': 10.0,
            'max_correlated_exposure': 15.0,
            'correlation_threshold': 0.7,
            'default_risk_percent': 1.0,
            'default_atr_multiplier': 2.0,
            'default_risk_reward_ratio': 2.0,
            'min_position_size': 0.001,
            'max_position_size': 100.0,
            'trading_fee_percent': self.config.fee_rate * 100 if hasattr(self.config, 'fee_rate') else 0.1
        }
        
        # Override with config if available
        if self.config.trading_bot_config and 'risk' in self.config.trading_bot_config:
            risk = self.config.trading_bot_config['risk']
            risk_config['max_total_risk_percent'] = risk.get('max_risk_per_trade_percent', 5.0) * 2.5
            risk_config['max_positions'] = risk.get('max_positions', 10)
            risk_config['daily_loss_limit_percent'] = risk.get('daily_loss_limit_percent', 3.0)
            risk_config['weekly_loss_limit_percent'] = risk.get('weekly_loss_limit_percent', 10.0)
            risk_config['default_risk_percent'] = risk.get('risk_per_trade_percent', 1.0)
            risk_config['default_atr_multiplier'] = risk.get('stop_loss_atr_multiplier', 2.0)
            risk_config['default_risk_reward_ratio'] = risk.get('take_profit_rr_ratio', 2.0)
        
        return risk_config
    
    async def _load_data(self) -> None:
        """Load and validate historical data."""
        logger.info("Loading historical data...")
        
        # Load data
        self.historical_data = self.data_loader.load_data()
        
        # Validate data
        self.data_loader.validate_data(self.historical_data)
        
        # Filter by date range
        self.historical_data = self.data_loader.filter_by_date_range(
            self.historical_data
        )
        
        # Initialize data cache
        self.data_cache = BacktestDataCache(self.historical_data)
        
        # Initialize stream manager
        self.stream_manager = BacktestStreamManager(
            historical_data=self.historical_data,
            data_cache=self.data_cache
        )
        
        # Initialize exchange client
        self.exchange_client = BacktestExchangeClient(
            historical_data=self.historical_data,
            order_simulator=self.order_simulator,
            account_state=self.account_state
        )
        
        # Subscribe to symbols
        await self.stream_manager.subscribe_symbols(self.config.symbols)
        
        logger.info(
            f"Loaded data for {len(self.historical_data)} symbols: "
            f"{list(self.historical_data.keys())}"
        )
    
    async def _run_backtest_loop(self) -> None:
        """Run the main backtest loop."""
        logger.info("Starting backtest loop...")
        
        # Start stream manager
        await self.stream_manager.start()
        
        # Get all unique timestamps across all symbols
        all_timestamps = self._get_all_timestamps()
        all_timestamps.sort()
        
        logger.info(f"Processing {len(all_timestamps)} candles...")
        
        # Process each candle
        for i, timestamp in enumerate(all_timestamps):
            # Update time controller
            self.time_controller.advance_to_time(timestamp)
            current_time = self.time_controller.get_current_time()
            
            # Update time for all time-aware components
            self.exchange_client.set_current_time(current_time)
            self.risk_manager.set_current_time(current_time)
            self.portfolio_tracker.set_current_time(current_time)
            self.state_manager.set_current_time(current_time)
            
            # Update stream manager (pushes data to cache)
            await self.stream_manager.update_time(current_time)
            
            # Process orders for this candle
            await self._process_candle(timestamp)
            
            # Update account state with current prices
            await self._update_account_state()
            
            # Update portfolio tracker with current prices
            await self._update_portfolio_tracker()
            
            # Record equity point
            self._record_equity_point()
            
            # Log progress
            if (i + 1) % 1000 == 0:
                progress = self.time_controller.get_progress()
                equity = self.portfolio_tracker.get_equity()
                logger.info(
                    f"Progress: {progress:.1%} ({i+1}/{len(all_timestamps)} candles) - "
                    f"Equity: ${equity:.2f}"
                )
        
        # Stop stream manager
        await self.stream_manager.stop()
        
        logger.info("Backtest loop completed")
    
    async def _process_candle(self, timestamp: datetime) -> None:
        """
        Process a single candle.
        
        This executes the trading strategy logic for each symbol.
        
        Args:
            timestamp: Candle timestamp
        """
        # Process orders for each symbol
        for symbol in self.config.symbols:
            # Get candle for this timestamp
            candle = self.data_cache.get_candle_at_time(
                symbol=symbol,
                timeframe=self.config.timeframe,
                timestamp=timestamp
            )
            
            if candle:
                # Convert to BacktestOHLCV format
                backtest_candle = BacktestOHLCV(
                    symbol=candle.symbol,
                    timestamp=candle.timestamp,
                    open=float(candle.open),
                    high=float(candle.high),
                    low=float(candle.low),
                    close=float(candle.close),
                    volume=float(candle.volume)
                )
                
                # Process orders (check for fills on existing orders)
                fills = self.order_simulator.process_orders(backtest_candle)
                
                # Record fills
                for fill in fills:
                    self.trade_history.append(fill)
                
                # Check for stop loss / take profit on active positions
                await self._check_position_exits(symbol, backtest_candle)
                
                # Run strategy logic to generate new signals
                await self._execute_strategy(symbol, candle, timestamp)
    
    async def _update_account_state(self) -> None:
        """Update account state with current prices."""
        # Get current prices for all symbols
        current_prices = {}
        for symbol in self.config.symbols:
            price = self.data_cache.get_latest_price(symbol)
            if price is not None:
                current_prices[symbol] = price
        
        # Update unrealized P&L
        self.account_state.update_unrealized_pnl(current_prices)
        
        # Update state manager with prices
        for symbol, price in current_prices.items():
            self.state_manager.update_last_price(symbol, price)
    
    async def _update_portfolio_tracker(self) -> None:
        """Update portfolio tracker with current prices."""
        current_prices = {}
        for symbol in self.config.symbols:
            price = self.data_cache.get_latest_price(symbol)
            if price is not None:
                current_prices[symbol] = price
        
        # Update all position prices
        self.portfolio_tracker.update_all_positions(current_prices)
        
        # Update risk manager with position P&L
        for position in self.portfolio_tracker.get_all_positions():
            self.risk_manager.update_position_pnl(
                position.position_id,
                position.unrealized_pnl
            )
    
    async def _check_position_exits(self, symbol: str, candle: BacktestOHLCV) -> None:
        """
        Check if any active positions should be exited (stop loss or take profit).
        
        Args:
            symbol: Trading symbol
            candle: Current candle data
        """
        # Get position from portfolio tracker
        position = self.portfolio_tracker.get_position_by_symbol(symbol)
        if position is None:
            return
        
        stop_loss = position.stop_loss
        take_profit = position.take_profit
        side = position.side
        
        exit_triggered = False
        exit_price = None
        exit_reason = None
        
        # Check stop loss
        if side == 'long':
            if candle.low <= stop_loss:
                exit_triggered = True
                exit_price = max(candle.open, stop_loss)  # Assume fill at stop or better
                exit_reason = 'stop_loss'
            elif candle.high >= take_profit:
                exit_triggered = True
                exit_price = min(candle.open, take_profit)  # Assume fill at take profit or better
                exit_reason = 'take_profit'
        else:  # short
            if candle.high >= stop_loss:
                exit_triggered = True
                exit_price = min(candle.open, stop_loss)
                exit_reason = 'stop_loss'
            elif candle.low <= take_profit:
                exit_triggered = True
                exit_price = max(candle.open, take_profit)
                exit_reason = 'take_profit'
        
        if exit_triggered and exit_price:
            # Close position in portfolio tracker
            trade_record = self.portfolio_tracker.close_position(
                position_id=position.position_id,
                exit_price=exit_price,
                exit_fees=self._calculate_exit_fees(position.size, exit_price)
            )
            
            if trade_record:
                # Record trade in risk manager
                self.risk_manager.record_trade_result(
                    symbol=symbol,
                    realized_pnl=trade_record.realized_pnl,
                    risk_amount=position.risk_amount
                )
                
                # Remove from state manager
                self.state_manager.close_position(
                    position_id=position.position_id,
                    exit_price=exit_price,
                    realized_pnl=trade_record.realized_pnl
                )
                
                # Remove from active positions tracking
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                
                # Update account state
                self.account_state.close_position(symbol, exit_price)
                
                # Update risk manager balance
                self.risk_manager.set_account_balance(self.portfolio_tracker.get_account_balance())
                
                # Record the trade
                trade = {
                    'timestamp': candle.timestamp.isoformat(),
                    'symbol': symbol,
                    'side': 'sell' if side == 'long' else 'buy',
                    'size': position.size,
                    'price': exit_price,
                    'pnl': trade_record.realized_pnl,
                    'reason': exit_reason,
                    'entry_price': position.entry_price
                }
                self.trade_history.append(trade)
                
                logger.info(
                    f"Position closed for {symbol}: {exit_reason} at {exit_price:.2f}, "
                    f"P&L: ${trade_record.realized_pnl:.2f}"
                )
    
    def _calculate_exit_fees(self, size: float, price: float) -> float:
        """Calculate exit fees for a trade (limit/maker)."""
        fee_calculator = self.account_state.get_fee_calculator() if self.account_state else None
        if fee_calculator:
            from src.backtest.fees import OrderType

            fee_result = fee_calculator.calculate_trade_fee(
                symbol="backtest_exit",
                amount=size,
                price=price,
                order_type=OrderType.LIMIT,
                metadata={'is_maker': True}
            )
            return fee_result['fee_paid']

        position_value = size * price
        fee_rate = self.config.fee_rate if hasattr(self.config, 'fee_rate') else 0.001
        return position_value * fee_rate
    
    async def _execute_strategy(self, symbol: str, candle: Any, timestamp: datetime) -> None:
        """
        Execute trading strategy for a symbol with risk checks.
        
        Args:
            symbol: Trading symbol
            candle: Candle data
            timestamp: Current timestamp
        """
        # Skip if we already have an active position for this symbol
        if self.portfolio_tracker.has_position(symbol):
            return
        
        # Get current balance
        balance = self.account_state.get_balance()
        current_equity = self.portfolio_tracker.get_equity()
        
        # Prepare candle data for strategy
        candle_data = {
            'open': float(candle.open),
            'high': float(candle.high),
            'low': float(candle.low),
            'close': float(candle.close),
            'volume': float(candle.volume)
        }
        
        # Run strategy
        decision = self.strategy_engine.process_candle(
            symbol=symbol,
            candle=candle_data,
            timestamp=timestamp,
            current_balance=current_equity
        )
        
        if not decision:
            return
        
        # Execute the trade
        signal = decision['signal']
        position_size = decision['position_size']
        entry_price = decision['entry_price']
        stop_loss = decision['stop_loss']
        take_profit = decision['take_profit']
        atr_value = decision.get('atr', 0)
        
        # Determine side
        side = 'long' if signal.direction == TradeDirection.LONG else 'short'
        
        # Perform pre-trade risk checks
        current_positions = self.portfolio_tracker.get_positions()
        risk_check = self.risk_manager.can_open_position(
            symbol=symbol,
            position_size=position_size,
            stop_loss_price=stop_loss,
            entry_price=entry_price,
            current_positions=current_positions,
            account_balance=current_equity
        )
        
        if not risk_check.can_trade:
            logger.warning(
                f"Risk check failed for {symbol}: {risk_check.reason}"
            )
            return
        
        # Check if we have enough balance
        position_value = position_size * entry_price
        if position_value > balance.free * 0.95:  # Leave some buffer
            logger.warning(
                f"Insufficient balance for {symbol} trade. "
                f"Needed: ${position_value:.2f}, Free: ${balance.free:.2f}"
            )
            return
        
        # Calculate risk amount for the position
        stop_loss_distance = abs(entry_price - stop_loss)
        risk_amount = position_size * stop_loss_distance
        
        # Calculate entry fees (limit/maker)
        entry_fees = 0.0
        fee_calculator = self.account_state.get_fee_calculator() if self.account_state else None
        if fee_calculator:
            from src.backtest.fees import OrderType

            fee_result = fee_calculator.calculate_trade_fee(
                symbol=symbol,
                amount=position_size,
                price=entry_price,
                order_type=OrderType.LIMIT,
                metadata={'is_maker': True}
            )
            entry_fees = fee_result['fee_paid']
        else:
            entry_fees = position_value * (self.config.fee_rate if hasattr(self.config, 'fee_rate') else 0.001)
        
        # Create position in portfolio tracker
        position = self.portfolio_tracker.add_position(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            entry_time=timestamp,
            fees=entry_fees
        )
        
        # Track position in risk manager
        self.risk_manager.update_position_risk(
            position_id=position.position_id,
            risk_amount=risk_amount,
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_percent=(risk_amount / current_equity * 100) if current_equity > 0 else 0
        )
        
        # Create position in state manager
        self.state_manager.create_position(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'risk_amount': risk_amount,
                'atr_at_entry': atr_value,
                'signal_strength': getattr(signal, 'strength', 'normal')
            }
        )
        
        # Open position in account state
        success = self.account_state.open_position(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            timestamp=timestamp,
            order_type='limit',
            is_maker=True
        )
        
        if success:
            # Track active position
            self.active_positions[symbol] = position.position_id
            
            # Update risk manager balance
            self.risk_manager.set_account_balance(self.portfolio_tracker.get_account_balance())
            
            logger.info(
                f"Position opened for {symbol}: {side} {position_size:.4f} @ {entry_price:.2f}, "
                f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Risk: ${risk_amount:.2f}"
            )
    
    def _record_equity_point(self) -> None:
        """Record an equity point for the current time."""
        equity = self.portfolio_tracker.get_equity()
        balance = self.portfolio_tracker.get_account_balance()
        unrealized_pnl = self.portfolio_tracker.calculate_unrealized_pnl()
        
        point = {
            'timestamp': self.time_controller.get_current_time().isoformat(),
            'equity': equity,
            'balance': balance,
            'free_balance': balance - self.portfolio_tracker.get_total_exposure(),
            'used_balance': self.portfolio_tracker.get_total_exposure(),
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.portfolio_tracker.get_daily_pnl(),
            'num_positions': len(self.portfolio_tracker.get_all_positions()),
            'total_risk': self.portfolio_tracker.get_total_risk()
        }
        
        self.equity_curve.append(point)
        
        # Take portfolio snapshot periodically
        if len(self.equity_curve) % 100 == 0:
            self.portfolio_tracker.take_snapshot()
    
    def _get_all_timestamps(self) -> List[datetime]:
        """
        Get all unique timestamps across all symbols.
        
        Returns:
            Sorted list of timestamps
        """
        timestamps = set()
        
        for symbol, candles in self.historical_data.items():
            for candle in candles:
                timestamps.add(candle.timestamp)
        
        return list(timestamps)
    
    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate backtest results.
        
        Returns:
            Dictionary with backtest results
        """
        final_equity = self.portfolio_tracker.get_equity()
        initial_balance = self.config.initial_balance
        total_return = (final_equity - initial_balance) / initial_balance
        
        # Get performance summary from portfolio tracker
        performance_summary = self.portfolio_tracker.get_performance_summary()
        
        # Get risk summary
        risk_summary = self.risk_manager.get_risk_summary(final_equity)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        
        # Get final risk report
        risk_report = self.portfolio_tracker.generate_risk_report()
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
            'max_drawdown': performance.get('max_drawdown', 0.0),
            'win_rate': performance_summary.get('win_rate', 0.0),
            'total_trades': performance_summary.get('total_trades', 0),
            'total_fees': self.account_state.get_total_fees_paid() if self.account_state else 0.0,
            'final_equity': final_equity,
            'initial_balance': initial_balance,
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat(),
            'equity_curve': self.equity_curve,
            'trade_history': self.trade_history,
            'performance': performance,
            'performance_summary': performance_summary,
            'risk_summary': risk_summary,
            'risk_report': {
                'total_exposure': risk_report.total_exposure,
                'total_exposure_percent': risk_report.total_exposure_percent,
                'total_risk': risk_report.total_risk,
                'total_risk_percent': risk_report.total_risk_percent,
                'unrealized_pnl': risk_report.unrealized_pnl,
                'realized_pnl_today': risk_report.realized_pnl_today,
                'realized_pnl_week': risk_report.realized_pnl_week,
                'realized_pnl_month': risk_report.realized_pnl_month,
                'num_open_positions': risk_report.num_open_positions,
                'win_rate': risk_report.win_rate,
                'avg_win': risk_report.avg_win,
                'avg_loss': risk_report.avg_loss,
                'profit_factor': risk_report.profit_factor,
                'max_drawdown_percent': risk_report.max_drawdown_percent,
                'current_drawdown_percent': risk_report.current_drawdown_percent
            },
            'state_summary': self.state_manager.get_state_summary() if self.state_manager else {}
        }
        
        return results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {}
        
        # Extract equity values
        equity_values = [point['equity'] for point in self.equity_curve]
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            returns.append(ret)
        
        if not returns:
            return {}
        
        # Calculate Sharpe ratio (annualized)
        import numpy as np
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            # Assuming 15m candles, 96 per day, 35040 per year
            sharpe_ratio = (avg_return / std_return) * np.sqrt(35040)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        peak = equity_values[0]
        max_drawdown = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate win rate from trade history
        winning_trades = sum(
            1 for trade in self.portfolio_tracker.get_trade_history()
            if trade.realized_pnl > 0
        )
        total_trades = len(self.portfolio_tracker.get_trade_history())
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_return': avg_return,
            'std_return': std_return
        }
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """
        Get the equity curve.
        
        Returns:
            List of equity points
        """
        return self.equity_curve
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get the trade history.
        
        Returns:
            List of trades
        """
        return self.trade_history
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get backtest statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'config': self.config.to_dict(),
            'time_controller': {
                'current_time': self.time_controller.get_current_time().isoformat() if self.time_controller else None,
                'candle_count': self.time_controller.get_candle_count() if self.time_controller else 0,
                'progress': self.time_controller.get_progress() if self.time_controller else 0.0
            },
            'account_state': self.account_state.get_stats() if self.account_state else {},
            'order_simulator': self.order_simulator.get_stats() if self.order_simulator else {},
            'data_cache': self.data_cache.get_stats() if self.data_cache else {},
            'risk_manager': self.risk_manager.get_risk_summary(
                self.portfolio_tracker.get_equity()
            ) if self.risk_manager and self.portfolio_tracker else {},
            'portfolio_tracker': self.portfolio_tracker.get_performance_summary() if self.portfolio_tracker else {},
            'state_manager': self.state_manager.get_state_summary() if self.state_manager else {}
        }
