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
        
        # Historical data
        self.historical_data: Dict[str, List[OHLCV]] = {}
        
        # State tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Strategy engine
        self.strategy_engine: Optional[BacktestStrategyEngine] = None
        
        # Track active positions
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        
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
        
        # Initialize order simulator
        self.order_simulator = BacktestOrderSimulator(
            config=self.config,
            account_state=self.account_state
        )
        
        # Initialize data loader
        self.data_loader = HistoricalDataLoader(self.config)
        
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
            
            # Update exchange client time
            self.exchange_client.set_current_time(current_time)
            
            # Update stream manager (pushes data to cache)
            await self.stream_manager.update_time(current_time)
            
            # Process orders for this candle
            await self._process_candle(timestamp)
            
            # Update account state with current prices
            await self._update_account_state()
            
            # Record equity point
            self._record_equity_point()
            
            # Log progress
            if (i + 1) % 1000 == 0:
                progress = self.time_controller.get_progress()
                logger.info(
                    f"Progress: {progress:.1%} ({i+1}/{len(all_timestamps)} candles)"
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
    
    async def _check_position_exits(self, symbol: str, candle: BacktestOHLCV) -> None:
        """
        Check if any active positions should be exited (stop loss or take profit).
        
        Args:
            symbol: Trading symbol
            candle: Current candle data
        """
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        side = position['side']
        
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
            # Close the position
            position_size = position['size']
            entry_price = position['entry_price']
            
            # Calculate P&L
            if side == 'long':
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            # Record the trade
            trade = {
                'timestamp': candle.timestamp.isoformat(),
                'symbol': symbol,
                'side': 'sell' if side == 'long' else 'buy',
                'size': position_size,
                'price': exit_price,
                'pnl': pnl,
                'reason': exit_reason,
                'entry_price': entry_price
            }
            self.trade_history.append(trade)
            
            # Update account state
            self.account_state.close_position(symbol, exit_price)
            
            # Remove from active positions
            del self.active_positions[symbol]
            
            logger.info(f"Position closed for {symbol}: {exit_reason} at {exit_price:.2f}, P&L: ${pnl:.2f}")
    
    async def _execute_strategy(self, symbol: str, candle: Any, timestamp: datetime) -> None:
        """
        Execute trading strategy for a symbol.
        
        Args:
            symbol: Trading symbol
            candle: Candle data
            timestamp: Current timestamp
        """
        # Skip if we already have an active position for this symbol
        if symbol in self.active_positions:
            return
        
        # Get current balance
        balance = self.account_state.get_balance()
        current_equity = self.account_state.get_equity()
        
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
        
        # Determine side
        side = 'long' if signal.direction == TradeDirection.LONG else 'short'
        
        # Check if we have enough balance
        position_value = position_size * entry_price
        if position_value > balance.free * 0.95:  # Leave some buffer
            logger.warning(f"Insufficient balance for {symbol} trade. Needed: ${position_value:.2f}, Free: ${balance.free:.2f}")
            return
        
        # Open position
        success = self.account_state.open_position(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price
        )
        
        if success:
            # Track active position
            self.active_positions[symbol] = {
                'side': side,
                'size': position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': timestamp
            }
            
            logger.info(
                f"Position opened for {symbol}: {side} {position_size:.4f} @ {entry_price:.2f}, "
                f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
            )
    
    def _record_equity_point(self) -> None:
        """Record an equity point for the current time."""
        equity = self.account_state.get_equity()
        balance = self.account_state.get_balance()
        
        point = {
            'timestamp': self.time_controller.get_current_time().isoformat(),
            'equity': equity,
            'balance': balance.total,
            'free_balance': balance.free,
            'used_balance': balance.used,
            'unrealized_pnl': equity - balance.total,
            'realized_pnl': self.account_state.get_total_realized_pnl(),
            'num_positions': len(self.active_positions)
        }
        
        self.equity_curve.append(point)
    
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
        final_equity = self.account_state.get_equity()
        initial_balance = self.config.initial_balance
        total_return = (final_equity - initial_balance) / initial_balance
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics()
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
            'max_drawdown': performance.get('max_drawdown', 0.0),
            'win_rate': performance.get('win_rate', 0.0),
            'total_trades': len(self.trade_history),
            'total_fees': self.account_state.get_total_fees_paid(),
            'final_equity': final_equity,
            'initial_balance': initial_balance,
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat(),
            'equity_curve': self.equity_curve,
            'trade_history': self.trade_history,
            'performance': performance
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
        
        # Calculate win rate
        winning_trades = sum(
            1 for trade in self.trade_history
            if trade.get('side') == 'sell' and trade.get('amount', 0) > 0
        )
        
        # This is a simplified win rate calculation
        # In a real implementation, you'd track entry/exit pairs
        win_rate = winning_trades / len(self.trade_history) if self.trade_history else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
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
            'data_cache': self.data_cache.get_stats() if self.data_cache else {}
        }
