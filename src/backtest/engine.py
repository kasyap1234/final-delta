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
import yaml

from src.backtest.config import BacktestConfig
from src.backtest.time_controller import TimeController
from src.backtest.account_state import AccountState
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.mock.order_simulator import (
    BacktestOrderSimulator,
    OHLCV as BacktestOHLCV,
)
from src.backtest.mock.data_cache import BacktestDataCache
from src.backtest.mock.stream_manager import BacktestStreamManager
from src.backtest.mock.exchange_client import BacktestExchangeClient
from src.backtest.strategy_engine import (
    BacktestStrategyEngine,
    StrategyConfig,
    TradeDirection,
)
from src.indicators.market_regime import get_regime_profile
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

        # Track peak prices for trailing stops
        self.position_peak_prices: Dict[str, float] = {}

        # Track if breakeven stop has been set for each position
        self.breakeven_set: Dict[str, bool] = {}

        # Track pending entry/exit orders until they are actually filled.
        self.pending_entry_orders: Dict[str, Dict[str, Any]] = {}
        self.pending_exit_orders: Dict[str, Dict[str, Any]] = {}

        # Execution-level events (submitted/rejected/filled) for auditability.
        self.order_events: List[Dict[str, Any]] = []

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

        # Ensure backtest strategy/risk settings default to live bot settings
        # when year-specific backtest configs omit trading_bot section.
        self._ensure_live_strategy_config()

        # Initialize time controller
        self.time_controller = TimeController(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            timeframe=self.config.timeframe,
        )

        # Initialize account state
        self.account_state = AccountState(
            initial_balance=self.config.initial_balance,
            currency=self.config.initial_currency,
        )

        # Initialize fee calculator and attach to account state
        fee_calculator = self.config.create_fee_calculator()
        self.account_state.set_fee_calculator(fee_calculator)

        # Initialize order simulator
        from src.backtest.mock.order_simulator import SimulatorConfig

        simulator_config = SimulatorConfig()
        simulator_config.maker_fee_rate = (
            self.config.fee_config.maker_fee_rate
            if getattr(self.config, "fee_config", None) is not None
            else self.config.maker_fee_pct / 100.0
        )
        simulator_config.taker_fee_rate = (
            self.config.fee_config.taker_fee_rate
            if getattr(self.config, "fee_config", None) is not None
            else self.config.taker_fee_pct / 100.0
        )
        # slippage_percent is configured in percent units (0.01 => 0.01%)
        simulator_config.market_order_slippage_bps = max(
            0.0, float(self.config.slippage_pct) * 100.0
        )
        latency_ms = max(0, int(self.config.latency_ms))
        simulator_config.enable_latency = latency_ms > 0
        simulator_config.latency_config.base_order_submit_ms = float(latency_ms)
        simulator_config.latency_config.base_order_cancel_ms = float(
            max(1, latency_ms // 2)
        )
        simulator_config.latency_config.base_order_fill_ms = float(latency_ms)
        self.order_simulator = BacktestOrderSimulator(
            config=simulator_config, account_state=self.account_state
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
            initial_balance=self.config.initial_balance,
        )

        # Set correlation groups if available in config
        if (
            hasattr(self.config, "correlation_groups")
            and self.config.correlation_groups
        ):
            self.risk_manager.set_correlation_groups(self.config.correlation_groups)

        # Initialize strategy engine
        strategy_config = StrategyConfig()
        if self.config.trading_bot_config:
            strategy_section = self.config.trading_bot_config.get("strategy")
            if strategy_section is None:
                strategy_section = self.config.trading_bot_config.get("indicators", {})
            risk_section = self.config.trading_bot_config.get("risk_management")
            if risk_section is None:
                risk_section = self.config.trading_bot_config.get("risk", {})

            strategy_config.ema_short = strategy_section.get(
                "ema_fast", strategy_section.get("ema_short", strategy_config.ema_short)
            )
            strategy_config.ema_medium = strategy_section.get(
                "ema_medium", strategy_config.ema_medium
            )
            strategy_config.ema_long = strategy_section.get(
                "ema_slow", strategy_section.get("ema_long", strategy_config.ema_long)
            )
            strategy_config.ema_trend = strategy_section.get(
                "ema_trend", strategy_config.ema_trend
            )
            strategy_config.rsi_period = strategy_section.get(
                "rsi_period", strategy_config.rsi_period
            )
            strategy_config.atr_period = strategy_section.get(
                "atr_period", strategy_config.atr_period
            )
            strategy_config.pivot_lookback = strategy_section.get(
                "pivot_lookback", strategy_config.pivot_lookback
            )
            strategy_config.strong_signal_threshold = strategy_section.get(
                "strong_signal_threshold", strategy_config.strong_signal_threshold
            )
            strategy_config.weak_signal_threshold = strategy_section.get(
                "weak_signal_threshold", strategy_config.weak_signal_threshold
            )
            strategy_config.min_signal_confidence = strategy_section.get(
                "min_signal_confidence", strategy_config.min_signal_confidence
            )
            strategy_config.min_adx_for_entry = strategy_section.get(
                "min_adx_for_entry", strategy_config.min_adx_for_entry
            )
            strategy_config.min_ema_spread_for_entry = strategy_section.get(
                "min_ema_spread_for_entry", strategy_config.min_ema_spread_for_entry
            )
            strategy_config.max_atr_percent_for_entry = strategy_section.get(
                "max_atr_percent_for_entry", strategy_config.max_atr_percent_for_entry
            )
            strategy_config.atr_percent_lookback = strategy_section.get(
                "atr_percent_lookback", strategy_config.atr_percent_lookback
            )

            strategy_config.max_risk_per_trade_percent = risk_section.get(
                "max_risk_per_trade_percent", strategy_config.max_risk_per_trade_percent
            )
            strategy_config.take_profit_rr_ratio = risk_section.get(
                "take_profit_r_ratio",
                risk_section.get(
                    "take_profit_rr_ratio", strategy_config.take_profit_rr_ratio
                ),
            )

            strategy_config.adx_strong_trend = strategy_section.get(
                "adx_strong_trend", strategy_config.adx_strong_trend
            )
            strategy_config.adx_weak_trend = strategy_section.get(
                "adx_weak_trend", strategy_config.adx_weak_trend
            )
            strategy_config.bb_squeeze_threshold = strategy_section.get(
                "bb_squeeze_threshold", strategy_config.bb_squeeze_threshold
            )
            strategy_config.bb_volatile_threshold = strategy_section.get(
                "bb_volatile_threshold", strategy_config.bb_volatile_threshold
            )
            strategy_config.min_regime_confidence = strategy_section.get(
                "min_confidence", strategy_config.min_regime_confidence
            )

        self.strategy_engine = BacktestStrategyEngine(
            config=strategy_config, account_balance=self.config.initial_balance
        )

        logger.info("Backtest components initialized")

    def _ensure_live_strategy_config(self) -> None:
        """Backfill missing trading_bot config from live/default config files."""
        if self.config.trading_bot_config:
            return

        candidate_paths = [
            Path("config/config.yaml"),
            Path("config/config.example.yaml"),
            Path("config/backtest.yaml"),
        ]

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}

                if "trading_bot" in raw and isinstance(raw["trading_bot"], dict):
                    self.config.trading_bot_config = raw["trading_bot"]
                    logger.info(f"Loaded strategy parity config from {path}")
                    return

                # Live config files typically have strategy/risk/order at root.
                if {"strategy", "risk_management"}.issubset(raw.keys()):
                    self.config.trading_bot_config = raw
                    logger.info(f"Loaded strategy parity config from {path}")
                    return
            except Exception as exc:
                logger.warning(f"Failed reading fallback trading config {path}: {exc}")

        logger.warning(
            "No live trading config found for parity fallback; using strategy defaults."
        )

    def _get_risk_config(self) -> Dict[str, Any]:
        """Extract risk configuration from backtest config."""
        risk_config = {
            "max_total_exposure_percent": 150.0,
            "max_total_risk_percent": 5.0,
            "max_positions": 10,
            "daily_loss_limit_percent": 3.0,
            "weekly_loss_limit_percent": 10.0,
            "max_correlated_exposure": 15.0,
            "correlation_threshold": 0.7,
            "default_risk_percent": 1.0,
            "default_atr_multiplier": 2.0,
            "default_risk_reward_ratio": 2.0,
            "min_position_size": 0.001,
            "max_position_size": 100.0,
            "trading_fee_percent": self.config.fee_rate * 100
            if hasattr(self.config, "fee_rate")
            else 0.1,
        }

        # Override with config if available
        if self.config.trading_bot_config:
            risk = self.config.trading_bot_config.get("risk_management")
            if risk is None:
                risk = self.config.trading_bot_config.get("risk")
        else:
            risk = None

        if risk:
            risk_config["max_total_risk_percent"] = (
                risk.get("max_risk_per_trade_percent", 5.0) * 2.5
            )
            risk_config["max_positions"] = risk.get("max_positions", 10)
            risk_config["daily_loss_limit_percent"] = risk.get(
                "daily_loss_limit_percent", 3.0
            )
            risk_config["weekly_loss_limit_percent"] = risk.get(
                "weekly_loss_limit_percent", 10.0
            )
            risk_config["default_risk_percent"] = risk.get(
                "risk_per_trade_percent", 1.0
            )
            risk_config["default_atr_multiplier"] = risk.get(
                "stop_loss_atr_multiplier", 2.0
            )
            risk_config["default_risk_reward_ratio"] = risk.get(
                "take_profit_rr_ratio", 2.0
            )

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
            historical_data=self.historical_data, data_cache=self.data_cache
        )

        # Initialize exchange client
        self.exchange_client = BacktestExchangeClient(
            historical_data=self.historical_data,
            order_simulator=self.order_simulator,
            account_state=self.account_state,
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

            # Update account state and portfolio tracker with current prices (single price fetch)
            await self._update_prices()

            # Record equity point
            self._record_equity_point()

            # Log progress
            if (i + 1) % 1000 == 0:
                progress = self.time_controller.get_progress()
                equity = self.portfolio_tracker.get_equity()
                logger.info(
                    f"Progress: {progress:.1%} ({i + 1}/{len(all_timestamps)} candles) - "
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
                symbol=symbol, timeframe=self.config.timeframe, timestamp=timestamp
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
                    volume=float(candle.volume),
                )

                # Process orders (check for fills on existing orders)
                fills = self.order_simulator.process_orders(backtest_candle)
                self._process_order_fills(fills, backtest_candle)
                self._cleanup_symbol_order_state(symbol, backtest_candle)

                # Check for stop loss / take profit on active positions
                await self._check_position_exits(symbol, backtest_candle)

                # Run strategy logic to generate new signals
                await self._execute_strategy(symbol, candle, timestamp)

    def _process_order_fills(
        self, fills: List[Dict[str, Any]], candle: BacktestOHLCV
    ) -> None:
        """Apply simulator fills to portfolio/risk/state in a live-like way."""
        for fill in fills:
            order_id = fill.get("order_id")
            if not order_id:
                continue

            self.order_events.append(
                {
                    "timestamp": candle.timestamp.isoformat(),
                    "event": "fill",
                    "symbol": fill.get("symbol"),
                    "order_id": order_id,
                    "side": fill.get("side"),
                    "amount": fill.get("amount"),
                    "price": fill.get("price"),
                    "fee": fill.get("fee"),
                }
            )

            if order_id in self.pending_entry_orders:
                meta = self.pending_entry_orders.pop(order_id)
                self._cancel_remaining_if_open(order_id, candle.timestamp)
                self._open_position_from_fill(
                    meta=meta, fill=fill, timestamp=candle.timestamp
                )
                continue

            if order_id in self.pending_exit_orders:
                meta = self.pending_exit_orders.pop(order_id)
                self._cancel_remaining_if_open(order_id, candle.timestamp)
                self._close_position_from_fill(
                    meta=meta, fill=fill, timestamp=candle.timestamp
                )

    def _cancel_remaining_if_open(self, order_id: str, timestamp: datetime) -> None:
        """Cancel any unfilled remainder after first actionable fill."""
        order = (
            self.order_simulator.get_order(order_id) if self.order_simulator else None
        )
        if order and order.is_active and getattr(order, "remaining", 0.0) > 0.0:
            self.order_simulator.cancel_order(order_id, current_time=timestamp)

    def _timeframe_to_seconds(self) -> int:
        """Convert configured timeframe (e.g. 15m/1h/1d) to seconds."""
        tf = str(self.config.timeframe).strip().lower()
        if len(tf) < 2:
            return 900
        try:
            value = int(tf[:-1])
        except ValueError:
            return 900
        unit = tf[-1]
        if unit == "m":
            return value * 60
        if unit == "h":
            return value * 3600
        if unit == "d":
            return value * 86400
        return 900

    def _submit_fallback_entry_order(
        self, meta: Dict[str, Any], candle: BacktestOHLCV
    ) -> None:
        """Submit taker/aggressive fallback entry order after stale/timeout maker order."""
        exec_cfg = meta.get("execution_cfg", {})
        fallback_type = str(exec_cfg.get("fallback_order_type", "market")).lower()
        if fallback_type not in {"limit", "market"}:
            fallback_type = "market"

        order_side = "buy" if meta.get("side") == "long" else "sell"
        fallback_price = float(candle.close) if fallback_type == "limit" else None
        fallback_tif = str(exec_cfg.get("fallback_time_in_force", "IOC"))

        order = self.order_simulator.create_order(
            symbol=meta["symbol"],
            side=order_side,
            order_type=fallback_type,
            amount=float(meta["position_size"]),
            price=fallback_price,
            post_only=False,
            time_in_force=fallback_tif,
            params={"fallback": True},
        )
        self.order_simulator.submit_order(order, current_time=candle.timestamp)

        new_meta = dict(meta)
        new_meta.update(
            {
                "order_id": order.id,
                "submitted_at": candle.timestamp.isoformat(),
                "submitted_ts": float(candle.timestamp.timestamp()),
                "is_fallback": True,
                "fallback_attempted": True,
            }
        )
        # Preserve regime_value from original meta
        if "regime_value" not in new_meta:
            new_meta["regime_value"] = meta.get("regime_value", "unknown")
        self.pending_entry_orders[order.id] = new_meta
        self.order_events.append(
            {
                "timestamp": candle.timestamp.isoformat(),
                "event": "entry_fallback_submitted",
                "symbol": meta["symbol"],
                "order_id": order.id,
                "from_order_id": meta.get("failed_order_id", meta.get("order_id")),
                "side": order_side,
                "order_type": fallback_type,
                "price": fallback_price,
                "amount": float(meta["position_size"]),
            }
        )

    def _cleanup_symbol_order_state(self, symbol: str, candle: BacktestOHLCV) -> None:
        """Drop stale orders and escalate unfilled entry orders per execution policy."""
        stale_statuses = {"rejected", "cancelled", "expired"}
        now_ts = float(candle.timestamp.timestamp())
        timeframe_seconds = self._timeframe_to_seconds()

        entry_to_remove: List[str] = []
        fallback_queue: List[Dict[str, Any]] = []
        for order_id, meta in list(self.pending_entry_orders.items()):
            if meta.get("symbol") != symbol:
                continue
            order = (
                self.order_simulator.get_order(order_id)
                if self.order_simulator
                else None
            )
            if not order:
                continue

            exec_cfg = meta.get("execution_cfg", {})
            fallback_allowed = (
                bool(exec_cfg.get("enable_taker_fallback", True))
                and not bool(meta.get("is_fallback", False))
                and not bool(meta.get("fallback_attempted", False))
            )

            if order.status.value in stale_statuses:
                entry_to_remove.append(order_id)
                self.order_events.append(
                    {
                        "timestamp": candle.timestamp.isoformat(),
                        "event": "entry_order_stale",
                        "symbol": symbol,
                        "order_id": order_id,
                        "status": order.status.value,
                    }
                )
                if fallback_allowed:
                    queued_meta = dict(meta)
                    queued_meta["failed_order_id"] = order_id
                    queued_meta["fallback_attempted"] = True
                    fallback_queue.append(queued_meta)
                continue

            if order.is_active and not bool(meta.get("is_fallback", False)):
                timeout_seconds = float(
                    exec_cfg.get("entry_timeout_seconds", 0.0) or 0.0
                )
                if timeout_seconds <= 0.0:
                    timeout_candles = max(
                        1, int(exec_cfg.get("entry_timeout_candles", 4))
                    )
                    timeout_seconds = float(timeout_candles * timeframe_seconds)
                submitted_ts = float(meta.get("submitted_ts", now_ts))
                if now_ts - submitted_ts >= timeout_seconds:
                    self.order_simulator.cancel_order(
                        order_id, current_time=candle.timestamp
                    )
                    entry_to_remove.append(order_id)
                    self.order_events.append(
                        {
                            "timestamp": candle.timestamp.isoformat(),
                            "event": "entry_timeout_cancelled",
                            "symbol": symbol,
                            "order_id": order_id,
                            "timeout_seconds": timeout_seconds,
                        }
                    )
                    if fallback_allowed:
                        queued_meta = dict(meta)
                        queued_meta["failed_order_id"] = order_id
                        queued_meta["fallback_attempted"] = True
                        fallback_queue.append(queued_meta)

        for order_id in entry_to_remove:
            self.pending_entry_orders.pop(order_id, None)

        for queued_meta in fallback_queue:
            self._submit_fallback_entry_order(queued_meta, candle)

        exit_to_remove = []
        for order_id, meta in list(self.pending_exit_orders.items()):
            if meta.get("symbol") != symbol:
                continue
            order = (
                self.order_simulator.get_order(order_id)
                if self.order_simulator
                else None
            )
            if order and order.status.value in stale_statuses:
                exit_to_remove.append(order_id)
                self.order_events.append(
                    {
                        "timestamp": candle.timestamp.isoformat(),
                        "event": "exit_order_stale",
                        "symbol": symbol,
                        "order_id": order_id,
                        "status": order.status.value,
                    }
                )
        for order_id in exit_to_remove:
            self.pending_exit_orders.pop(order_id, None)

    def _open_position_from_fill(
        self,
        meta: Dict[str, Any],
        fill: Dict[str, Any],
        timestamp: datetime,
    ) -> None:
        """Create portfolio/risk/state position only after entry order fill."""
        symbol = meta["symbol"]
        if self.portfolio_tracker.has_position(symbol):
            return

        side = meta["side"]
        entry_price = float(fill.get("price", meta["entry_price"]))
        position_size = float(fill.get("amount", meta["position_size"]))
        if position_size <= 0.0:
            return

        stop_loss = float(meta["stop_loss"])
        take_profit = float(meta["take_profit"])
        atr_value = float(meta.get("atr_value", 0.0))
        entry_fees = float(fill.get("fee", 0.0))

        stop_loss_distance = abs(entry_price - stop_loss)
        risk_amount = position_size * stop_loss_distance
        current_equity = self.portfolio_tracker.get_equity()

        position = self.portfolio_tracker.add_position(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            entry_time=timestamp,
            fees=entry_fees,
        )

        self.risk_manager.update_position_risk(
            position_id=position.position_id,
            risk_amount=risk_amount,
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_percent=(risk_amount / current_equity * 100)
            if current_equity > 0
            else 0.0,
        )

        self.state_manager.create_position(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=entry_price,
            position_id=position.position_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "risk_amount": risk_amount,
                "atr_at_entry": atr_value,
                "signal_strength": meta.get("signal_strength", "normal"),
                "entry_order_id": meta.get("order_id"),
            },
        )

        self.active_positions[symbol] = position.position_id

        if self.strategy_engine:
            # Get regime value from meta if available
            regime_value = meta.get("regime_value", "unknown")
            self.strategy_engine.register_position(
                symbol=symbol,
                entry_price=entry_price,
                position_type=side,
                position_size=position_size,
                stop_loss_price=stop_loss,
                atr=atr_value,
                regime_value=regime_value,
            )

        self.risk_manager.set_account_balance(
            self.portfolio_tracker.get_account_balance()
        )

        logger.info(
            f"Position filled for {symbol}: {side} {position_size:.4f} @ {entry_price:.2f}, "
            f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, fees: ${entry_fees:.2f}"
        )

    def _close_position_from_fill(
        self,
        meta: Dict[str, Any],
        fill: Dict[str, Any],
        timestamp: datetime,
    ) -> None:
        """Close position only after exit order fill."""
        symbol = meta["symbol"]
        position_id = meta["position_id"]
        exit_reason = meta["exit_reason"]

        position = self.portfolio_tracker.get_position(position_id)
        if position is None:
            return

        exit_price = float(fill.get("price", meta.get("requested_price", 0.0)))
        exit_fees = float(fill.get("fee", 0.0))

        trade_record = self.portfolio_tracker.close_position(
            position_id=position.position_id,
            exit_price=exit_price,
            exit_time=timestamp,
            exit_fees=exit_fees,
        )
        if not trade_record:
            return

        self.risk_manager.record_trade_result(
            symbol=symbol,
            realized_pnl=trade_record.realized_pnl,
            risk_amount=position.risk_amount,
        )

        self.state_manager.close_position(
            position_id=position.position_id,
            exit_price=exit_price,
            realized_pnl=trade_record.realized_pnl,
        )

        self.active_positions.pop(symbol, None)
        if self.strategy_engine:
            self.strategy_engine.close_position(symbol)
        self.position_peak_prices.pop(symbol, None)
        self.breakeven_set.pop(symbol, None)

        self.risk_manager.set_account_balance(
            self.portfolio_tracker.get_account_balance()
        )

        self.trade_history.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "side": "sell" if position.side == "long" else "buy",
                "size": position.size,
                "price": exit_price,
                "exit_price": exit_price,
                "pnl": trade_record.realized_pnl,
                "fees": exit_fees,
                "reason": exit_reason,
                "entry_price": position.entry_price,
                "position_id": position.position_id,
            }
        )

        logger.info(
            f"Position closed for {symbol}: {exit_reason} @ {exit_price:.2f}, "
            f"P&L: ${trade_record.realized_pnl:.2f}, fees: ${exit_fees:.2f}"
        )

    async def _update_prices(self) -> None:
        """Update account state and portfolio tracker with current prices (single fetch)."""
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

        # Update all position prices
        self.portfolio_tracker.update_all_positions(current_prices)

        # Update risk manager with position P&L
        for position in self.portfolio_tracker.get_all_positions():
            self.risk_manager.update_position_pnl(
                position.position_id, position.unrealized_pnl
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
        if self._has_pending_exit_for_symbol(symbol):
            return

        stop_loss = position.stop_loss
        take_profit = position.take_profit
        side = position.side

        exit_triggered = False
        exit_price = None
        exit_reason = None

        # Update peak price tracking for trailing stops
        if symbol not in self.position_peak_prices:
            self.position_peak_prices[symbol] = position.entry_price

        if side == "long":
            self.position_peak_prices[symbol] = max(
                self.position_peak_prices[symbol], candle.high
            )
        else:
            self.position_peak_prices[symbol] = min(
                self.position_peak_prices[symbol], candle.low
            )

        # Check trailing stop using regime profile parameters
        peak_price = self.position_peak_prices[symbol]
        indicators = (
            self.strategy_engine.indicator_manager.get_latest(symbol)
            if self.strategy_engine
            else None
        )
        atr = getattr(indicators, "atr", None) if indicators else None
        if atr and atr > 0:
            # Get trailing stop params from current regime profile
            regime_val = "unknown"
            if (
                self.strategy_engine
                and self.strategy_engine.regime_detector
                and hasattr(self.strategy_engine.regime_detector, "_current_regime")
            ):
                regime_val = self.strategy_engine.regime_detector._current_regime.value
            profile = get_regime_profile(regime_val)

            trailing_activation = atr * profile["trailing_activation_atr"]
            trailing_distance = atr * profile["trailing_distance_atr"]
            breakeven_atr = profile.get("breakeven_atr", 1.0)

            if side == "long":
                profit_from_entry = peak_price - position.entry_price

                breakeven_threshold = atr * breakeven_atr
                if (
                    profit_from_entry > breakeven_threshold
                    and symbol not in self.breakeven_set
                ):
                    position.stop_loss = position.entry_price
                    self.breakeven_set[symbol] = True
                    stop_loss = position.stop_loss

                if profit_from_entry > trailing_activation:
                    trailing_stop = peak_price - trailing_distance
                    if candle.low <= trailing_stop:
                        exit_triggered = True
                        exit_price = max(candle.open, trailing_stop)
                        exit_reason = "trailing_stop"
            else:  # short
                profit_from_entry = position.entry_price - peak_price

                breakeven_threshold = atr * breakeven_atr
                if (
                    profit_from_entry > breakeven_threshold
                    and symbol not in self.breakeven_set
                ):
                    position.stop_loss = position.entry_price
                    self.breakeven_set[symbol] = True
                    stop_loss = position.stop_loss

                if profit_from_entry > trailing_activation:
                    trailing_stop = peak_price + trailing_distance
                    if candle.high >= trailing_stop:
                        exit_triggered = True
                        exit_price = min(candle.open, trailing_stop)
                        exit_reason = "trailing_stop"

        # Check stop loss / take profit (highest priority — overrides trailing stop)
        if side == "long":
            if candle.low <= stop_loss:
                exit_triggered = True
                exit_price = max(candle.open, stop_loss)
                exit_reason = "stop_loss"
            elif candle.high >= take_profit:
                exit_triggered = True
                exit_price = min(candle.open, take_profit)
                exit_reason = "take_profit"
        else:  # short
            if candle.high >= stop_loss:
                exit_triggered = True
                exit_price = min(candle.open, stop_loss)
                exit_reason = "stop_loss"
            elif candle.low <= take_profit:
                exit_triggered = True
                exit_price = max(candle.open, take_profit)
                exit_reason = "take_profit"

        # Check TP levels from exit manager (partial exits)
        if not exit_triggered and self.strategy_engine:
            position_id = f"{symbol}_{side}"
            tp_exit = self.strategy_engine.exit_manager._check_take_profit_levels(
                position_id=position_id,
                symbol=symbol,
                side=side,
                current_price=candle.close,
            )
            if tp_exit:
                exit_triggered = True
                exit_price = tp_exit.price
                exit_reason = f"take_profit_level_{tp_exit.reason}"

        # Check signal-based exits (matching live bot's _manage_positions)
        if not exit_triggered and self.strategy_engine:
            candle_data = {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            exit_signal = self.strategy_engine.check_position_exit(
                symbol=symbol,
                candle=candle_data,
                timestamp=candle.timestamp,
                entry_price=position.entry_price,
                position_type=side,
            )
            if exit_signal:
                exit_triggered = True
                exit_price = candle.close
                exit_reason = f"signal_{exit_signal.reason}"

        if exit_triggered and exit_price:
            self._submit_exit_order(
                position=position,
                candle=candle,
                requested_price=exit_price,
                exit_reason=exit_reason,
            )

    def _calculate_exit_fees(
        self, size: float, price: float, exit_reason: str = "take_profit"
    ) -> float:
        """Calculate exit fees for a trade.

        Stop losses and trailing stops are market orders (taker fees).
        Take profits are limit orders (maker fees).

        Args:
            size: Position size
            price: Exit price
            exit_reason: Why the position was closed (stop_loss, trailing_stop, take_profit, signal_*)
        """
        fee_calculator = (
            self.account_state.get_fee_calculator() if self.account_state else None
        )
        if fee_calculator:
            from src.backtest.fees import OrderType

            # Determine order type based on exit reason
            # Stop losses and trailing stops execute as market orders (taker)
            # Take profits and signal exits can be limit orders (maker)
            if exit_reason in ("stop_loss", "trailing_stop"):
                order_type = OrderType.MARKET
                is_maker = False
            else:
                order_type = OrderType.LIMIT
                is_maker = True

            fee_result = fee_calculator.calculate_trade_fee(
                symbol="backtest_exit",
                amount=size,
                price=price,
                order_type=order_type,
                metadata={"is_maker": is_maker},
            )
            return fee_result["fee_paid"]

        position_value = size * price
        # Fallback: use taker fee for stops, maker for TP
        if exit_reason in ("stop_loss", "trailing_stop"):
            fee_rate = 0.0006  # taker
        else:
            fee_rate = 0.0002  # maker
        return position_value * fee_rate

    def _has_pending_entry_for_symbol(self, symbol: str) -> bool:
        return any(
            meta.get("symbol") == symbol for meta in self.pending_entry_orders.values()
        )

    def _has_pending_exit_for_symbol(self, symbol: str) -> bool:
        return any(
            meta.get("symbol") == symbol for meta in self.pending_exit_orders.values()
        )

    def _get_order_execution_config(self) -> Dict[str, Any]:
        """Return backtest execution settings aligned with live bot order config."""
        config = self.config.trading_bot_config or {}
        order_cfg = config.get("order", {})
        execution_cfg = config.get("execution", {})

        entry_mode = str(
            order_cfg.get("entry_execution_mode", "maker_then_taker")
        ).lower()
        if entry_mode not in {"maker_only", "maker_then_taker", "taker_only"}:
            entry_mode = "maker_then_taker"

        order_type = str(order_cfg.get("order_type", "limit")).lower()
        if order_type not in {"limit", "market"}:
            order_type = "limit"
        if entry_mode == "taker_only":
            order_type = "market"

        fallback_order_type = str(
            order_cfg.get("fallback_order_type", "market")
        ).lower()
        if fallback_order_type not in {"limit", "market"}:
            fallback_order_type = "market"

        post_only = bool(
            order_cfg.get("post_only", True if order_type == "limit" else False)
        )
        if entry_mode == "taker_only":
            post_only = False

        return {
            "entry_execution_mode": entry_mode,
            "order_type": order_type,
            "post_only": post_only,
            "price_offset_percent": float(order_cfg.get("price_offset_percent", 0.01)),
            "time_in_force": str(order_cfg.get("time_in_force", "GTC")),
            "entry_timeout_seconds": float(order_cfg.get("entry_timeout_seconds", 45)),
            "entry_timeout_candles": int(order_cfg.get("entry_timeout_candles", 4)),
            "enable_taker_fallback": bool(order_cfg.get("enable_taker_fallback", True)),
            "fallback_order_type": fallback_order_type,
            "fallback_time_in_force": str(
                order_cfg.get("fallback_time_in_force", "IOC")
            ),
            "max_slippage_percent": float(
                execution_cfg.get("max_slippage_percent", 0.5)
            ),
        }

    def _build_entry_order_price(
        self, side: str, decision_price: float, config: Dict[str, Any]
    ) -> Optional[float]:
        if config["order_type"] == "market":
            return None

        offset_multiplier = max(0.0, config["price_offset_percent"]) / 100.0
        if side == "buy":
            return decision_price * (1.0 - offset_multiplier)
        return decision_price * (1.0 + offset_multiplier)

    def _submit_exit_order(
        self,
        position: Any,
        candle: BacktestOHLCV,
        requested_price: float,
        exit_reason: str,
    ) -> None:
        """Submit an exit order and wait for simulator fill before booking close."""
        symbol = position.symbol
        if self._has_pending_exit_for_symbol(symbol):
            return

        close_side = "sell" if position.side == "long" else "buy"
        # Live bot close flow uses close_position() with market/IOC semantics.
        order_type = "market"
        order_price = None
        post_only = False
        tif = "IOC"

        order = self.order_simulator.create_order(
            symbol=symbol,
            side=close_side,
            order_type=order_type,
            amount=position.size,
            price=order_price,
            post_only=post_only,
            time_in_force=tif,
            params={"exit_reason": exit_reason, "position_id": position.position_id},
        )
        self.order_simulator.submit_order(order, current_time=candle.timestamp)

        self.pending_exit_orders[order.id] = {
            "order_id": order.id,
            "position_id": position.position_id,
            "symbol": symbol,
            "exit_reason": exit_reason,
            "requested_price": requested_price,
        }
        self.order_events.append(
            {
                "timestamp": candle.timestamp.isoformat(),
                "event": "exit_submitted",
                "symbol": symbol,
                "order_id": order.id,
                "side": close_side,
                "order_type": order_type,
                "price": order_price,
                "amount": position.size,
                "exit_reason": exit_reason,
            }
        )

    async def _execute_strategy(
        self, symbol: str, candle: Any, timestamp: datetime
    ) -> None:
        """
        Execute trading strategy for a symbol with risk checks.

        Args:
            symbol: Trading symbol
            candle: Candle data
            timestamp: Current timestamp
        """
        # Skip if we already have a filled or pending position for this symbol.
        if self.portfolio_tracker.has_position(
            symbol
        ) or self._has_pending_entry_for_symbol(symbol):
            return

        current_equity = self.portfolio_tracker.get_equity()
        current_exposure = self.portfolio_tracker.get_total_exposure()

        # Prepare candle data for strategy
        candle_data = {
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume),
        }

        # Run strategy
        decision = self.strategy_engine.process_candle(
            symbol=symbol,
            candle=candle_data,
            timestamp=timestamp,
            current_balance=current_equity,
        )

        if not decision:
            return

        # Build trade intent from strategy decision
        signal = decision["signal"]
        position_size = decision["position_size"]
        entry_price = decision["entry_price"]
        stop_loss = decision["stop_loss"]
        take_profit = decision["take_profit"]
        atr_value = decision.get("atr", 0)

        # Determine side
        side = "long" if signal.direction == TradeDirection.LONG else "short"

        # Clamp requested size to portfolio limits before final risk check.
        current_positions = self.portfolio_tracker.get_positions()
        clamped_size = self.risk_manager.clamp_position_size_to_limits(
            symbol=symbol,
            requested_size=position_size,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            current_positions=current_positions,
            account_balance=current_equity,
        )
        if clamped_size <= 0.0:
            return
        if clamped_size < position_size:
            logger.info(
                f"Clamped {symbol} size from {position_size:.6f} to {clamped_size:.6f} "
                "to satisfy risk/exposure limits."
            )
        position_size = clamped_size

        # Perform pre-trade risk checks
        risk_check = self.risk_manager.can_open_position(
            symbol=symbol,
            position_size=position_size,
            stop_loss_price=stop_loss,
            entry_price=entry_price,
            current_positions=current_positions,
            account_balance=current_equity,
        )

        if not risk_check.can_trade:
            logger.warning(f"Risk check failed for {symbol}: {risk_check.reason}")
            return

        # Check if we have enough balance (leverage-aware)
        position_value = position_size * entry_price
        max_leverage = 5.0  # Default
        if self.config.trading_bot_config:
            risk_cfg = self.config.trading_bot_config.get("risk")
            if risk_cfg is None:
                risk_cfg = self.config.trading_bot_config.get("risk_management", {})
            max_leverage = risk_cfg.get("max_leverage", 5.0)

        required_margin = position_value / max_leverage
        used_margin = current_exposure / max_leverage
        available_margin = current_equity - used_margin

        if required_margin > available_margin * 0.95:
            logger.warning(
                f"Insufficient margin for {symbol} trade. "
                f"Position value: ${position_value:.2f}, Required margin: ${required_margin:.2f}, "
                f"Available: ${available_margin:.2f}"
            )
            return

        # Submit entry order to simulator; position is opened only on subsequent fill.
        order_cfg = self._get_order_execution_config()
        order_side = "buy" if side == "long" else "sell"
        entry_mode = order_cfg.get("entry_execution_mode", "maker_then_taker")
        if entry_mode == "taker_only":
            sim_order_type = "market"
            order_price = None
            post_only = False
            tif = str(order_cfg.get("fallback_time_in_force", "IOC"))
        else:
            order_price = self._build_entry_order_price(
                order_side, entry_price, order_cfg
            )
            sim_order_type = order_cfg["order_type"]
            if sim_order_type == "limit" and order_cfg["post_only"]:
                sim_order_type = "post_only_limit"
            post_only = sim_order_type == "post_only_limit"
            tif = order_cfg["time_in_force"]

        order = self.order_simulator.create_order(
            symbol=symbol,
            side=order_side,
            order_type=sim_order_type,
            amount=position_size,
            price=order_price,
            post_only=post_only,
            time_in_force=tif,
            params={"strategy_signal": getattr(signal, "direction", None)},
        )
        self.order_simulator.submit_order(order, current_time=timestamp)

        self.pending_entry_orders[order.id] = {
            "order_id": order.id,
            "symbol": symbol,
            "side": side,
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr_value,
            "signal_strength": getattr(
                signal, "confidence", getattr(signal, "strength", 0.0)
            ),
            "submitted_at": timestamp.isoformat(),
            "submitted_ts": float(timestamp.timestamp()),
            "execution_cfg": dict(order_cfg),
            "is_fallback": False,
            "fallback_attempted": False,
            "regime_value": decision.get("regime_value", "unknown"),
        }
        self.order_events.append(
            {
                "timestamp": timestamp.isoformat(),
                "event": "entry_submitted",
                "symbol": symbol,
                "order_id": order.id,
                "side": order_side,
                "order_type": sim_order_type,
                "price": order_price,
                "amount": position_size,
                "decision_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        )

    def _record_equity_point(self) -> None:
        """Record an equity point for the current time."""
        equity = self.portfolio_tracker.get_equity()
        balance = self.portfolio_tracker.get_account_balance()
        unrealized_pnl = self.portfolio_tracker.calculate_unrealized_pnl()

        point = {
            "timestamp": self.time_controller.get_current_time().isoformat(),
            "equity": equity,
            "balance": balance,
            "free_balance": balance - self.portfolio_tracker.get_total_exposure(),
            "used_balance": self.portfolio_tracker.get_total_exposure(),
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.portfolio_tracker.get_daily_pnl(),
            "num_positions": len(self.portfolio_tracker.get_all_positions()),
            "total_risk": self.portfolio_tracker.get_total_risk(),
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
            timestamps.update(c.timestamp for c in candles)

        return sorted(timestamps)

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

        raw_win_rate = performance_summary.get("win_rate", 0.0)
        win_rate = (raw_win_rate / 100.0) if raw_win_rate > 1.0 else raw_win_rate

        results = {
            "total_return": total_return,
            "sharpe_ratio": performance.get("sharpe_ratio", 0.0),
            "max_drawdown": performance.get("max_drawdown", 0.0),
            # Keep top-level win_rate as decimal (0-1) for consistent percentage rendering.
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100.0,
            "total_trades": performance_summary.get("total_trades", 0),
            "total_fees": self._calculate_total_fees(),
            "final_equity": final_equity,
            "initial_balance": initial_balance,
            "start_date": self.config.start_date.isoformat(),
            "end_date": self.config.end_date.isoformat(),
            "equity_curve": self.equity_curve,
            "trade_history": self.trade_history,
            "orders": self.order_events,
            "performance": performance,
            "performance_summary": performance_summary,
            "risk_summary": risk_summary,
            "risk_report": {
                "total_exposure": risk_report.total_exposure,
                "total_exposure_percent": risk_report.total_exposure_percent,
                "total_risk": risk_report.total_risk,
                "total_risk_percent": risk_report.total_risk_percent,
                "unrealized_pnl": risk_report.unrealized_pnl,
                "realized_pnl_today": risk_report.realized_pnl_today,
                "realized_pnl_week": risk_report.realized_pnl_week,
                "realized_pnl_month": risk_report.realized_pnl_month,
                "num_open_positions": risk_report.num_open_positions,
                "win_rate": risk_report.win_rate,
                "avg_win": risk_report.avg_win,
                "avg_loss": risk_report.avg_loss,
                "profit_factor": risk_report.profit_factor,
                "max_drawdown_percent": risk_report.max_drawdown_percent,
                "current_drawdown_percent": risk_report.current_drawdown_percent,
            },
            "state_summary": self.state_manager.get_state_summary()
            if self.state_manager
            else {},
        }

        return results

    def _calculate_total_fees(self) -> float:
        """Calculate total fees from portfolio tracker trade history."""
        total_fees = 0.0
        if self.portfolio_tracker:
            for trade in self.portfolio_tracker.get_trade_history():
                total_fees += getattr(trade, "fees_paid", 0.0)
        return total_fees

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {}

        # Extract equity values
        equity_values = [point["equity"] for point in self.equity_curve]

        # Calculate returns
        returns = []
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] - equity_values[i - 1]) / equity_values[i - 1]
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
            1
            for trade in self.portfolio_tracker.get_trade_history()
            if trade.realized_pnl > 0
        )
        total_trades = len(self.portfolio_tracker.get_trade_history())
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100.0,
            "total_trades": total_trades,
            "avg_return": avg_return,
            "std_return": std_return,
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
            "config": self.config.to_dict(),
            "time_controller": {
                "current_time": self.time_controller.get_current_time().isoformat()
                if self.time_controller
                else None,
                "candle_count": self.time_controller.get_candle_count()
                if self.time_controller
                else 0,
                "progress": self.time_controller.get_progress()
                if self.time_controller
                else 0.0,
            },
            "account_state": self.account_state.get_stats()
            if self.account_state
            else {},
            "order_simulator": self.order_simulator.get_stats()
            if self.order_simulator
            else {},
            "pending_entry_orders": len(self.pending_entry_orders),
            "pending_exit_orders": len(self.pending_exit_orders),
            "order_events": len(self.order_events),
            "data_cache": self.data_cache.get_stats() if self.data_cache else {},
            "risk_manager": self.risk_manager.get_risk_summary(
                self.portfolio_tracker.get_equity()
            )
            if self.risk_manager and self.portfolio_tracker
            else {},
            "portfolio_tracker": self.portfolio_tracker.get_performance_summary()
            if self.portfolio_tracker
            else {},
            "state_manager": self.state_manager.get_state_summary()
            if self.state_manager
            else {},
        }
