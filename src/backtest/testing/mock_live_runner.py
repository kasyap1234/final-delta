"""
Mock live runner module for capturing live trading bot behavior.

This module provides a mock implementation that runs the live trading bot
in "paper trading" mode and captures all decisions and states for comparison
with backtest results.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class CapturedSignal:
    """Captured signal data from live trading."""
    timestamp: datetime
    symbol: str
    signal_type: str
    strength: float
    price: float
    indicators: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'price': self.price,
            'indicators': self.indicators,
            'reason': self.reason,
            'details': self.details
        }


@dataclass
class CapturedOrder:
    """Captured order data from live trading."""
    timestamp: datetime
    order_id: str
    symbol: str
    side: str
    order_type: str
    size: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    filled_size: float = 0.0
    fill_price: float = 0.0
    fill_timestamp: Optional[datetime] = None
    fees: float = 0.0
    status: str = "pending"
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'size': self.size,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'filled_size': self.filled_size,
            'fill_price': self.fill_price,
            'fill_timestamp': self.fill_timestamp.isoformat() if self.fill_timestamp else None,
            'fees': self.fees,
            'status': self.status,
            'slippage': self.slippage,
            'metadata': self.metadata
        }


@dataclass
class CapturedPosition:
    """Captured position data from live trading."""
    timestamp: datetime
    position_id: str
    symbol: str
    side: str
    entry_price: float
    current_price: float
    size: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    risk_amount: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'risk_amount': self.risk_amount,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'metadata': self.metadata
        }


@dataclass
class CapturedRiskCheck:
    """Captured risk check data from live trading."""
    timestamp: datetime
    symbol: str
    check_type: str
    allowed: bool
    position_size: float
    risk_amount: float
    risk_percent: float
    reason: str = ""
    current_exposure: float = 0.0
    current_positions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'check_type': self.check_type,
            'allowed': self.allowed,
            'position_size': self.position_size,
            'risk_amount': self.risk_amount,
            'risk_percent': self.risk_percent,
            'reason': self.reason,
            'current_exposure': self.current_exposure,
            'current_positions': self.current_positions,
            'metadata': self.metadata
        }


@dataclass
class CapturedState:
    """Captured state snapshot from live trading."""
    timestamp: datetime
    account_balance: float
    equity: float
    free_balance: float
    used_balance: float
    unrealized_pnl: float
    realized_pnl: float
    num_positions: int
    num_orders: int
    positions: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'account_balance': self.account_balance,
            'equity': self.equity,
            'free_balance': self.free_balance,
            'used_balance': self.used_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'num_positions': self.num_positions,
            'num_orders': self.num_orders,
            'positions': self.positions,
            'orders': self.orders,
            'metadata': self.metadata
        }


@dataclass
class CapturedHedge:
    """Captured hedge data from live trading."""
    timestamp: datetime
    hedge_id: str
    parent_position_id: str
    symbol: str
    hedge_symbol: str
    side: str
    size: float
    entry_price: float
    status: str
    trigger_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'hedge_id': self.hedge_id,
            'parent_position_id': self.parent_position_id,
            'symbol': self.symbol,
            'hedge_symbol': self.hedge_symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'status': self.status,
            'trigger_reason': self.trigger_reason,
            'metadata': self.metadata
        }


class LiveDataCapture:
    """Captures all live trading data for comparison."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the data capture.
        
        Args:
            output_dir: Directory to save captured data
        """
        self.output_dir = Path(output_dir) if output_dir else Path("live_capture")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture storage
        self.signals: List[CapturedSignal] = []
        self.orders: List[CapturedOrder] = []
        self.positions: List[CapturedPosition] = []
        self.risk_checks: List[CapturedRiskCheck] = []
        self.states: List[CapturedState] = []
        self.hedges: List[CapturedHedge] = []
        
        # Current state tracking
        self.current_positions: Dict[str, CapturedPosition] = {}
        self.current_orders: Dict[str, CapturedOrder] = {}
        
        # State snapshots at regular intervals
        self.state_snapshots: List[CapturedState] = []
        
        logger.info(f"LiveDataCapture initialized, output dir: {self.output_dir}")
    
    def capture_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        price: float,
        indicators: Optional[Dict[str, Any]] = None,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> CapturedSignal:
        """Capture a trading signal."""
        signal = CapturedSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            indicators=indicators or {},
            reason=reason,
            details=details or {}
        )
        self.signals.append(signal)
        logger.debug(f"Captured signal: {symbol} {signal_type} @ {price}")
        return signal
    
    def capture_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CapturedOrder:
        """Capture an order placement."""
        order = CapturedOrder(
            timestamp=datetime.now(),
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        self.orders.append(order)
        self.current_orders[order_id] = order
        logger.debug(f"Captured order: {order_id} {side} {size} @ {price}")
        return order
    
    def update_order_fill(
        self,
        order_id: str,
        filled_size: float,
        fill_price: float,
        fees: float,
        status: str = "filled"
    ):
        """Update an order with fill information."""
        if order_id in self.current_orders:
            order = self.current_orders[order_id]
            order.filled_size = filled_size
            order.fill_price = fill_price
            order.fill_timestamp = datetime.now()
            order.fees = fees
            order.status = status
            
            # Calculate slippage
            if order.price > 0:
                if order.side in ['buy', 'long']:
                    order.slippage = (fill_price - order.price) / order.price
                else:
                    order.slippage = (order.price - fill_price) / order.price
            
            logger.debug(f"Updated order fill: {order_id} @ {fill_price}")
    
    def capture_position_open(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        risk_amount: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CapturedPosition:
        """Capture a position opening."""
        position = CapturedPosition(
            timestamp=datetime.now(),
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            entry_time=datetime.now(),
            metadata=metadata or {}
        )
        self.positions.append(position)
        self.current_positions[position_id] = position
        logger.debug(f"Captured position open: {position_id} {side} {size} @ {entry_price}")
        return position
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        unrealized_pnl: float
    ):
        """Update a position with current market data."""
        if position_id in self.current_positions:
            position = self.current_positions[position_id]
            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl
    
    def capture_position_close(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str
    ):
        """Capture a position closing."""
        if position_id in self.current_positions:
            position = self.current_positions[position_id]
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.realized_pnl = realized_pnl
            position.exit_reason = exit_reason
            
            # Remove from current positions
            del self.current_positions[position_id]
            logger.debug(f"Captured position close: {position_id} P&L: {realized_pnl}")
    
    def capture_risk_check(
        self,
        symbol: str,
        check_type: str,
        allowed: bool,
        position_size: float,
        risk_amount: float,
        risk_percent: float,
        reason: str = "",
        current_exposure: float = 0.0,
        current_positions: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CapturedRiskCheck:
        """Capture a risk management check."""
        check = CapturedRiskCheck(
            timestamp=datetime.now(),
            symbol=symbol,
            check_type=check_type,
            allowed=allowed,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            reason=reason,
            current_exposure=current_exposure,
            current_positions=current_positions,
            metadata=metadata or {}
        )
        self.risk_checks.append(check)
        logger.debug(f"Captured risk check: {symbol} allowed={allowed}")
        return check
    
    def capture_state(
        self,
        account_balance: float,
        equity: float,
        free_balance: float,
        used_balance: float,
        unrealized_pnl: float,
        realized_pnl: float,
        num_positions: int,
        num_orders: int,
        positions: Optional[List[Dict[str, Any]]] = None,
        orders: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CapturedState:
        """Capture a state snapshot."""
        state = CapturedState(
            timestamp=datetime.now(),
            account_balance=account_balance,
            equity=equity,
            free_balance=free_balance,
            used_balance=used_balance,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            num_positions=num_positions,
            num_orders=num_orders,
            positions=positions or [],
            orders=orders or [],
            metadata=metadata or {}
        )
        self.states.append(state)
        self.state_snapshots.append(state)
        return state
    
    def capture_hedge(
        self,
        hedge_id: str,
        parent_position_id: str,
        symbol: str,
        hedge_symbol: str,
        side: str,
        size: float,
        entry_price: float,
        status: str,
        trigger_reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CapturedHedge:
        """Capture a hedge operation."""
        hedge = CapturedHedge(
            timestamp=datetime.now(),
            hedge_id=hedge_id,
            parent_position_id=parent_position_id,
            symbol=symbol,
            hedge_symbol=hedge_symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            status=status,
            trigger_reason=trigger_reason,
            metadata=metadata or {}
        )
        self.hedges.append(hedge)
        logger.debug(f"Captured hedge: {hedge_id} for {symbol}")
        return hedge
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state summary."""
        return {
            'timestamp': datetime.now().isoformat(),
            'num_signals': len(self.signals),
            'num_orders': len(self.orders),
            'num_positions': len(self.positions),
            'num_risk_checks': len(self.risk_checks),
            'num_states': len(self.states),
            'num_hedges': len(self.hedges),
            'current_positions': list(self.current_positions.keys()),
            'current_orders': list(self.current_orders.keys())
        }
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export all captured data to JSON."""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'summary': self.get_current_state(),
            'signals': [s.to_dict() for s in self.signals],
            'orders': [o.to_dict() for o in self.orders],
            'positions': [p.to_dict() for p in self.positions],
            'risk_checks': [r.to_dict() for r in self.risk_checks],
            'states': [s.to_dict() for s in self.states],
            'hedges': [h.to_dict() for h in self.hedges]
        }
        
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            Path(filepath).write_text(json_str)
        else:
            default_path = self.output_dir / f"live_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            default_path.write_text(json_str)
            filepath = str(default_path)
        
        logger.info(f"Exported live capture to {filepath}")
        return json_str
    
    def get_comparable_format(self) -> Dict[str, Any]:
        """
        Get data in format comparable to backtest output.
        
        Returns:
            Dictionary with standardized format matching backtest results
        """
        # Calculate final equity from last state
        final_equity = self.states[-1].equity if self.states else 0.0
        initial_equity = self.states[0].equity if self.states else 0.0
        
        # Build trade history from positions
        trade_history = []
        for pos in self.positions:
            if pos.exit_time:  # Only closed positions
                trade_history.append({
                    'timestamp': pos.exit_time.isoformat(),
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'exit_price': pos.exit_price,
                    'size': pos.size,
                    'pnl': pos.realized_pnl,
                    'reason': pos.exit_reason,
                    'risk_amount': pos.risk_amount
                })
        
        # Build equity curve from states
        equity_curve = []
        for state in self.states:
            equity_curve.append({
                'timestamp': state.timestamp.isoformat(),
                'equity': state.equity,
                'balance': state.account_balance,
                'free_balance': state.free_balance,
                'used_balance': state.used_balance,
                'unrealized_pnl': state.unrealized_pnl,
                'realized_pnl': state.realized_pnl,
                'num_positions': state.num_positions,
                'num_orders': state.num_orders
            })
        
        # Calculate performance metrics
        total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
        winning_trades = sum(1 for t in trade_history if t['pnl'] > 0)
        total_trades = len(trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_equity': final_equity,
            'initial_balance': initial_equity,
            'equity_curve': equity_curve,
            'trade_history': trade_history,
            'signals': [s.to_dict() for s in self.signals],
            'orders': [o.to_dict() for o in self.orders],
            'risk_checks': [r.to_dict() for r in self.risk_checks],
            'hedges': [h.to_dict() for h in self.hedges]
        }


class MockLiveRunner:
    """
    Mock live trading runner that captures all bot behavior.
    
    This class wraps the live trading bot and captures all
    decisions, signals, orders, and state changes for comparison
    with backtest results.
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        capture_interval: float = 1.0
    ):
        """
        Initialize the mock live runner.
        
        Args:
            config_path: Path to bot configuration
            output_dir: Directory to save captured data
            capture_interval: Interval between state captures in seconds
        """
        self.config_path = config_path
        self.output_dir = output_dir or "live_capture"
        self.capture_interval = capture_interval
        
        self.data_capture = LiveDataCapture(output_dir)
        self.running = False
        self._capture_task: Optional[asyncio.Task] = None
        
        # Callbacks for interception
        self._original_callbacks: Dict[str, Callable] = {}
        
        logger.info("MockLiveRunner initialized")
    
    async def run_paper_trading(
        self,
        duration_seconds: Optional[float] = None,
        max_signals: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run paper trading and capture all data.
        
        Args:
            duration_seconds: How long to run (None for indefinite)
            max_signals: Stop after capturing this many signals
            
        Returns:
            Captured data in comparable format
        """
        logger.info("Starting paper trading capture...")
        self.running = True
        
        # Start periodic state capture
        self._capture_task = asyncio.create_task(self._periodic_capture())
        
        try:
            # Run for specified duration or until stopped
            if duration_seconds:
                await asyncio.wait_for(
                    self._run_capture_loop(max_signals),
                    timeout=duration_seconds
                )
            else:
                await self._run_capture_loop(max_signals)
                
        except asyncio.TimeoutError:
            logger.info(f"Paper trading completed after {duration_seconds}s")
        except Exception as e:
            logger.error(f"Error in paper trading: {e}")
        finally:
            self.running = False
            if self._capture_task:
                self._capture_task.cancel()
                try:
                    await self._capture_task
                except asyncio.CancelledError:
                    pass
        
        # Export captured data
        return self.data_capture.get_comparable_format()
    
    async def _run_capture_loop(self, max_signals: Optional[int] = None):
        """Main capture loop."""
        while self.running:
            # Check if we should stop based on signal count
            if max_signals and len(self.data_capture.signals) >= max_signals:
                logger.info(f"Reached max signals ({max_signals}), stopping")
                break
            
            await asyncio.sleep(0.1)
    
    async def _periodic_capture(self):
        """Periodically capture state."""
        while self.running:
            try:
                # This would be called with actual bot state
                # For now, just sleep
                await asyncio.sleep(self.capture_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic capture: {e}")
    
    def intercept_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        price: float,
        **kwargs
    ) -> CapturedSignal:
        """Intercept and capture a signal."""
        return self.data_capture.capture_signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            **kwargs
        )
    
    def intercept_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: float,
        **kwargs
    ) -> CapturedOrder:
        """Intercept and capture an order."""
        return self.data_capture.capture_order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            **kwargs
        )
    
    def intercept_position_open(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        **kwargs
    ) -> CapturedPosition:
        """Intercept and capture a position opening."""
        return self.data_capture.capture_position_open(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            **kwargs
        )
    
    def intercept_position_close(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str
    ):
        """Intercept and capture a position closing."""
        self.data_capture.capture_position_close(
            position_id=position_id,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            exit_reason=exit_reason
        )
    
    def intercept_risk_check(
        self,
        symbol: str,
        check_type: str,
        allowed: bool,
        position_size: float,
        risk_amount: float,
        **kwargs
    ) -> CapturedRiskCheck:
        """Intercept and capture a risk check."""
        return self.data_capture.capture_risk_check(
            symbol=symbol,
            check_type=check_type,
            allowed=allowed,
            position_size=position_size,
            risk_amount=risk_amount,
            **kwargs
        )
    
    def capture_bot_state(self, bot_state: Dict[str, Any]):
        """Capture current bot state."""
        self.data_capture.capture_state(
            account_balance=bot_state.get('account_balance', 0.0),
            equity=bot_state.get('equity', 0.0),
            free_balance=bot_state.get('free_balance', 0.0),
            used_balance=bot_state.get('used_balance', 0.0),
            unrealized_pnl=bot_state.get('unrealized_pnl', 0.0),
            realized_pnl=bot_state.get('realized_pnl', 0.0),
            num_positions=bot_state.get('num_positions', 0),
            num_orders=bot_state.get('num_orders', 0),
            positions=bot_state.get('positions', []),
            orders=bot_state.get('orders', [])
        )
    
    def stop(self):
        """Stop the mock runner."""
        self.running = False
        logger.info("MockLiveRunner stopped")
    
    def export_results(self, filepath: Optional[str] = None) -> str:
        """Export captured results."""
        return self.data_capture.export_to_json(filepath)
    
    def get_capture(self) -> LiveDataCapture:
        """Get the data capture instance."""
        return self.data_capture
