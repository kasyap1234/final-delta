"""Hedge manager module for comprehensive hedge management.

This module provides the HedgeManager class for monitoring positions,
detecting hedge triggers, managing position groups, and executing
hedge strategies with profit taking and re-hedging logic.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from .position_group import (
    PositionGroup, 
    OriginalPosition, 
    HedgePosition, 
    HedgeStatus
)
from .hedge_executor import (
    HedgeExecutor, 
    HedgeRequest, 
    HedgeExecutionResult,
    HedgeExecutorConfig
)
from ..correlation.correlation_calculator import CorrelationCalculator
from ..execution.order_executor import OrderExecutor

logger = logging.getLogger(__name__)


@dataclass
class HedgeManagerConfig:
    """Configuration for HedgeManager."""
    # Hedge trigger settings
    hedge_trigger_threshold: float = 0.5  # 50% of SL distance
    
    # Profit taking settings
    profit_target_ratio: float = 2.0  # 2:1 R:R
    
    # Re-hedging settings
    enable_rehedging: bool = True
    max_hedges_per_position: int = 5
    
    # Position group settings
    auto_close_on_breakeven: bool = True
    
    # Execution settings
    hedge_executor_config: Optional[HedgeExecutorConfig] = None


@dataclass
class HedgeTriggerResult:
    """Result of hedge trigger check."""
    should_hedge: bool
    trigger_level: int
    current_loss_pct: float
    loss_amount: float
    trigger_threshold: float
    message: str


@dataclass
class HedgeCloseResult:
    """Result of closing a hedge position."""
    success: bool
    hedge_id: str
    realized_pnl: float
    close_price: Optional[float] = None
    error_message: Optional[str] = None


class HedgeManager:
    """Manages hedge positions for the trading bot.
    
    This class handles:
    - Monitoring original positions for hedge triggers
    - Managing position groups (original + hedges)
    - Opening hedge positions when triggers are hit
    - Taking profit on hedges at 2:1 R:R
    - Re-hedging if original loss continues
    - Closing all positions when total P&L >= 0
    
    Attributes:
        hedge_executor: HedgeExecutor for executing hedge positions
        correlation_calc: CorrelationCalculator for finding hedge assets
        config: HedgeManagerConfig with management parameters
        position_groups: Dictionary of position groups (group_id -> PositionGroup)
        _hedge_callbacks: Callbacks for hedge events
    """
    
    def __init__(
        self,
        hedge_executor: HedgeExecutor,
        correlation_calc: CorrelationCalculator,
        config: Optional[HedgeManagerConfig] = None
    ):
        """Initialize the hedge manager.
        
        Args:
            hedge_executor: HedgeExecutor for executing hedges
            correlation_calc: CorrelationCalculator for hedge asset selection
            config: Optional configuration (uses defaults if not provided)
        """
        self.hedge_executor = hedge_executor
        self.correlation_calc = correlation_calc
        self.config = config or HedgeManagerConfig()
        
        # Position groups storage
        self.position_groups: Dict[str, PositionGroup] = {}
        
        # Callbacks for hedge events
        self._hedge_callbacks: Dict[str, List[Callable]] = {
            'on_hedge_opened': [],
            'on_hedge_closed': [],
            'on_group_closed': [],
            'on_trigger_detected': []
        }
        
        # Running state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"HedgeManager initialized with trigger_threshold={self.config.hedge_trigger_threshold}, "
            f"profit_ratio={self.config.profit_target_ratio}"
        )
    
    def register_position(
        self,
        position_data: Dict[str, Any]
    ) -> PositionGroup:
        """Register a new original position for hedge monitoring.
        
        Args:
            position_data: Dictionary with position details:
                - id: Position ID
                - symbol: Trading pair
                - side: 'long' or 'short'
                - size: Position size
                - entry_price: Entry price
                - stop_loss: Stop loss price
                - current_price: Current market price
                - take_profit: Optional take profit price
                
        Returns:
            PositionGroup for the registered position
        """
        # Create OriginalPosition
        original = OriginalPosition(
            id=position_data['id'],
            symbol=position_data['symbol'],
            side=position_data['side'],
            size=position_data['size'],
            entry_price=position_data['entry_price'],
            stop_loss=position_data['stop_loss'],
            take_profit=position_data.get('take_profit'),
            current_price=position_data.get('current_price', position_data['entry_price'])
        )
        
        # Create position group
        group = PositionGroup(
            original_position=original,
            target_profit_ratio=self.config.profit_target_ratio,
            hedge_trigger_threshold=self.config.hedge_trigger_threshold
        )
        
        self.position_groups[group.group_id] = group
        
        logger.info(
            f"Registered position {original.id} ({original.symbol}) for hedge monitoring"
        )
        
        return group
    
    def check_hedge_trigger(
        self,
        position: Dict[str, Any]
    ) -> HedgeTriggerResult:
        """Check if a hedge should be triggered for a position.
        
        Args:
            position: Position data dictionary with:
                - id: Position ID
                - symbol: Trading pair
                - side: 'long' or 'short'
                - size: Position size
                - entry_price: Entry price
                - stop_loss: Stop loss price
                - current_price: Current market price
                - unrealized_pnl: Current unrealized P&L
                
        Returns:
            HedgeTriggerResult with trigger analysis
        """
        # Get or create position group
        group_id = position['id']
        if group_id in self.position_groups:
            group = self.position_groups[group_id]
            # Update current price
            if 'current_price' in position:
                group.original.current_price = position['current_price']
                group.original.update_pnl(position['current_price'])
        else:
            # Create temporary group for analysis
            group = self.register_position(position)
        
        # Calculate loss percentage of SL distance
        loss_pct = group.original.get_loss_percentage_of_sl()
        sl_distance = group.original.get_stop_loss_distance()
        current_loss = abs(group.original.unrealized_pnl)
        
        # Check if we should trigger
        should_trigger = group.should_trigger_hedge()
        trigger_level = group.get_hedge_trigger_level()
        
        # Check max hedges limit
        if should_trigger and group.original.hedge_count >= self.config.max_hedges_per_position:
            should_trigger = False
            message = f"Max hedges ({self.config.max_hedges_per_position}) reached"
        elif should_trigger:
            message = f"Trigger level {trigger_level} reached ({loss_pct*100:.1f}% of SL distance)"
        else:
            next_trigger = group.next_hedge_trigger_level
            message = f"Loss at {loss_pct*100:.1f}% of SL, next trigger at {next_trigger*100:.0f}%"
        
        result = HedgeTriggerResult(
            should_hedge=should_trigger,
            trigger_level=trigger_level,
            current_loss_pct=loss_pct,
            loss_amount=current_loss,
            trigger_threshold=group.next_hedge_trigger_level,
            message=message
        )
        
        if should_trigger:
            logger.info(f"Hedge trigger detected for {position['id']}: {message}")
            self._notify_callbacks('on_trigger_detected', group, result)
        
        return result
    
    async def open_hedge(
        self,
        original_position: Dict[str, Any],
        correlation_calc: Optional[CorrelationCalculator] = None
    ) -> Optional[HedgeExecutionResult]:
        """Open a new hedge position for an original position.
        
        Args:
            original_position: Original position data
            correlation_calc: Optional correlation calculator (uses self.correlation_calc if None)
            
        Returns:
            HedgeExecutionResult if hedge was opened, None otherwise
        """
        corr_calc = correlation_calc or self.correlation_calc
        
        # Get position group
        group_id = original_position['id']
        if group_id not in self.position_groups:
            logger.error(f"Position group {group_id} not found")
            return None
        
        group = self.position_groups[group_id]
        
        # Check max hedges
        if group.original.hedge_count >= self.config.max_hedges_per_position:
            logger.warning(
                f"Max hedges ({self.config.max_hedges_per_position}) reached for {group_id}"
            )
            return None
        
        # Find best hedge asset
        hedge_asset = self.hedge_executor.find_hedge_asset(
            group.original.symbol,
            corr_calc
        )
        
        if not hedge_asset:
            logger.error(f"Could not find hedge asset for {group.original.symbol}")
            return None
        
        hedge_symbol = hedge_asset['symbol']
        
        # Create hedge request
        request = HedgeRequest(
            original_symbol=group.original.symbol,
            original_side=group.original.side,
            original_size=group.original.size,
            original_entry_price=group.original.entry_price,
            original_stop_loss=group.original.stop_loss,
            current_price=original_position.get('current_price', group.original.current_price),
            hedge_symbol=hedge_symbol,
            num_chunks=self.hedge_executor.config.num_chunks
        )
        
        # Execute hedge
        logger.info(
            f"Opening hedge #{group.original.hedge_count + 1} for {group_id}: "
            f"{hedge_symbol} (correlation: {hedge_asset['correlation']:.3f})"
        )
        
        result = await self.hedge_executor.execute_hedge_chunks(request)
        
        if result.success:
            # Create HedgePosition
            hedge = HedgePosition(
                id=result.hedge_id,
                symbol=result.symbol,
                side=result.side,
                size=result.filled_size,
                entry_price=result.average_price or request.current_price,
                current_price=request.current_price,
                opened_at=datetime.utcnow(),
                chunk_orders=[c.order_id for c in result.chunks if c.order_id]
            )
            
            # Add to group
            group.add_hedge(hedge)
            
            # Store trigger info
            group.original.last_hedge_trigger_price = group.original.current_price
            group.original.last_hedge_trigger_loss = abs(group.original.unrealized_pnl)
            
            logger.info(
                f"Hedge {result.hedge_id} opened: {result.side} {result.filled_size} "
                f"{result.symbol} @ {result.average_price}"
            )
            
            self._notify_callbacks('on_hedge_opened', group, hedge, result)
        else:
            logger.error(f"Failed to open hedge for {group_id}: {result.error_message}")
        
        return result
    
    async def close_hedge(
        self,
        hedge_position_id: str,
        reason: str = "manual"
    ) -> HedgeCloseResult:
        """Close a specific hedge position.
        
        Args:
            hedge_position_id: ID of the hedge to close
            reason: Reason for closing (manual, profit_target, stop_loss, etc.)
            
        Returns:
            HedgeCloseResult with close details
        """
        # Find hedge in position groups
        hedge: Optional[HedgePosition] = None
        group: Optional[PositionGroup] = None
        
        for g in self.position_groups.values():
            if hedge_position_id in g.hedges:
                hedge = g.hedges[hedge_position_id]
                group = g
                break
        
        if not hedge or not group:
            return HedgeCloseResult(
                success=False,
                hedge_id=hedge_position_id,
                realized_pnl=0.0,
                error_message="Hedge position not found"
            )
        
        if not hedge.is_open:
            return HedgeCloseResult(
                success=False,
                hedge_id=hedge_position_id,
                realized_pnl=hedge.realized_pnl,
                error_message="Hedge already closed"
            )
        
        # Determine close side (opposite of hedge side)
        close_side = 'sell' if hedge.side == 'long' else 'buy'
        
        logger.info(
            f"Closing hedge {hedge_position_id}: {close_side} {hedge.size} {hedge.symbol} "
            f"(reason: {reason})"
        )
        
        try:
            # Place market order to close
            result = await self.hedge_executor.order_executor.place_limit_order(
                symbol=hedge.symbol,
                side=close_side,
                amount=hedge.size,
                price=hedge.current_price,  # Use current price as limit
                post_only=False  # Allow market fill for closing
            )
            
            if result.success:
                # Calculate realized P&L
                if result.price:
                    if hedge.side == 'long':
                        realized_pnl = (result.price - hedge.entry_price) * hedge.size
                    else:
                        realized_pnl = (hedge.entry_price - result.price) * hedge.size
                else:
                    realized_pnl = hedge.unrealized_pnl
                
                # Update hedge
                hedge.realized_pnl = realized_pnl
                hedge.status = "closed"
                hedge.closed_at = datetime.utcnow()
                
                logger.info(
                    f"Hedge {hedge_position_id} closed: P&L=${realized_pnl:.2f}"
                )
                
                self._notify_callbacks('on_hedge_closed', group, hedge, reason)
                
                return HedgeCloseResult(
                    success=True,
                    hedge_id=hedge_position_id,
                    realized_pnl=realized_pnl,
                    close_price=result.price
                )
            else:
                logger.error(f"Failed to close hedge {hedge_position_id}: {result.error_message}")
                return HedgeCloseResult(
                    success=False,
                    hedge_id=hedge_position_id,
                    realized_pnl=0.0,
                    error_message=result.error_message
                )
                
        except Exception as e:
            logger.error(f"Error closing hedge {hedge_position_id}: {e}")
            return HedgeCloseResult(
                success=False,
                hedge_id=hedge_position_id,
                realized_pnl=0.0,
                error_message=str(e)
            )
    
    async def update_hedge_status(
        self,
        position_group: PositionGroup,
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Update hedge status and check for profit targets or re-hedge triggers.
        
        Args:
            position_group: PositionGroup to update
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary with update results:
                - hedges_closed: List of hedge IDs closed for profit
                - new_hedge_opened: Whether a new hedge was opened
                - should_close_all: Whether all positions should close
                - total_pnl: Current total P&L
        """
        result = {
            'hedges_closed': [],
            'new_hedge_opened': False,
            'should_close_all': False,
            'total_pnl': 0.0
        }
        
        # Update prices
        position_group.update_prices(current_prices)
        
        # Calculate total P&L
        total_pnl = position_group.calculate_total_pnl()
        result['total_pnl'] = total_pnl
        
        # Check if we should close all positions
        if position_group.should_close_all():
            result['should_close_all'] = True
            logger.info(
                f"Position group {position_group.group_id} at breakeven/profit: "
                f"P&L=${total_pnl:.2f}"
            )
            return result
        
        # Check hedge profit targets
        profitable_hedges = position_group.check_hedge_profit_targets()
        for hedge in profitable_hedges:
            close_result = await self.close_hedge(hedge.id, reason="profit_target")
            if close_result.success:
                result['hedges_closed'].append(hedge.id)
        
        # Check for re-hedge trigger
        if self.config.enable_rehedging:
            should_trigger = position_group.should_trigger_hedge()
            
            if should_trigger and position_group.original.hedge_count < self.config.max_hedges_per_position:
                logger.info(
                    f"Re-hedge trigger for {position_group.group_id}: "
                    f"level {position_group.get_hedge_trigger_level()}"
                )
                
                hedge_result = await self.open_hedge(
                    original_position={
                        'id': position_group.original.id,
                        'symbol': position_group.original.symbol,
                        'side': position_group.original.side,
                        'size': position_group.original.size,
                        'entry_price': position_group.original.entry_price,
                        'stop_loss': position_group.original.stop_loss,
                        'current_price': position_group.original.current_price
                    }
                )
                
                if hedge_result and hedge_result.success:
                    result['new_hedge_opened'] = True
        
        return result
    
    async def close_all_positions(
        self,
        position_group: PositionGroup,
        reason: str = "breakeven"
    ) -> Dict[str, Any]:
        """Close all positions in a group (original + hedges).
        
        Args:
            position_group: PositionGroup to close
            reason: Reason for closing
            
        Returns:
            Dictionary with close results
        """
        results = {
            'group_id': position_group.group_id,
            'hedges_closed': [],
            'original_closed': False,
            'total_realized_pnl': 0.0,
            'errors': []
        }
        
        logger.info(f"Closing all positions for group {position_group.group_id} (reason: {reason})")
        
        # Close all hedges first
        for hedge_id in list(position_group.hedges.keys()):
            close_result = await self.close_hedge(hedge_id, reason=reason)
            if close_result.success:
                results['hedges_closed'].append(hedge_id)
                results['total_realized_pnl'] += close_result.realized_pnl
            else:
                results['errors'].append(f"Hedge {hedge_id}: {close_result.error_message}")
        
        # Mark group as closed
        summary = position_group.close_group()
        results['summary'] = summary
        
        self._notify_callbacks('on_group_closed', position_group, results)
        
        logger.info(
            f"Position group {position_group.group_id} closed. "
            f"Total P&L: ${results['total_realized_pnl']:.2f}"
        )
        
        return results
    
    def get_position_group(self, position_id: str) -> Optional[PositionGroup]:
        """Get the position group for a given position ID.
        
        Args:
            position_id: ID of the original position
            
        Returns:
            PositionGroup if found, None otherwise
        """
        return self.position_groups.get(position_id)
    
    def calculate_group_pnl(self, position_group: PositionGroup) -> float:
        """Calculate total P&L for a position group.
        
        Args:
            position_group: PositionGroup to calculate P&L for
            
        Returns:
            Total P&L (realized + unrealized)
        """
        return position_group.calculate_total_pnl()
    
    def get_all_groups(self) -> List[PositionGroup]:
        """Get all position groups.
        
        Returns:
            List of all position groups
        """
        return list(self.position_groups.values())
    
    def get_active_groups(self) -> List[PositionGroup]:
        """Get all active position groups (with open positions).
        
        Returns:
            List of active position groups
        """
        return [
            g for g in self.position_groups.values()
            if g.original.is_open or g.get_open_hedges()
        ]
    
    def register_callback(
        self,
        event: str,
        callback: Callable
    ) -> None:
        """Register a callback for hedge events.
        
        Args:
            event: Event name ('on_hedge_opened', 'on_hedge_closed', 'on_group_closed', 'on_trigger_detected')
            callback: Function to call when event occurs
        """
        if event in self._hedge_callbacks:
            self._hedge_callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event type: {event}")
    
    def _notify_callbacks(self, event: str, *args) -> None:
        """Notify all callbacks for an event.
        
        Args:
            event: Event name
            *args: Arguments to pass to callbacks
        """
        for callback in self._hedge_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(*args))
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")
    
    async def start_monitoring(self, price_update_interval: float = 5.0) -> None:
        """Start automatic monitoring of position groups.
        
        Args:
            price_update_interval: Seconds between price updates
        """
        if self._running:
            logger.warning("Hedge monitoring already running")
            return
        
        self._running = True
        logger.info(f"Starting hedge monitoring (interval: {price_update_interval}s)")
        
        while self._running:
            try:
                await self._monitor_step()
                await asyncio.sleep(price_update_interval)
            except Exception as e:
                logger.error(f"Error in hedge monitoring: {e}")
                await asyncio.sleep(price_update_interval)
    
    async def _monitor_step(self) -> None:
        """Execute one monitoring step."""
        # Get current prices for all symbols
        symbols = set()
        for group in self.get_active_groups():
            symbols.add(group.original.symbol)
            for hedge in group.hedges.values():
                symbols.add(hedge.symbol)
        
        # Fetch prices (this would integrate with your price feed)
        # For now, we assume prices are updated externally
        
        # Update all groups
        for group in self.get_active_groups():
            # Price updates should be provided externally
            pass
    
    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Hedge monitoring stopped")
    
    def remove_group(self, group_id: str) -> bool:
        """Remove a position group from tracking.
        
        Args:
            group_id: ID of the group to remove
            
        Returns:
            True if group was removed, False if not found
        """
        if group_id in self.position_groups:
            del self.position_groups[group_id]
            logger.info(f"Removed position group {group_id}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hedge manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_groups = len(self.position_groups)
        active_groups = len(self.get_active_groups())
        total_hedges = sum(g.original.hedge_count for g in self.position_groups.values())
        open_hedges = sum(len(g.get_open_hedges()) for g in self.position_groups.values())
        
        total_pnl = sum(g.calculate_total_pnl() for g in self.position_groups.values())
        
        return {
            'total_groups': total_groups,
            'active_groups': active_groups,
            'total_hedges_opened': total_hedges,
            'open_hedges': open_hedges,
            'total_pnl': total_pnl,
            'monitoring_active': self._running
        }
