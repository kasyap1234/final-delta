"""Hedge management module for cryptocurrency trading bot.

This module provides comprehensive hedge management capabilities including:
- Position group tracking (original + hedge positions)
- Hedge trigger detection based on stop-loss distance
- Multi-chunk hedge execution with post-only limit orders
- Profit taking at 2:1 risk:reward
- Re-hedging when losses continue
- Automatic close-all when total P&L >= 0

Example Usage:
    ```python
    from hedge import HedgeManager, PositionGroup, HedgeExecutor
    from execution import OrderExecutor
    from correlation import CorrelationCalculator
    
    # Initialize components
    hedge_executor = HedgeExecutor(order_executor)
    hedge_manager = HedgeManager(hedge_executor, correlation_calc)
    
    # Register original position
    position = {
        'id': 'pos_1',
        'symbol': 'BTC/USD',
        'side': 'long',
        'size': 0.1,
        'entry_price': 50000,
        'stop_loss': 49000,
        'current_price': 50000
    }
    group = hedge_manager.register_position(position)
    
    # Check if hedge needed
    if hedge_manager.check_hedge_trigger(position).should_hedge:
        hedge_result = await hedge_manager.open_hedge(position)
    
    # Update and monitor
    update_result = await hedge_manager.update_hedge_status(group, current_prices)
    if update_result['should_close_all']:
        await hedge_manager.close_all_positions(group)
    ```
"""

from .position_group import (
    PositionGroup,
    OriginalPosition,
    HedgePosition,
    HedgeStatus,
)

from .hedge_executor import (
    HedgeExecutor,
    HedgeRequest,
    HedgeChunk,
    HedgeExecutionResult,
    HedgeExecutorConfig,
    PRIORITY_ASSETS,
)

from .hedge_manager import (
    HedgeManager,
    HedgeManagerConfig,
    HedgeTriggerResult,
    HedgeCloseResult,
)

__all__ = [
    # Position Group
    'PositionGroup',
    'OriginalPosition',
    'HedgePosition',
    'HedgeStatus',
    
    # Hedge Executor
    'HedgeExecutor',
    'HedgeRequest',
    'HedgeChunk',
    'HedgeExecutionResult',
    'HedgeExecutorConfig',
    'PRIORITY_ASSETS',
    
    # Hedge Manager
    'HedgeManager',
    'HedgeManagerConfig',
    'HedgeTriggerResult',
    'HedgeCloseResult',
]

__version__ = '1.0.0'
