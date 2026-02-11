"""Hedge manager module for backtest hedge management.

Refactored to inherit from the live HedgeManager for strategy parity.
Only overrides exchange-specific interactions for backtest mock clients.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from src.hedge.hedge_manager import HedgeManager
from .position_group import PositionGroup
from .hedge_executor import BacktestHedgeExecutor
from src.correlation.correlation_calculator import CorrelationCalculator
from src.backtest.mock.exchange_client import BacktestExchangeClient
from src.backtest.account_state import AccountState
from src.shared.hedge_types import (
    OriginalPosition,
    HedgePosition,
    HedgeManagerConfig,
    HedgeTriggerResult,
    HedgeCloseResult,
)

logger = logging.getLogger(__name__)


class BacktestHedgeManager(HedgeManager):
    """Manages hedge positions for the backtest trading bot.

    Inherits all core hedge management logic from HedgeManager and only
    overrides exchange-specific methods for backtest simulation.

    Strategy parity guarantees:
    - Uses identical trigger detection logic from base class
    - Uses identical position group management from base class
    - Uses identical profit target / re-hedge logic from base class
    - Only exchange interaction differs (mock vs real)

    Attributes:
        hedge_executor: BacktestHedgeExecutor for executing hedge positions
        correlation_calc: CorrelationCalculator for finding hedge assets
        exchange_client: BacktestExchangeClient for order execution
        account_state: AccountState for position tracking
        config: HedgeManagerConfig with management parameters
    """

    def __init__(
        self,
        hedge_executor: BacktestHedgeExecutor,
        correlation_calc: CorrelationCalculator,
        exchange_client: BacktestExchangeClient,
        account_state: AccountState,
        config: Optional[HedgeManagerConfig] = None,
    ):
        self.exchange_client = exchange_client
        self.account_state = account_state

        HedgeManager.__init__(
            self,
            hedge_executor=hedge_executor,
            correlation_calc=correlation_calc,
            config=config,
        )

        logger.info(
            f"BacktestHedgeManager initialized with trigger_threshold={self.config.hedge_trigger_threshold}, "
            f"profit_ratio={self.config.profit_target_ratio} (inherits from HedgeManager)"
        )

    async def close_hedge(
        self, hedge_position_id: str, reason: str = "manual"
    ) -> HedgeCloseResult:
        """Close a specific hedge position using backtest exchange client.

        Overrides HedgeManager.close_hedge to use BacktestExchangeClient.

        Args:
            hedge_position_id: ID of the hedge to close
            reason: Reason for closing

        Returns:
            HedgeCloseResult with close details
        """
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
                error_message="Hedge position not found",
            )

        if not hedge.is_open:
            return HedgeCloseResult(
                success=False,
                hedge_id=hedge_position_id,
                realized_pnl=hedge.realized_pnl,
                error_message="Hedge already closed",
            )

        close_side = "sell" if hedge.side == "long" else "buy"

        logger.info(
            f"Closing hedge {hedge_position_id}: {close_side} {hedge.size} {hedge.symbol} "
            f"(reason: {reason})"
        )

        try:
            order = await self.exchange_client.create_order(
                symbol=hedge.symbol,
                side=close_side,
                order_type="limit",
                amount=hedge.size,
                price=hedge.current_price,
                post_only=False,
            )

            if order.get("id"):
                fill_price = order.get(
                    "average", order.get("price", hedge.current_price)
                )
                if hedge.side == "long":
                    realized_pnl = (fill_price - hedge.entry_price) * hedge.size
                else:
                    realized_pnl = (hedge.entry_price - fill_price) * hedge.size

                hedge.realized_pnl = realized_pnl
                hedge.status = "closed"
                hedge.closed_at = datetime.utcnow()

                logger.info(
                    f"Hedge {hedge_position_id} closed: P&L=${realized_pnl:.2f}"
                )
                self._notify_callbacks("on_hedge_closed", group, hedge, reason)

                return HedgeCloseResult(
                    success=True,
                    hedge_id=hedge_position_id,
                    realized_pnl=realized_pnl,
                    close_price=hedge.current_price,
                )
            else:
                logger.error(
                    f"Failed to close hedge {hedge_position_id}: Order creation failed"
                )
                return HedgeCloseResult(
                    success=False,
                    hedge_id=hedge_position_id,
                    realized_pnl=0.0,
                    error_message="Order creation failed",
                )

        except Exception as e:
            logger.error(f"Error closing hedge {hedge_position_id}: {e}")
            return HedgeCloseResult(
                success=False,
                hedge_id=hedge_position_id,
                realized_pnl=0.0,
                error_message=str(e),
            )

    async def process_candle(
        self, symbol: str, current_price: float, current_time: datetime
    ) -> List[Dict[str, Any]]:
        """Process a new candle for all active position groups.

        Args:
            symbol: The symbol of the current candle
            current_price: Current price from the candle
            current_time: Current timestamp

        Returns:
            List of action results from hedge processing
        """
        results = []
        current_prices = {symbol: current_price}

        for group in self.get_active_groups():
            if group.original.symbol == symbol:
                current_prices[group.original.symbol] = current_price

            for hedge in group.get_open_hedges():
                if hedge.symbol == symbol:
                    current_prices[hedge.symbol] = current_price

        for group in self.get_active_groups():
            update_result = await self.update_hedge_status(group, current_prices)

            if update_result["should_close_all"]:
                close_result = await self.close_all_positions(group, reason="breakeven")
                results.append(
                    {
                        "action": "close_all",
                        "group_id": group.group_id,
                        "result": close_result,
                    }
                )
            elif update_result["hedges_closed"]:
                results.append(
                    {
                        "action": "close_hedges",
                        "group_id": group.group_id,
                        "hedge_ids": update_result["hedges_closed"],
                    }
                )
            elif update_result["new_hedge_opened"]:
                results.append({"action": "open_hedge", "group_id": group.group_id})

            trigger_result = self.check_hedge_trigger(
                {
                    "id": group.original.id,
                    "symbol": group.original.symbol,
                    "side": group.original.side,
                    "size": group.original.size,
                    "entry_price": group.original.entry_price,
                    "stop_loss": group.original.stop_loss,
                    "current_price": current_prices.get(
                        group.original.symbol, group.original.current_price
                    ),
                    "unrealized_pnl": group.original.unrealized_pnl,
                }
            )

            if trigger_result.should_hedge:
                hedge_result = await self.open_hedge(
                    {
                        "id": group.original.id,
                        "symbol": group.original.symbol,
                        "side": group.original.side,
                        "size": group.original.size,
                        "entry_price": group.original.entry_price,
                        "stop_loss": group.original.stop_loss,
                        "current_price": current_prices.get(
                            group.original.symbol, group.original.current_price
                        ),
                    }
                )

                if hedge_result:
                    results.append(
                        {
                            "action": "open_hedge",
                            "group_id": group.group_id,
                            "result": hedge_result.success,
                        }
                    )

        return results
