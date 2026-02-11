"""Hedge executor module for backtesting.

Refactored to inherit from the live HedgeExecutor for strategy parity.
Only overrides exchange-specific interactions for backtest mock client.

This ensures complete parity between live and backtest hedge behavior,
preventing strategy drift as noted in AGENTS.md.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.hedge.hedge_executor import HedgeExecutor
from src.correlation.correlation_calculator import CorrelationCalculator
from src.backtest.mock.exchange_client import BacktestExchangeClient
from src.backtest.account_state import AccountState
from src.shared.hedge_types import (
    HedgeRequest,
    HedgeChunk,
    HedgeExecutionResult,
    HedgeExecutorConfig,
)
from src.execution.order_executor import OrderResult

logger = logging.getLogger(__name__)


class BacktestHedgeExecutor(HedgeExecutor):
    """Executes hedge positions in backtest environment.

    Inherits all core logic from HedgeExecutor and only overrides
    exchange-specific methods to use BacktestExchangeClient instead
    of live OrderExecutor.

    This ensures complete parity between live and backtest hedge behavior.

    Strategy parity guarantees:
    - Uses identical find_hedge_asset() logic from base class
    - Uses identical calculate_hedge_size() logic from base class
    - Uses identical chunk calculation from base class
    - Only exchange interaction differs (mock vs real)

    Attributes:
        exchange_client: BacktestExchangeClient for placing orders
        account_state: AccountState for tracking positions
        config: HedgeExecutorConfig with execution parameters
        _active_chunks: Dictionary tracking active chunk executions
    """

    def __init__(
        self,
        exchange_client: BacktestExchangeClient,
        account_state: AccountState,
        config: Optional[HedgeExecutorConfig] = None,
    ):
        """Initialize the backtest hedge executor.

        Args:
            exchange_client: BacktestExchangeClient for placing orders
            account_state: AccountState for tracking positions
            config: Optional configuration (uses defaults if not provided)
        """
        self.exchange_client = exchange_client
        self.account_state = account_state
        self.config = config or HedgeExecutorConfig()
        self._active_chunks: Dict[str, HedgeChunk] = {}

        logger.info(
            f"BacktestHedgeExecutor initialized with {self.config.num_chunks} chunks, "
            f"size_ratio={self.config.hedge_size_ratio} (inherits from HedgeExecutor)"
        )

    async def execute_hedge_chunks(
        self, hedge_request: HedgeRequest
    ) -> HedgeExecutionResult:
        """Execute a hedge position split into multiple chunks.

        Uses the same logic as HedgeExecutor.execute_hedge_chunks but
        with backtest-specific chunk execution.

        Args:
            hedge_request: Hedge request with position details

        Returns:
            HedgeExecutionResult with execution details
        """
        start_time = datetime.utcnow()
        hedge_id = f"hedge_{hedge_request.original_symbol.replace('/', '_')}_{int(start_time.timestamp())}"

        hedge_symbol = hedge_request.hedge_symbol
        if not hedge_symbol:
            return HedgeExecutionResult(
                success=False,
                hedge_id=hedge_id,
                symbol="",
                side="",
                total_size=0,
                filled_size=0,
                error_message="No hedge symbol provided",
            )

        hedge_side = "short" if hedge_request.original_side == "long" else "long"

        hedge_size = hedge_request.hedge_size
        if not hedge_size:
            hedge_size = self.calculate_hedge_size(hedge_request.original_size)

        chunk_sizes = self._calculate_chunk_sizes(hedge_size, self.config.num_chunks)

        logger.info(
            f"Executing hedge {hedge_id}: {hedge_side} {hedge_size} {hedge_symbol} "
            f"in {len(chunk_sizes)} chunks"
        )

        chunks: List[HedgeChunk] = []
        for i, size in enumerate(chunk_sizes):
            chunk = HedgeChunk(
                chunk_id=f"{hedge_id}_chunk_{i + 1}",
                symbol=hedge_symbol,
                side=hedge_side,
                size=size,
                target_price=hedge_request.current_price,
            )
            chunks.append(chunk)
            self._active_chunks[chunk.chunk_id] = chunk

        filled_chunks = 0
        total_filled_size = 0.0
        total_filled_value = 0.0

        for i, chunk in enumerate(chunks):
            try:
                result = await self._execute_chunk(chunk, hedge_request)

                if result.success:
                    chunk.status = "filled"
                    chunk.filled_amount = result.filled
                    chunk.filled_price = result.price
                    filled_chunks += 1
                    total_filled_size += chunk.filled_amount
                    if chunk.filled_price:
                        total_filled_value += chunk.filled_amount * chunk.filled_price
                else:
                    chunk.status = "failed"
                    chunk.error_message = result.error_message
                    logger.warning(
                        f"Chunk {chunk.chunk_id} failed: {result.error_message}"
                    )

            except Exception as e:
                logger.error(f"Error executing chunk {chunk.chunk_id}: {e}")
                chunk.status = "failed"
                chunk.error_message = str(e)

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        avg_price = (
            total_filled_value / total_filled_size if total_filled_size > 0 else None
        )

        success = filled_chunks > 0

        for chunk in chunks:
            self._active_chunks.pop(chunk.chunk_id, None)

        result = HedgeExecutionResult(
            success=success,
            hedge_id=hedge_id,
            symbol=hedge_symbol,
            side=hedge_side,
            total_size=hedge_size,
            filled_size=total_filled_size,
            average_price=avg_price,
            chunks=chunks,
            execution_time_ms=execution_time,
        )

        logger.info(
            f"Hedge execution {hedge_id} complete: "
            f"{filled_chunks}/{len(chunks)} chunks filled, "
            f"size={total_filled_size}/{hedge_size}"
        )

        return result

    async def _execute_chunk(
        self, chunk: HedgeChunk, request: HedgeRequest
    ) -> OrderResult:
        """Execute a single hedge chunk using backtest exchange client.

        Overrides HedgeExecutor._execute_chunk to use BacktestExchangeClient.

        Args:
            chunk: The chunk to execute
            request: Original hedge request

        Returns:
            OrderResult from the order execution
        """
        chunk.status = "open"
        chunk.placed_at = datetime.utcnow()

        side = "sell" if chunk.side == "short" else "buy"

        logger.info(
            f"Placing chunk {chunk.chunk_id}: {side} {chunk.size} {chunk.symbol} "
            f"@ {chunk.target_price}"
        )

        try:
            order = await self.exchange_client.create_order(
                symbol=chunk.symbol,
                side=side,
                order_type="limit",
                amount=chunk.size,
                price=chunk.target_price,
                post_only=self.config.post_only,
            )

            fill_price = order.get("average", order.get("price", chunk.target_price))
            filled_amount = order.get("filled", chunk.size)
            order_id = order.get("id")

            if order_id:
                chunk.order_id = order_id

            result = OrderResult(
                success=True,
                order_id=order_id,
                symbol=chunk.symbol,
                side=side,
                amount=chunk.size,
                price=fill_price,
                filled=filled_amount,
                error_message=None,
            )

            chunk.status = "filled"
            chunk.filled_amount = filled_amount
            chunk.filled_price = fill_price

            return result

        except Exception as e:
            logger.error(f"Error placing chunk order: {e}")
            chunk.status = "failed"
            chunk.error_message = str(e)

            return OrderResult(
                success=False,
                order_id=None,
                symbol=chunk.symbol,
                side=side,
                amount=chunk.size,
                price=chunk.target_price,
                filled=0.0,
                error_message=str(e),
            )

    def get_chunk_status(self, chunk_id: str) -> Optional[HedgeChunk]:
        """Get the status of a specific chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            HedgeChunk if found, None otherwise
        """
        return self._active_chunks.get(chunk_id)

    def get_all_active_chunks(self) -> List[HedgeChunk]:
        """Get all currently active chunks.

        Returns:
            List of active hedge chunks
        """
        return list(self._active_chunks.values())
