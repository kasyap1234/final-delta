"""Hedge executor module for backtesting.

This module provides the BacktestHedgeExecutor class for finding optimal hedge assets,
calculating hedge sizes, and executing hedge positions through the mock exchange.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.correlation.correlation_calculator import CorrelationCalculator
from src.backtest.mock.exchange_client import BacktestExchangeClient
from src.backtest.account_state import AccountState

logger = logging.getLogger(__name__)


PRIORITY_ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD", "BTC/USDT", "ETH/USDT", "SOL/USDT"]


@dataclass
class HedgeRequest:
    """Request to open a hedge position."""
    original_symbol: str
    original_side: str
    original_size: float
    original_entry_price: float
    original_stop_loss: float
    current_price: float
    hedge_symbol: Optional[str] = None
    hedge_size: Optional[float] = None
    num_chunks: int = 3
    priority_assets: List[str] = field(default_factory=lambda: PRIORITY_ASSETS.copy())


@dataclass
class HedgeChunk:
    """Represents a single chunk of a hedge position."""
    chunk_id: str
    symbol: str
    side: str
    size: float
    target_price: float
    order_id: Optional[str] = None
    status: str = "pending"
    filled_amount: float = 0.0
    filled_price: Optional[float] = None
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class HedgeExecutionResult:
    """Result of hedge execution."""
    success: bool
    hedge_id: str
    symbol: str
    side: str
    total_size: float
    filled_size: float
    average_price: Optional[float] = None
    chunks: List[HedgeChunk] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class HedgeExecutorConfig:
    """Configuration for HedgeExecutor."""
    num_chunks: int = 3
    chunk_delay_seconds: float = 2.0
    hedge_size_ratio: float = 0.5
    min_correlation: float = 0.5
    priority_assets: List[str] = field(default_factory=lambda: PRIORITY_ASSETS.copy())
    post_only: bool = True
    max_retries_per_chunk: int = 3
    price_adjustment_step: float = 0.001


class BacktestHedgeExecutor:
    """Executes hedge positions in backtest environment.

    This class handles:
    - Finding the best correlated asset for hedging
    - Calculating appropriate hedge size
    - Executing hedge orders through the mock exchange
    - Tracking hedge position lifecycle

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
        config: Optional[HedgeExecutorConfig] = None
    ):
        """Initialize the hedge executor.

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
            f"size_ratio={self.config.hedge_size_ratio}"
        )

    async def execute_hedge_chunks(
        self,
        hedge_request: HedgeRequest
    ) -> HedgeExecutionResult:
        """Execute a hedge position split into multiple chunks.

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
                error_message="No hedge symbol provided"
            )

        hedge_side = 'short' if hedge_request.original_side == 'long' else 'long'

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
                chunk_id=f"{hedge_id}_chunk_{i+1}",
                symbol=hedge_symbol,
                side=hedge_side,
                size=size,
                target_price=hedge_request.current_price
            )
            chunks.append(chunk)
            self._active_chunks[chunk.chunk_id] = chunk

        filled_chunks = 0
        total_filled_size = 0.0
        total_filled_value = 0.0

        for i, chunk in enumerate(chunks):
            try:
                result = await self._execute_chunk(chunk, hedge_request)

                if result.get('success'):
                    chunk.status = "filled"
                    chunk.filled_amount = result.get('filled', 0.0)
                    chunk.filled_price = result.get('price')
                    filled_chunks += 1
                    total_filled_size += chunk.filled_amount
                    if chunk.filled_price:
                        total_filled_value += chunk.filled_amount * chunk.filled_price
                else:
                    chunk.status = "failed"
                    chunk.error_message = result.get('error_message', 'Unknown error')
                    logger.warning(f"Chunk {chunk.chunk_id} failed: {chunk.error_message}")

            except Exception as e:
                logger.error(f"Error executing chunk {chunk.chunk_id}: {e}")
                chunk.status = "failed"
                chunk.error_message = str(e)

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        avg_price = total_filled_value / total_filled_size if total_filled_size > 0 else None

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
            execution_time_ms=execution_time
        )

        logger.info(
            f"Hedge execution {hedge_id} complete: "
            f"{filled_chunks}/{len(chunks)} chunks filled, "
            f"size={total_filled_size}/{hedge_size}"
        )

        return result

    async def _execute_chunk(
        self,
        chunk: HedgeChunk,
        request: HedgeRequest
    ) -> Dict[str, Any]:
        """Execute a single hedge chunk.

        Args:
            chunk: The chunk to execute
            request: Original hedge request

        Returns:
            Order result dictionary
        """
        chunk.status = "open"
        chunk.placed_at = datetime.utcnow()

        side = 'sell' if chunk.side == 'short' else 'buy'

        logger.info(
            f"Placing chunk {chunk.chunk_id}: {side} {chunk.size} {chunk.symbol} "
            f"@ {chunk.target_price}"
        )

        try:
            # Use limit orders with post_only to match live bot behavior
            order = await self.exchange_client.create_order(
                symbol=chunk.symbol,
                side=side,
                order_type='limit',
                amount=chunk.size,
                price=chunk.target_price,
                post_only=self.config.post_only
            )

            chunk.order_id = order.get('id')
            
            # Get fill price from order if available, otherwise use target
            fill_price = order.get('average', order.get('price', chunk.target_price))
            filled_amount = order.get('filled', chunk.size)

            return {
                'success': True,
                'order_id': order.get('id'),
                'filled': filled_amount,
                'price': fill_price
            }

        except Exception as e:
            logger.error(f"Error placing chunk order: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }

    def find_hedge_asset(
        self,
        original_symbol: str,
        correlation_calc: CorrelationCalculator,
        candidates: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the best correlated asset for hedging.

        Priority order:
        1. Priority assets (BTC, ETH, SOL) with highest correlation
        2. Other available assets sorted by correlation

        Args:
            original_symbol: The original position symbol
            correlation_calc: CorrelationCalculator for correlation data
            candidates: Optional list of candidate symbols

        Returns:
            Dictionary with 'symbol' and 'correlation' or None
        """
        priority_candidates = [
            p for p in self.config.priority_assets
            if p != original_symbol
        ]

        best_hedge = None
        best_correlation = 0.0

        for candidate in priority_candidates:
            try:
                corr_result = correlation_calc.get_best_hedge_asset(
                    original_symbol,
                    candidates=[candidate],
                    min_correlation=self.config.min_correlation
                )

                if corr_result and abs(corr_result['correlation']) > best_correlation:
                    best_correlation = abs(corr_result['correlation'])
                    best_hedge = corr_result

            except Exception as e:
                logger.debug(f"Could not get correlation for {candidate}: {e}")
                continue

        if not best_hedge:
            try:
                all_candidates = candidates or correlation_calc.price_history.get_available_symbols()
                all_candidates = [c for c in all_candidates if c != original_symbol]

                best_hedge = correlation_calc.get_best_hedge_asset(
                    original_symbol,
                    candidates=all_candidates,
                    min_correlation=self.config.min_correlation
                )
            except Exception as e:
                logger.warning(f"Could not find hedge asset: {e}")

        if best_hedge:
            logger.info(
                f"Selected hedge asset for {original_symbol}: "
                f"{best_hedge['symbol']} (correlation: {best_hedge['correlation']:.3f})"
            )
        else:
            logger.warning(f"Could not find suitable hedge asset for {original_symbol}")

        return best_hedge

    def calculate_hedge_size(self, original_size: float) -> float:
        """Calculate hedge position size.

        Hedge size is 50% of original position by default.

        Args:
            original_size: Size of the original position

        Returns:
            Calculated hedge size
        """
        return original_size * self.config.hedge_size_ratio

    def calculate_hedge_size_with_limits(
        self,
        original_size: float,
        original_price: float,
        hedge_price: float,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None
    ) -> float:
        """Calculate hedge size with optional min/max limits.

        Args:
            original_size: Original position size
            original_price: Original position price
            hedge_price: Hedge asset price
            min_size: Minimum hedge size
            max_size: Maximum hedge size

        Returns:
            Adjusted hedge size
        """
        base_size = self.calculate_hedge_size(original_size)

        if original_price > 0 and hedge_price > 0:
            price_ratio = original_price / hedge_price
            adjusted_size = base_size * price_ratio
        else:
            adjusted_size = base_size

        if min_size is not None:
            adjusted_size = max(adjusted_size, min_size)
        if max_size is not None:
            adjusted_size = min(adjusted_size, max_size)

        return adjusted_size

    def _calculate_chunk_sizes(
        self,
        total_size: float,
        num_chunks: int
    ) -> List[float]:
        """Split total size into equal chunks.

        Args:
            total_size: Total position size to split
            num_chunks: Number of chunks

        Returns:
            List of chunk sizes
        """
        if num_chunks <= 0:
            return [total_size]

        base_chunk = total_size / num_chunks

        chunk_sizes = []
        for i in range(num_chunks):
            if i == num_chunks - 1:
                remaining = total_size - sum(chunk_sizes)
                chunk_sizes.append(round(remaining, 6))
            else:
                chunk_sizes.append(round(base_chunk, 6))

        return chunk_sizes

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
