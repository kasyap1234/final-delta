"""
Exchange client module for backtesting.

This module provides a mock exchange client that serves historical data
and simulates order execution with realistic behavior matching live trading.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from src.data.data_cache import OHLCV, PriceData
from src.backtest.mock.order_simulator import (
    BacktestOrderSimulator, Order, OHLCV as BacktestOHLCV,
    OrderStatus, SimulatorConfig
)
from src.backtest.mock.price_impact_model import PriceImpactModel
from src.backtest.account_state import AccountState, Balance

logger = logging.getLogger(__name__)


class BacktestExchangeClient:
    """
    Mock exchange client that serves historical data and simulates order execution.
    
    This class implements the same interface as ExchangeClient but:
    - Returns historical OHLCV data instead of fetching from exchange
    - Simulates order fills based on historical price action
    - Supports post-only order rejection simulation
    - Implements retry logic with exponential backoff
    - Supports partial fills based on order book depth
    - Tracks simulated account balance and positions
    - Handles order amendments/cancellations like live exchange
    - Does not make any network calls
    
    Example:
        ```python
        client = BacktestExchangeClient(
            historical_data=data,
            order_simulator=simulator,
            account_state=account
        )
        
        # Create order
        order = await client.create_order(
            symbol='BTC/USD',
            type='limit',
            side='buy',
            amount=0.1,
            price=50000,
            params={'postOnly': True}
        )
        
        # Poll for status like live exchange
        status = await client.fetch_order(order['id'], 'BTC/USD')
        ```
    """
    
    def __init__(
        self,
        historical_data: Dict[str, List[OHLCV]],
        order_simulator: BacktestOrderSimulator,
        account_state: AccountState,
        enable_polling: bool = True,
        poll_interval: float = 1.0
    ):
        """
        Initialize the backtest exchange client.
        
        Args:
            historical_data: Dictionary mapping symbols to OHLCV data
            order_simulator: Order simulator for order execution
            account_state: Account state tracker
            enable_polling: Whether to simulate order status polling
            poll_interval: Simulated poll interval in seconds
        """
        self.historical_data = historical_data
        self.order_simulator = order_simulator
        self.account_state = account_state
        self.enable_polling = enable_polling
        self.poll_interval = poll_interval
        
        self.current_time: Optional[datetime] = None
        self._order_counter = 0
        self._pending_cancellations: Dict[str, bool] = {}
        self._pending_amendments: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            'orders_created': 0,
            'orders_cancelled': 0,
            'orders_amended': 0,
            'poll_requests': 0,
            'api_calls': 0,
        }
        
        logger.info("BacktestExchangeClient initialized")
    
    def set_current_time(self, current_time: datetime) -> None:
        """
        Set the current simulation time.
        
        Args:
            current_time: Current simulation time
        """
        self.current_time = current_time
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200
    ) -> List[List[float]]:
        """
        Return historical OHLCV data up to current_time.
        
        The data is sliced from the pre-loaded historical dataset,
        returning only candles that have occurred up to the current
        simulation time.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string
            limit: Maximum number of candles to return
            
        Returns:
            List of OHLCV arrays in CCXT format
        """
        self._stats['api_calls'] += 1
        data = self.historical_data.get(symbol, [])
        
        # Filter data up to current_time and by timeframe
        if self.current_time:
            filtered = [
                c for c in data
                if c.timestamp <= self.current_time and c.timeframe == timeframe
            ]
        else:
            filtered = [c for c in data if c.timeframe == timeframe]
        
        # Return last N candles in CCXT format
        candles = filtered[-limit:] if filtered else []
        
        # Convert to CCXT format: [timestamp, open, high, low, close, volume]
        return [
            [
                int(c.timestamp.timestamp() * 1000),  # milliseconds
                float(c.open),
                float(c.high),
                float(c.low),
                float(c.close),
                float(c.volume)
            ]
            for c in candles
        ]
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Return ticker data for the current time.
        
        Uses the latest candle's close price as the current price.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data dictionary
        """
        self._stats['api_calls'] += 1
        data = self.historical_data.get(symbol, [])
        
        # Filter data up to current_time
        if self.current_time:
            filtered = [
                c for c in data
                if c.timestamp <= self.current_time
            ]
        else:
            filtered = data
        
        if not filtered:
            raise ValueError(f"No data available for symbol: {symbol}")
        
        latest = filtered[-1]
        close_price = float(latest.close)
        
        # Simulate bid/ask spread
        bid = close_price * 0.9999
        ask = close_price * 1.0001
        
        return {
            'symbol': symbol,
            'last': close_price,
            'bid': bid,
            'ask': ask,
            'high': float(latest.high),
            'low': float(latest.low),
            'volume': float(latest.volume),
            'timestamp': int(latest.timestamp.timestamp() * 1000),
            'datetime': latest.timestamp.isoformat()
        }
    
    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Return the simulated account balance.
        
        Returns:
            Balance data dictionary
        """
        self._stats['api_calls'] += 1
        balance = self.account_state.get_balance()
        
        return {
            'total': balance.total,
            'free': balance.free,
            'used': balance.used,
            'currency': balance.currency
        }
    
    async def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return simulated open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of position dictionaries
        """
        self._stats['api_calls'] += 1
        return self.account_state.get_positions(symbol)
    
    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an order and submit to the order simulator.
        
        The order is not immediately filled. The order simulator will
        determine if/when it fills based on historical price action.
        
        Supports post-only orders and retry logic like live trading.
        
        Args:
            symbol: Trading pair symbol
            type: Order type ('limit', 'market', 'post_only_limit')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional order parameters including:
                - postOnly: Whether order is post-only
                - timeInForce: 'GTC', 'IOC', 'FOK'
                - clientOrderId: Client order ID
            
        Returns:
            Order dictionary in CCXT format
        """
        self._stats['api_calls'] += 1
        params = params or {}
        
        if type == 'limit' and price is None:
            raise ValueError("Price is required for limit orders")
        
        # Extract parameters
        post_only = params.get('postOnly', False) or type == 'post_only_limit'
        time_in_force = params.get('timeInForce', 'GTC')
        client_order_id = params.get('clientOrderId')
        
        # Create order through simulator
        order = self.order_simulator.create_order(
            symbol=symbol,
            side=side,
            order_type=type,
            amount=amount,
            price=price,
            post_only=post_only,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            params=params
        )
        
        # Submit to simulator
        self.order_simulator.submit_order(order)
        
        self._order_counter += 1
        self._stats['orders_created'] += 1
        
        logger.debug(
            f"Order created: {order.id} {side} {amount} {symbol} @ {price}, "
            f"type={type}, post_only={post_only}"
        )
        
        # Simulate network delay
        if self.enable_polling:
            await asyncio.sleep(0.01)
        
        return order.to_dict()
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Cancel an order in the simulator.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol
            params: Additional parameters
            
        Returns:
            Cancelled order dictionary
            
        Raises:
            ValueError: If order not found or already filled/cancelled
        """
        self._stats['api_calls'] += 1
        params = params or {}
        
        order = self.order_simulator.get_order(order_id)
        
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        
        # Check if order can be cancelled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in status: {order.status.value}")
        
        # Cancel the order
        cancelled = self.order_simulator.cancel_order(order_id)
        
        if cancelled is None:
            raise ValueError(f"Failed to cancel order: {order_id}")
        
        self._stats['orders_cancelled'] += 1
        
        logger.debug(f"Order cancelled: {order_id}")
        
        # Simulate network delay
        if self.enable_polling:
            await asyncio.sleep(0.01)
        
        return cancelled.to_dict()
    
    async def fetch_order(
        self,
        order_id: str,
        symbol: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get order status from simulator.
        
        Simulates order status polling like live exchange.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol
            params: Additional parameters
            
        Returns:
            Order dictionary in CCXT format
            
        Raises:
            ValueError: If order not found
        """
        self._stats['api_calls'] += 1
        self._stats['poll_requests'] += 1
        
        order = self.order_simulator.get_order(order_id)
        
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        
        # Simulate polling delay
        if self.enable_polling:
            await asyncio.sleep(0.01)
        
        return order.to_dict()
    
    async def fetch_open_orders(
        self,
        symbol: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol filter
            params: Additional parameters
            
        Returns:
            List of order dictionaries
        """
        self._stats['api_calls'] += 1
        orders = self.order_simulator.get_open_orders(symbol)
        return [o.to_dict() for o in orders]
    
    async def edit_order(
        self,
        order_id: str,
        symbol: str,
        type: str,
        side: str,
        amount: Optional[float] = None,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Edit/amend an existing order.
        
        In live trading, not all exchanges support true order modification.
        This implementation cancels the old order and places a new one.
        
        Args:
            order_id: Order ID to edit
            symbol: Trading pair symbol
            type: Order type
            side: Order side
            amount: New amount (optional)
            price: New price (optional)
            params: Additional parameters
            
        Returns:
            Updated order dictionary
        """
        self._stats['api_calls'] += 1
        params = params or {}
        
        # Get current order
        current = self.order_simulator.get_order(order_id)
        if current is None:
            raise ValueError(f"Order not found: {order_id}")
        
        if not current.is_active:
            raise ValueError(f"Cannot edit order in status: {current.status.value}")
        
        logger.info(f"Editing order {order_id}: new_price={price}, new_amount={amount}")
        
        # Cancel existing order
        cancel_result = await self.cancel_order(order_id, symbol)
        
        if not cancel_result:
            raise ValueError(f"Failed to cancel order for editing: {order_id}")
        
        # Create new order with updated parameters
        new_amount = amount if amount is not None else current.remaining
        new_price = price if price is not None else current.price
        
        new_order = await self.create_order(
            symbol=symbol,
            type=type or current.order_type,
            side=side or current.side,
            amount=new_amount,
            price=new_price,
            params={
                'postOnly': current.post_only,
                'timeInForce': current.time_in_force,
                'previousOrderId': order_id
            }
        )
        
        self._stats['orders_amended'] += 1
        
        logger.info(f"Order amended: old={order_id}, new={new_order['id']}")
        
        return new_order
    
    async def retry_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Retry a rejected order with optional price adjustment.
        
        Simulates the retry logic from live OrderExecutor.
        
        Args:
            order_id: Order ID to retry
            new_price: Optional new price
            max_retries: Maximum retry attempts
            
        Returns:
            Updated order dictionary
            
        Raises:
            ValueError: If retry fails
        """
        order = self.order_simulator.get_order(order_id)
        
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        
        if order.status != OrderStatus.REJECTED:
            raise ValueError(f"Cannot retry order in status: {order.status.value}")
        
        if order.retry_count >= max_retries:
            raise ValueError(f"Max retries exceeded for order: {order_id}")
        
        # Calculate new price if not provided
        if new_price is None and order.post_only:
            new_price = self.order_simulator.adjust_price_for_retry(
                order, order.retry_count
            )
        
        # Retry the order
        retried = self.order_simulator.retry_order(order_id, new_price)
        
        if retried is None:
            raise ValueError(f"Failed to retry order: {order_id}")
        
        # Simulate retry delay
        delay = min(
            self.order_simulator.config.retry_delay_base * (2 ** order.retry_count),
            self.order_simulator.config.retry_delay_max
        )
        await asyncio.sleep(delay)
        
        logger.info(f"Order retried: {order_id}, attempt {order.retry_count}")
        
        return retried.to_dict()
    
    async def wait_for_order_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: float = 60.0,
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """
        Wait for an order to be filled (simulated).
        
        Polls the order status until it's filled, cancelled, or times out.
        
        Args:
            order_id: Order ID to wait for
            symbol: Trading pair symbol
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks
            
        Returns:
            Final order status dictionary
        """
        start_time = time.time()
        last_status = None
        
        logger.info(f"Waiting for order {order_id} to fill (timeout: {timeout}s)")
        
        while time.time() - start_time < timeout:
            try:
                order = await self.fetch_order(order_id, symbol)
                
                current_status = order.get('status')
                
                # Log status changes
                if current_status != last_status:
                    filled = order.get('filled', 0)
                    amount = order.get('amount', 0)
                    logger.info(f"Order {order_id} status: {current_status}, filled: {filled}/{amount}")
                    last_status = current_status
                
                # Check if order is complete
                if current_status in ['filled', 'cancelled', 'rejected', 'expired']:
                    return order
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error polling order {order_id}: {e}")
                await asyncio.sleep(poll_interval)
        
        # Timeout reached
        logger.warning(f"Timeout waiting for order {order_id}")
        
        # Get final status
        final = await self.fetch_order(order_id, symbol)
        final['timeout'] = True
        final['error_message'] = f"Timeout after {timeout}s"
        
        return final
    
    async def close(self) -> None:
        """Cleanup (no-op for backtest)."""
        logger.info("BacktestExchangeClient closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()
        stats['current_time'] = self.current_time.isoformat() if self.current_time else None
        stats['symbols'] = len(self.historical_data)
        
        # Add order simulator stats
        stats['simulator'] = self.order_simulator.get_stats()
        
        return stats
    
    def reset(self) -> None:
        """Reset the exchange client."""
        self.current_time = None
        self._order_counter = 0
        self._pending_cancellations.clear()
        self._pending_amendments.clear()
        
        self._stats = {
            'orders_created': 0,
            'orders_cancelled': 0,
            'orders_amended': 0,
            'poll_requests': 0,
            'api_calls': 0,
        }
        
        logger.info("BacktestExchangeClient reset")
