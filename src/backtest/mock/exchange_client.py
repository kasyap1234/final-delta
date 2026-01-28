"""
Exchange client module for backtesting.

This module provides a mock exchange client that serves historical data
and simulates order execution.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from src.data.data_cache import OHLCV, PriceData
from src.backtest.mock.order_simulator import BacktestOrderSimulator, Order, OHLCV as BacktestOHLCV
from src.backtest.account_state import AccountState, Balance

logger = logging.getLogger(__name__)


class BacktestExchangeClient:
    """
    Mock exchange client that serves historical data and simulates order execution.
    
    This class implements the same interface as ExchangeClient but:
    - Returns historical OHLCV data instead of fetching from exchange
    - Simulates order fills based on historical price action
    - Tracks simulated account balance and positions
    - Does not make any network calls
    """
    
    def __init__(
        self,
        historical_data: Dict[str, List[OHLCV]],
        order_simulator: BacktestOrderSimulator,
        account_state: AccountState
    ):
        """
        Initialize the backtest exchange client.
        
        Args:
            historical_data: Dictionary mapping symbols to OHLCV data
            order_simulator: Order simulator for order execution
            account_state: Account state tracker
        """
        self.historical_data = historical_data
        self.order_simulator = order_simulator
        self.account_state = account_state
        self.current_time: Optional[datetime] = None
        self._order_counter = 0
        
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
        return self.account_state.get_positions(symbol)
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an order and submit to the order simulator.
        
        The order is not immediately filled. The order simulator will
        determine if/when it fills based on historical price action.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            order_type: Order type ('limit', 'market', etc.)
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional order parameters
            
        Returns:
            Order dictionary
        """
        if order_type == 'limit' and price is None:
            raise ValueError("Price is required for limit orders")
        
        # Generate order ID
        self._order_counter += 1
        order_id = f"order_{self._order_counter}_{int(self.current_time.timestamp()) if self.current_time else 0}"
        
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price or 0.0,
            timestamp=self.current_time or datetime.now()
        )
        
        # Submit to simulator
        self.order_simulator.submit_order(order)
        
        logger.debug(
            f"Order created: {order_id} {side} {amount} {symbol} @ {price}"
        )
        
        return order.to_dict()
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Cancel an order in the simulator.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol
            
        Returns:
            Cancelled order dictionary
        """
        order = self.order_simulator.cancel_order(order_id)
        
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        
        logger.debug(f"Order cancelled: {order_id}")
        
        return order.to_dict()
    
    async def fetch_order(
        self,
        order_id: str,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get order status from simulator.
        
        Args:
            order_id: Order identifier
            symbol: Trading pair symbol
            
        Returns:
            Order dictionary
        """
        order = self.order_simulator.get_order(order_id)
        
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        
        return order.to_dict()
    
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of order dictionaries
        """
        orders = self.order_simulator.get_open_orders(symbol)
        return [o.to_dict() for o in orders]
    
    async def close(self) -> None:
        """Cleanup (no-op for backtest)."""
        logger.info("BacktestExchangeClient closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'symbols': len(self.historical_data),
            'total_orders': self._order_counter
        }
    
    def reset(self) -> None:
        """Reset the exchange client."""
        self.current_time = None
        self._order_counter = 0
        logger.info("BacktestExchangeClient reset")
