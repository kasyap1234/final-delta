"""
Database Manager for the cryptocurrency trading bot.

This module provides the DatabaseManager class for managing SQLite database
operations including trade journaling, signal logging, position tracking,
order history, market data storage, and performance metrics.
"""

import sqlite3
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Union
from contextlib import contextmanager
from pathlib import Path

from .models import (
    Trade, Signal, Position, Order, Hedge, MarketData,
    Correlation, Performance, BalanceHistory, TradeJournal, SystemLog,
    TradeSide, TradeStatus, SignalType, PositionSide, PositionStatus,
    OrderSide, OrderType, OrderStatus, TimeInForce, HedgeStatus,
    CorrelationType, MetricType, LogLevel
)


logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class DatabaseManager:
    """
    Manages SQLite database operations for the trading bot.
    
    Provides methods for:
    - Database initialization and schema management
    - Trade journaling and tracking
    - Signal logging and outcome tracking
    - Position management
    - Order history
    - Market data storage
    - Correlation calculations
    - Performance metrics
    - Balance history
    """
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        """
        Initialize the DatabaseManager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory_exists()
        self._connection: Optional[sqlite3.Connection] = None
        
    def _ensure_directory_exists(self) -> None:
        """Ensure the database directory exists."""
        directory = os.path.dirname(self.db_path)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Database operation failed: {e}") from e
        finally:
            conn.close()
    
    def _get_cursor(self, conn: sqlite3.Connection) -> sqlite3.Cursor:
        """Get a cursor from connection with proper settings."""
        cursor = conn.cursor()
        return cursor
    
    def initialize_database(self) -> None:
        """
        Initialize the database with all required tables and indexes.
        
        Reads the schema.sql file and executes all SQL statements.
        """
        schema_path = Path(__file__).parent / "schema.sql"
        
        if not schema_path.exists():
            raise DatabaseError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            # Execute schema statements
            cursor.executescript(schema_sql)
            logger.info("Database initialized successfully")
    
    def reset_database(self, confirm: bool = False) -> None:
        """
        Reset the database by dropping all tables and reinitializing.
        
        Args:
            confirm: Must be True to proceed with reset
        
        Raises:
            ValueError: If confirm is not True
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to reset database")
        
        tables = [
            'trades', 'signals', 'positions', 'orders', 'hedges',
            'market_data', 'correlations', 'performance', 'balance_history',
            'trade_journal', 'system_logs'
        ]
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            logger.warning("Database reset - all tables dropped")
        
        self.initialize_database()
    
    # =========================================================================
    # TRADE OPERATIONS
    # =========================================================================
    
    def save_trade(self, trade: Trade) -> str:
        """
        Save a trade record to the database.
        
        Args:
            trade: Trade model instance
            
        Returns:
            The trade_id of the saved trade
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = trade.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, symbol, side, entry_price, exit_price, quantity,
                    entry_time, exit_time, pnl, pnl_percent, fees, status,
                    strategy, signal_ids, position_id, metadata, updated_at
                ) VALUES (
                    :trade_id, :symbol, :side, :entry_price, :exit_price, :quantity,
                    :entry_time, :exit_time, :pnl, :pnl_percent, :fees, :status,
                    :strategy, :signal_ids, :position_id, :metadata, CURRENT_TIMESTAMP
                )
            """, data)
            
            logger.debug(f"Trade saved: {trade.trade_id}")
            return trade.trade_id
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a trade record.
        
        Args:
            trade_id: ID of the trade to update
            updates: Dictionary of fields to update
            
        Returns:
            True if trade was updated, False if not found
        """
        if not updates:
            return False
        
        # Convert enum values to strings
        if 'status' in updates and isinstance(updates['status'], TradeStatus):
            updates['status'] = updates['status'].value
        if 'side' in updates and isinstance(updates['side'], TradeSide):
            updates['side'] = updates['side'].value
        
        # Convert datetime to ISO format
        for key in ['entry_time', 'exit_time']:
            if key in updates and isinstance(updates[key], datetime):
                updates[key] = updates[key].isoformat()
        
        # Convert list to JSON
        if 'signal_ids' in updates and isinstance(updates['signal_ids'], list):
            updates['signal_ids'] = json.dumps(updates['signal_ids'])
        
        # Convert dict to JSON
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])
        
        set_clause = ', '.join([f"{k} = :{k}" for k in updates.keys()])
        updates['trade_id'] = trade_id
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                UPDATE trades 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE trade_id = :trade_id
            """, updates)
            
            if cursor.rowcount > 0:
                logger.debug(f"Trade updated: {trade_id}")
                return True
            return False
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """
        Get a trade by ID.
        
        Args:
            trade_id: The trade ID to look up
            
        Returns:
            Trade instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Trade.from_dict(dict(row))
            return None
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[Union[TradeStatus, str]] = None,
        side: Optional[Union[TradeSide, str]] = None,
        strategy: Optional[str] = None,
        position_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Trade]:
        """
        Query trades with optional filters.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            side: Filter by side
            strategy: Filter by strategy
            position_id: Filter by position ID
            start_time: Filter trades after this time
            end_time: Filter trades before this time
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of Trade instances
        """
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if status:
            status_val = status.value if isinstance(status, TradeStatus) else status
            conditions.append("status = ?")
            params.append(status_val)
        if side:
            side_val = side.value if isinstance(side, TradeSide) else side
            conditions.append("side = ?")
            params.append(side_val)
        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if position_id:
            conditions.append("position_id = ?")
            params.append(position_id)
        if start_time:
            conditions.append("entry_time >= ?")
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            conditions.append("entry_time <= ?")
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM trades 
                WHERE {where_clause}
                ORDER BY entry_time DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            rows = cursor.fetchall()
            return [Trade.from_dict(dict(row)) for row in rows]
    
    def get_trade_stats(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get trade statistics.
        
        Args:
            symbol: Filter by symbol
            start_time: Start of period
            end_time: End of period
            
        Returns:
            Dictionary with trade statistics
        """
        conditions = ["status = 'closed'"]
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time:
            conditions.append("exit_time >= ?")
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            conditions.append("exit_time <= ?")
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        where_clause = " AND ".join(conditions)
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN pnl = 0 THEN 1 ELSE 0 END) as break_even_trades,
                    SUM(pnl) as total_pnl,
                    SUM(fees) as total_fees,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) as gross_loss,
                    AVG(pnl) as average_pnl,
                    MAX(pnl) as largest_win,
                    MIN(pnl) as largest_loss,
                    AVG(pnl_percent) as average_pnl_percent
                FROM trades
                WHERE {where_clause}
            """, params)
            
            row = cursor.fetchone()
            if row:
                stats = dict(row)
                # Calculate derived metrics
                total = stats.get('total_trades', 0) or 0
                wins = stats.get('winning_trades', 0) or 0
                losses = stats.get('losing_trades', 0) or 0
                
                if total > 0:
                    stats['win_rate'] = (wins / total) * 100
                if losses != 0:
                    gross_profit = abs(stats.get('gross_profit', 0) or 0)
                    gross_loss = abs(stats.get('gross_loss', 0) or 0)
                    stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                if wins > 0:
                    stats['average_win'] = (stats.get('gross_profit', 0) or 0) / wins
                if losses > 0:
                    stats['average_loss'] = (stats.get('gross_loss', 0) or 0) / losses
                    
                return stats
            return {}
    
    # =========================================================================
    # SIGNAL OPERATIONS
    # =========================================================================
    
    def save_signal(self, signal: Signal) -> str:
        """
        Save a signal record to the database.
        
        Args:
            signal: Signal model instance
            
        Returns:
            The signal_id of the saved signal
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = signal.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO signals (
                    signal_id, symbol, signal_type, strength, indicators,
                    price_at_signal, volume_at_signal, timeframe, strategy,
                    confidence, executed, trade_id, outcome, pnl, generated_at,
                    executed_at, expires_at, metadata
                ) VALUES (
                    :signal_id, :symbol, :signal_type, :strength, :indicators,
                    :price_at_signal, :volume_at_signal, :timeframe, :strategy,
                    :confidence, :executed, :trade_id, :outcome, :pnl, :generated_at,
                    :executed_at, :expires_at, :metadata
                )
            """, data)
            
            logger.debug(f"Signal saved: {signal.signal_id}")
            return signal.signal_id
    
    def update_signal(self, signal_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a signal record.
        
        Args:
            signal_id: ID of the signal to update
            updates: Dictionary of fields to update
            
        Returns:
            True if signal was updated, False if not found
        """
        if not updates:
            return False
        
        # Convert enum values to strings
        if 'signal_type' in updates and isinstance(updates['signal_type'], SignalType):
            updates['signal_type'] = updates['signal_type'].value
        
        # Convert datetime to ISO format
        for key in ['generated_at', 'executed_at', 'expires_at']:
            if key in updates and isinstance(updates[key], datetime):
                updates[key] = updates[key].isoformat()
        
        # Convert dict to JSON
        if 'indicators' in updates and isinstance(updates['indicators'], dict):
            updates['indicators'] = json.dumps(updates['indicators'])
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])
        
        set_clause = ', '.join([f"{k} = :{k}" for k in updates.keys()])
        updates['signal_id'] = signal_id
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                UPDATE signals 
                SET {set_clause}
                WHERE signal_id = :signal_id
            """, updates)
            
            if cursor.rowcount > 0:
                logger.debug(f"Signal updated: {signal_id}")
                return True
            return False
    
    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """
        Get a signal by ID.
        
        Args:
            signal_id: The signal ID to look up
            
        Returns:
            Signal instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(
                "SELECT * FROM signals WHERE signal_id = ?",
                (signal_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Signal.from_dict(dict(row))
            return None
    
    def get_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[Union[SignalType, str]] = None,
        executed: Optional[bool] = None,
        strategy: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Signal]:
        """
        Query signals with optional filters.
        
        Args:
            symbol: Filter by symbol
            signal_type: Filter by signal type
            executed: Filter by execution status
            strategy: Filter by strategy
            start_time: Filter signals after this time
            end_time: Filter signals before this time
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of Signal instances
        """
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if signal_type:
            type_val = signal_type.value if isinstance(signal_type, SignalType) else signal_type
            conditions.append("signal_type = ?")
            params.append(type_val)
        if executed is not None:
            conditions.append("executed = ?")
            params.append(executed)
        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if start_time:
            conditions.append("generated_at >= ?")
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            conditions.append("generated_at <= ?")
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM signals 
                WHERE {where_clause}
                ORDER BY generated_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            rows = cursor.fetchall()
            return [Signal.from_dict(dict(row)) for row in rows]
    
    def get_signal_performance(
        self,
        strategy: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get signal performance statistics.
        
        Args:
            strategy: Filter by strategy
            start_time: Start of period
            end_time: End of period
            
        Returns:
            Dictionary with signal performance statistics
        """
        conditions = ["executed = 1"]
        params = []
        
        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if start_time:
            conditions.append("generated_at >= ?")
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            conditions.append("generated_at <= ?")
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        where_clause = " AND ".join(conditions)
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN outcome = 'profit' THEN 1 ELSE 0 END) as profitable_signals,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losing_signals,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as average_pnl,
                    AVG(strength) as average_strength,
                    AVG(confidence) as average_confidence
                FROM signals
                WHERE {where_clause}
            """, params)
            
            row = cursor.fetchone()
            if row:
                stats = dict(row)
                total = stats.get('total_signals', 0) or 0
                profitable = stats.get('profitable_signals', 0) or 0
                
                if total > 0:
                    stats['accuracy'] = (profitable / total) * 100
                    
                return stats
            return {}
    
    # =========================================================================
    # POSITION OPERATIONS
    # =========================================================================
    
    def save_position(self, position: Position) -> str:
        """
        Save a position record to the database.
        
        Args:
            position: Position model instance
            
        Returns:
            The position_id of the saved position
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = position.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO positions (
                    position_id, symbol, side, size, entry_price, current_price,
                    mark_price, stop_loss, take_profit, liquidation_price,
                    unrealized_pnl, realized_pnl, margin_used, leverage, status,
                    hedge_id, opened_at, closed_at, last_updated, metadata, updated_at
                ) VALUES (
                    :position_id, :symbol, :side, :size, :entry_price, :current_price,
                    :mark_price, :stop_loss, :take_profit, :liquidation_price,
                    :unrealized_pnl, :realized_pnl, :margin_used, :leverage, :status,
                    :hedge_id, :opened_at, :closed_at, :last_updated, :metadata, CURRENT_TIMESTAMP
                )
            """, data)
            
            logger.debug(f"Position saved: {position.position_id}")
            return position.position_id
    
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a position record.
        
        Args:
            position_id: ID of the position to update
            updates: Dictionary of fields to update
            
        Returns:
            True if position was updated, False if not found
        """
        if not updates:
            return False
        
        # Convert enum values to strings
        if 'status' in updates and isinstance(updates['status'], PositionStatus):
            updates['status'] = updates['status'].value
        if 'side' in updates and isinstance(updates['side'], PositionSide):
            updates['side'] = updates['side'].value
        
        # Convert datetime to ISO format
        for key in ['opened_at', 'closed_at', 'last_updated']:
            if key in updates and isinstance(updates[key], datetime):
                updates[key] = updates[key].isoformat()
        
        # Convert dict to JSON
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])
        
        set_clause = ', '.join([f"{k} = :{k}" for k in updates.keys()])
        updates['position_id'] = position_id
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                UPDATE positions 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE position_id = :position_id
            """, updates)
            
            if cursor.rowcount > 0:
                logger.debug(f"Position updated: {position_id}")
                return True
            return False
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get a position by ID.
        
        Args:
            position_id: The position ID to look up
            
        Returns:
            Position instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(
                "SELECT * FROM positions WHERE position_id = ?",
                (position_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Position.from_dict(dict(row))
            return None
    
    def get_positions(
        self,
        symbol: Optional[str] = None,
        status: Optional[Union[PositionStatus, str]] = None,
        side: Optional[Union[PositionSide, str]] = None,
        hedge_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Position]:
        """
        Query positions with optional filters.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            side: Filter by side
            hedge_id: Filter by hedge ID
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of Position instances
        """
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if status:
            status_val = status.value if isinstance(status, PositionStatus) else status
            conditions.append("status = ?")
            params.append(status_val)
        if side:
            side_val = side.value if isinstance(side, PositionSide) else side
            conditions.append("side = ?")
            params.append(side_val)
        if hedge_id:
            conditions.append("hedge_id = ?")
            params.append(hedge_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM positions 
                WHERE {where_clause}
                ORDER BY opened_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            rows = cursor.fetchall()
            return [Position.from_dict(dict(row)) for row in rows]
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open Position instances
        """
        return self.get_positions(symbol=symbol, status=PositionStatus.OPEN)
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float,
        closed_at: Optional[datetime] = None
    ) -> bool:
        """
        Close a position.
        
        Args:
            position_id: ID of the position to close
            exit_price: Exit price
            realized_pnl: Realized P&L
            closed_at: Close timestamp (defaults to now)
            
        Returns:
            True if position was closed, False if not found
        """
        if closed_at is None:
            closed_at = datetime.now()
        
        updates = {
            'status': PositionStatus.CLOSED.value,
            'current_price': exit_price,
            'realized_pnl': realized_pnl,
            'closed_at': closed_at.isoformat(),
            'unrealized_pnl': 0
        }
        
        return self.update_position(position_id, updates)
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of all positions.
        
        Returns:
            Dictionary with position summary statistics
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            
            # Open positions
            cursor.execute("""
                SELECT 
                    COUNT(*) as count,
                    SUM(unrealized_pnl) as total_unrealized_pnl,
                    SUM(size * entry_price) as total_exposure
                FROM positions
                WHERE status = 'open'
            """)
            open_stats = dict(cursor.fetchone() or {})
            
            # Closed positions today
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cursor.execute("""
                SELECT 
                    COUNT(*) as count,
                    SUM(realized_pnl) as total_realized_pnl
                FROM positions
                WHERE status = 'closed' AND closed_at >= ?
            """, (today.isoformat(),))
            closed_today = dict(cursor.fetchone() or {})
            
            # By side
            cursor.execute("""
                SELECT 
                    side,
                    COUNT(*) as count,
                    SUM(unrealized_pnl) as unrealized_pnl
                FROM positions
                WHERE status = 'open'
                GROUP BY side
            """)
            by_side = [dict(row) for row in cursor.fetchall()]
            
            return {
                'open_positions': open_stats.get('count', 0) or 0,
                'total_unrealized_pnl': open_stats.get('total_unrealized_pnl', 0) or 0,
                'total_exposure': open_stats.get('total_exposure', 0) or 0,
                'closed_today': closed_today.get('count', 0) or 0,
                'realized_pnl_today': closed_today.get('total_realized_pnl', 0) or 0,
                'by_side': by_side
            }
    
    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================
    
    def save_order(self, order: Order) -> str:
        """
        Save an order record to the database.
        
        Args:
            order: Order model instance
            
        Returns:
            The order_id of the saved order
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = order.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, exchange_order_id, symbol, side, order_type,
                    quantity, filled_quantity, remaining_quantity, price,
                    average_fill_price, stop_price, status, time_in_force,
                    fees, fee_currency, trade_id, position_id, signal_id,
                    submitted_at, filled_at, cancelled_at, metadata, updated_at
                ) VALUES (
                    :order_id, :exchange_order_id, :symbol, :side, :order_type,
                    :quantity, :filled_quantity, :remaining_quantity, :price,
                    :average_fill_price, :stop_price, :status, :time_in_force,
                    :fees, :fee_currency, :trade_id, :position_id, :signal_id,
                    :submitted_at, :filled_at, :cancelled_at, :metadata, CURRENT_TIMESTAMP
                )
            """, data)
            
            logger.debug(f"Order saved: {order.order_id}")
            return order.order_id
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an order record.
        
        Args:
            order_id: ID of the order to update
            updates: Dictionary of fields to update
            
        Returns:
            True if order was updated, False if not found
        """
        if not updates:
            return False
        
        # Convert enum values to strings
        if 'status' in updates and isinstance(updates['status'], OrderStatus):
            updates['status'] = updates['status'].value
        if 'side' in updates and isinstance(updates['side'], OrderSide):
            updates['side'] = updates['side'].value
        if 'order_type' in updates and isinstance(updates['order_type'], OrderType):
            updates['order_type'] = updates['order_type'].value
        
        # Convert datetime to ISO format
        for key in ['submitted_at', 'filled_at', 'cancelled_at']:
            if key in updates and isinstance(updates[key], datetime):
                updates[key] = updates[key].isoformat()
        
        # Convert dict to JSON
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])
        
        set_clause = ', '.join([f"{k} = :{k}" for k in updates.keys()])
        updates['order_id'] = order_id
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                UPDATE orders 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE order_id = :order_id
            """, updates)
            
            if cursor.rowcount > 0:
                logger.debug(f"Order updated: {order_id}")
                return True
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: The order ID to look up
            
        Returns:
            Order instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Order.from_dict(dict(row))
            return None
    
    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[Union[OrderStatus, str]] = None,
        side: Optional[Union[OrderSide, str]] = None,
        order_type: Optional[Union[OrderType, str]] = None,
        trade_id: Optional[str] = None,
        position_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Order]:
        """
        Query orders with optional filters.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            side: Filter by side
            order_type: Filter by order type
            trade_id: Filter by trade ID
            position_id: Filter by position ID
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of Order instances
        """
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if status:
            status_val = status.value if isinstance(status, OrderStatus) else status
            conditions.append("status = ?")
            params.append(status_val)
        if side:
            side_val = side.value if isinstance(side, OrderSide) else side
            conditions.append("side = ?")
            params.append(side_val)
        if order_type:
            type_val = order_type.value if isinstance(order_type, OrderType) else order_type
            conditions.append("order_type = ?")
            params.append(type_val)
        if trade_id:
            conditions.append("trade_id = ?")
            params.append(trade_id)
        if position_id:
            conditions.append("position_id = ?")
            params.append(position_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM orders 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            rows = cursor.fetchall()
            return [Order.from_dict(dict(row)) for row in rows]
    
    # =========================================================================
    # HEDGE OPERATIONS
    # =========================================================================
    
    def save_hedge(self, hedge: Hedge) -> str:
        """
        Save a hedge record to the database.
        
        Args:
            hedge: Hedge model instance
            
        Returns:
            The hedge_id of the saved hedge
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = hedge.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO hedges (
                    hedge_id, primary_position_id, hedge_position_id,
                    primary_symbol, hedge_symbol, correlation_at_hedge,
                    hedge_ratio, primary_size, hedge_size, status, pnl,
                    primary_pnl, hedge_pnl, opened_at, closed_at, metadata, updated_at
                ) VALUES (
                    :hedge_id, :primary_position_id, :hedge_position_id,
                    :primary_symbol, :hedge_symbol, :correlation_at_hedge,
                    :hedge_ratio, :primary_size, :hedge_size, :status, :pnl,
                    :primary_pnl, :hedge_pnl, :opened_at, :closed_at, :metadata, CURRENT_TIMESTAMP
                )
            """, data)
            
            logger.debug(f"Hedge saved: {hedge.hedge_id}")
            return hedge.hedge_id
    
    def update_hedge(self, hedge_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a hedge record.
        
        Args:
            hedge_id: ID of the hedge to update
            updates: Dictionary of fields to update
            
        Returns:
            True if hedge was updated, False if not found
        """
        if not updates:
            return False
        
        # Convert enum values to strings
        if 'status' in updates and isinstance(updates['status'], HedgeStatus):
            updates['status'] = updates['status'].value
        
        # Convert datetime to ISO format
        for key in ['opened_at', 'closed_at']:
            if key in updates and isinstance(updates[key], datetime):
                updates[key] = updates[key].isoformat()
        
        # Convert dict to JSON
        if 'metadata' in updates and isinstance(updates['metadata'], dict):
            updates['metadata'] = json.dumps(updates['metadata'])
        
        set_clause = ', '.join([f"{k} = :{k}" for k in updates.keys()])
        updates['hedge_id'] = hedge_id
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                UPDATE hedges 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE hedge_id = :hedge_id
            """, updates)
            
            if cursor.rowcount > 0:
                logger.debug(f"Hedge updated: {hedge_id}")
                return True
            return False
    
    def get_hedge(self, hedge_id: str) -> Optional[Hedge]:
        """
        Get a hedge by ID.
        
        Args:
            hedge_id: The hedge ID to look up
            
        Returns:
            Hedge instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(
                "SELECT * FROM hedges WHERE hedge_id = ?",
                (hedge_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Hedge.from_dict(dict(row))
            return None
    
    def get_hedges(
        self,
        status: Optional[Union[HedgeStatus, str]] = None,
        primary_symbol: Optional[str] = None,
        hedge_symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Hedge]:
        """
        Query hedges with optional filters.
        
        Args:
            status: Filter by status
            primary_symbol: Filter by primary symbol
            hedge_symbol: Filter by hedge symbol
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of Hedge instances
        """
        conditions = []
        params = []
        
        if status:
            status_val = status.value if isinstance(status, HedgeStatus) else status
            conditions.append("status = ?")
            params.append(status_val)
        if primary_symbol:
            conditions.append("primary_symbol = ?")
            params.append(primary_symbol)
        if hedge_symbol:
            conditions.append("hedge_symbol = ?")
            params.append(hedge_symbol)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM hedges 
                WHERE {where_clause}
                ORDER BY opened_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            rows = cursor.fetchall()
            return [Hedge.from_dict(dict(row)) for row in rows]
    
    # =========================================================================
    # MARKET DATA OPERATIONS
    # =========================================================================
    
    def save_market_data(self, market_data: MarketData) -> bool:
        """
        Save market data (OHLCV) to the database.
        
        Args:
            market_data: MarketData model instance
            
        Returns:
            True if saved successfully
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = market_data.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO market_data (
                    symbol, timeframe, timestamp, open, high, low, close,
                    volume, quote_volume, trades_count, taker_buy_volume,
                    taker_buy_quote_volume
                ) VALUES (
                    :symbol, :timeframe, :timestamp, :open, :high, :low, :close,
                    :volume, :quote_volume, :trades_count, :taker_buy_volume,
                    :taker_buy_quote_volume
                )
            """, data)
            
            return True
    
    def save_market_data_batch(self, market_data_list: List[MarketData]) -> int:
        """
        Save multiple market data records in a batch.
        
        Args:
            market_data_list: List of MarketData instances
            
        Returns:
            Number of records saved
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            
            data_tuples = [
                (
                    md.symbol, md.timeframe,
                    md.timestamp.isoformat() if isinstance(md.timestamp, datetime) else md.timestamp,
                    md.open, md.high, md.low, md.close, md.volume,
                    md.quote_volume, md.trades_count, md.taker_buy_volume,
                    md.taker_buy_quote_volume
                )
                for md in market_data_list
            ]
            
            cursor.executemany("""
                INSERT OR REPLACE INTO market_data (
                    symbol, timeframe, timestamp, open, high, low, close,
                    volume, quote_volume, trades_count, taker_buy_volume,
                    taker_buy_quote_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_tuples)
            
            logger.debug(f"Batch saved {len(market_data_list)} market data records")
            return len(market_data_list)
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketData]:
        """
        Get market data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_time: Start of data range
            end_time: End of data range
            
        Returns:
            List of MarketData instances
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute("""
                SELECT * FROM market_data
                WHERE symbol = ? AND timeframe = ?
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (
                symbol, timeframe,
                start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                end_time.isoformat() if isinstance(end_time, datetime) else end_time
            ))
            
            rows = cursor.fetchall()
            return [MarketData.from_dict(dict(row)) for row in rows]
    
    def get_latest_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1
    ) -> List[MarketData]:
        """
        Get the most recent market data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of candles to return
            
        Returns:
            List of MarketData instances (most recent first)
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute("""
                SELECT * FROM market_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, timeframe, limit))
            
            rows = cursor.fetchall()
            return [MarketData.from_dict(dict(row)) for row in rows]
    
    # =========================================================================
    # CORRELATION OPERATIONS
    # =========================================================================
    
    def save_correlation(self, correlation: Correlation) -> bool:
        """
        Save a correlation calculation to the database.
        
        Args:
            correlation: Correlation model instance
            
        Returns:
            True if saved successfully
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = correlation.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO correlations (
                    symbol_a, symbol_b, correlation, correlation_type,
                    timeframe, period, calculated_at, price_data_range_start,
                    price_data_range_end, p_value, metadata
                ) VALUES (
                    :symbol_a, :symbol_b, :correlation, :correlation_type,
                    :timeframe, :period, :calculated_at, :price_data_range_start,
                    :price_data_range_end, :p_value, :metadata
                )
            """, data)
            
            logger.debug(f"Correlation saved: {correlation.symbol_a}/{correlation.symbol_b}")
            return True
    
    def get_correlations(
        self,
        symbol_a: Optional[str] = None,
        symbol_b: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100
    ) -> List[Correlation]:
        """
        Get correlation calculations.
        
        Args:
            symbol_a: Filter by first symbol
            symbol_b: Filter by second symbol
            timeframe: Filter by timeframe
            limit: Maximum number of results
            
        Returns:
            List of Correlation instances
        """
        conditions = []
        params = []
        
        if symbol_a:
            conditions.append("symbol_a = ?")
            params.append(symbol_a)
        if symbol_b:
            conditions.append("symbol_b = ?")
            params.append(symbol_b)
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM correlations
                WHERE {where_clause}
                ORDER BY calculated_at DESC
                LIMIT ?
            """, params + [limit])
            
            rows = cursor.fetchall()
            return [Correlation.from_dict(dict(row)) for row in rows]
    
    def get_latest_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        timeframe: str
    ) -> Optional[Correlation]:
        """
        Get the most recent correlation for a symbol pair.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            timeframe: Timeframe
            
        Returns:
            Correlation instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute("""
                SELECT * FROM correlations
                WHERE symbol_a = ? AND symbol_b = ? AND timeframe = ?
                ORDER BY calculated_at DESC
                LIMIT 1
            """, (symbol_a, symbol_b, timeframe))
            
            row = cursor.fetchone()
            if row:
                return Correlation.from_dict(dict(row))
            return None
    
    # =========================================================================
    # PERFORMANCE OPERATIONS
    # =========================================================================
    
    def save_performance(self, performance: Performance) -> str:
        """
        Save performance metrics to the database.
        
        Args:
            performance: Performance model instance
            
        Returns:
            The metric_id of the saved performance record
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = performance.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO performance (
                    metric_id, metric_type, period_start, period_end,
                    total_trades, winning_trades, losing_trades, total_pnl,
                    gross_profit, gross_loss, win_rate, profit_factor,
                    average_win, average_loss, largest_win, largest_loss,
                    sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_percent,
                    volatility, return_on_investment, return_on_investment_percent,
                    starting_balance, ending_balance, metadata, calculated_at
                ) VALUES (
                    :metric_id, :metric_type, :period_start, :period_end,
                    :total_trades, :winning_trades, :losing_trades, :total_pnl,
                    :gross_profit, :gross_loss, :win_rate, :profit_factor,
                    :average_win, :average_loss, :largest_win, :largest_loss,
                    :sharpe_ratio, :sortino_ratio, :max_drawdown, :max_drawdown_percent,
                    :volatility, :return_on_investment, :return_on_investment_percent,
                    :starting_balance, :ending_balance, :metadata, :calculated_at
                )
            """, data)
            
            logger.debug(f"Performance saved: {performance.metric_id}")
            return performance.metric_id
    
    def get_performance(
        self,
        metric_type: Optional[Union[MetricType, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Performance]:
        """
        Get performance metrics.
        
        Args:
            metric_type: Filter by metric type
            start_time: Filter by period start
            end_time: Filter by period end
            limit: Maximum number of results
            
        Returns:
            List of Performance instances
        """
        conditions = []
        params = []
        
        if metric_type:
            type_val = metric_type.value if isinstance(metric_type, MetricType) else metric_type
            conditions.append("metric_type = ?")
            params.append(type_val)
        if start_time:
            conditions.append("period_start >= ?")
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            conditions.append("period_end <= ?")
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM performance
                WHERE {where_clause}
                ORDER BY calculated_at DESC
                LIMIT ?
            """, params + [limit])
            
            rows = cursor.fetchall()
            return [Performance.from_dict(dict(row)) for row in rows]
    
    def get_performance_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Args:
            start_time: Start of period (defaults to 30 days ago)
            end_time: End of period (defaults to now)
            
        Returns:
            Dictionary with performance summary statistics
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            
            # Trade statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    SUM(fees) as total_fees,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) as gross_loss,
                    AVG(pnl) as average_pnl,
                    MAX(pnl) as largest_win,
                    MIN(pnl) as largest_loss
                FROM trades
                WHERE status = 'closed'
                AND exit_time >= ? AND exit_time <= ?
            """, (start_time.isoformat(), end_time.isoformat()))
            
            trade_stats = dict(cursor.fetchone() or {})
            
            # Daily P&L for drawdown calculation
            cursor.execute("""
                SELECT 
                    DATE(exit_time) as date,
                    SUM(pnl - fees) as daily_pnl
                FROM trades
                WHERE status = 'closed'
                AND exit_time >= ? AND exit_time <= ?
                GROUP BY DATE(exit_time)
                ORDER BY date ASC
            """, (start_time.isoformat(), end_time.isoformat()))
            
            daily_pnl = [dict(row) for row in cursor.fetchall()]
            
            # Calculate max drawdown
            max_drawdown = 0.0
            max_drawdown_percent = 0.0
            peak = 0.0
            cumulative = 0.0
            
            for day in daily_pnl:
                cumulative += day.get('daily_pnl', 0) or 0
                if cumulative > peak:
                    peak = cumulative
                drawdown = peak - cumulative
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    if peak > 0:
                        max_drawdown_percent = (drawdown / peak) * 100
            
            # Calculate derived metrics
            total = trade_stats.get('total_trades', 0) or 0
            wins = trade_stats.get('winning_trades', 0) or 0
            losses = trade_stats.get('losing_trades', 0) or 0
            
            summary = {
                'period_start': start_time.isoformat(),
                'period_end': end_time.isoformat(),
                'total_trades': total,
                'winning_trades': wins,
                'losing_trades': losses,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_pnl': trade_stats.get('total_pnl', 0) or 0,
                'total_fees': trade_stats.get('total_fees', 0) or 0,
                'net_pnl': (trade_stats.get('total_pnl', 0) or 0) - (trade_stats.get('total_fees', 0) or 0),
                'gross_profit': trade_stats.get('gross_profit', 0) or 0,
                'gross_loss': trade_stats.get('gross_loss', 0) or 0,
                'profit_factor': abs((trade_stats.get('gross_profit', 0) or 0) / (trade_stats.get('gross_loss', 0) or 1)) if (trade_stats.get('gross_loss', 0) or 0) != 0 else float('inf'),
                'average_pnl': trade_stats.get('average_pnl', 0) or 0,
                'average_win': ((trade_stats.get('gross_profit', 0) or 0) / wins) if wins > 0 else 0,
                'average_loss': ((trade_stats.get('gross_loss', 0) or 0) / losses) if losses > 0 else 0,
                'largest_win': trade_stats.get('largest_win', 0) or 0,
                'largest_loss': trade_stats.get('largest_loss', 0) or 0,
                'max_drawdown': max_drawdown,
                'max_drawdown_percent': max_drawdown_percent,
                'daily_pnl': daily_pnl
            }
            
            return summary
    
    # =========================================================================
    # BALANCE HISTORY OPERATIONS
    # =========================================================================
    
    def save_balance(self, balance: BalanceHistory) -> str:
        """
        Save a balance history record.
        
        Args:
            balance: BalanceHistory model instance
            
        Returns:
            The balance_id of the saved record
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = balance.to_dict()
            
            cursor.execute("""
                INSERT INTO balance_history (
                    balance_id, timestamp, total_balance, available_balance,
                    margin_balance, unrealized_pnl, realized_pnl_today,
                    realized_pnl_week, realized_pnl_month, currency, source, metadata
                ) VALUES (
                    :balance_id, :timestamp, :total_balance, :available_balance,
                    :margin_balance, :unrealized_pnl, :realized_pnl_today,
                    :realized_pnl_week, :realized_pnl_month, :currency, :source, :metadata
                )
            """, data)
            
            logger.debug(f"Balance saved: {balance.balance_id}")
            return balance.balance_id
    
    def get_balance_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[BalanceHistory]:
        """
        Get balance history.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of results
            
        Returns:
            List of BalanceHistory instances
        """
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM balance_history
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit])
            
            rows = cursor.fetchall()
            return [BalanceHistory.from_dict(dict(row)) for row in rows]
    
    def get_latest_balance(self) -> Optional[BalanceHistory]:
        """
        Get the most recent balance record.
        
        Returns:
            BalanceHistory instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute("""
                SELECT * FROM balance_history
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                return BalanceHistory.from_dict(dict(row))
            return None
    
    # =========================================================================
    # TRADE JOURNAL OPERATIONS
    # =========================================================================
    
    def save_trade_journal(self, journal: TradeJournal) -> str:
        """
        Save a trade journal entry.
        
        Args:
            journal: TradeJournal model instance
            
        Returns:
            The journal_id of the saved entry
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = journal.to_dict()
            
            cursor.execute("""
                INSERT OR REPLACE INTO trade_journal (
                    journal_id, trade_id, entry_notes, exit_notes,
                    lessons_learned, emotional_state, market_conditions,
                    setup_quality, execution_quality, tags, screenshots, updated_at
                ) VALUES (
                    :journal_id, :trade_id, :entry_notes, :exit_notes,
                    :lessons_learned, :emotional_state, :market_conditions,
                    :setup_quality, :execution_quality, :tags, :screenshots, CURRENT_TIMESTAMP
                )
            """, data)
            
            logger.debug(f"Trade journal saved: {journal.journal_id}")
            return journal.journal_id
    
    def get_trade_journal(self, trade_id: str) -> Optional[TradeJournal]:
        """
        Get trade journal entry for a trade.
        
        Args:
            trade_id: The trade ID
            
        Returns:
            TradeJournal instance or None if not found
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(
                "SELECT * FROM trade_journal WHERE trade_id = ?",
                (trade_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return TradeJournal.from_dict(dict(row))
            return None
    
    # =========================================================================
    # SYSTEM LOG OPERATIONS
    # =========================================================================
    
    def save_system_log(self, log: SystemLog) -> str:
        """
        Save a system log entry.
        
        Args:
            log: SystemLog model instance
            
        Returns:
            The log_id of the saved entry
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            data = log.to_dict()
            
            cursor.execute("""
                INSERT INTO system_logs (
                    log_id, level, component, message, details, traceback, timestamp
                ) VALUES (
                    :log_id, :level, :component, :message, :details, :traceback, :timestamp
                )
            """, data)
            
            return log.log_id
    
    def get_system_logs(
        self,
        level: Optional[Union[LogLevel, str]] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SystemLog]:
        """
        Get system logs.
        
        Args:
            level: Filter by log level
            component: Filter by component
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of results
            
        Returns:
            List of SystemLog instances
        """
        conditions = []
        params = []
        
        if level:
            level_val = level.value if isinstance(level, LogLevel) else level
            conditions.append("level = ?")
            params.append(level_val)
        if component:
            conditions.append("component = ?")
            params.append(component)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(f"""
                SELECT * FROM system_logs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit])
            
            rows = cursor.fetchall()
            return [SystemLog.from_dict(dict(row)) for row in rows]
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries containing query results
        """
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        tables = [
            'trades', 'signals', 'positions', 'orders', 'hedges',
            'market_data', 'correlations', 'performance', 'balance_history',
            'trade_journal', 'system_logs'
        ]
        
        stats = {}
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                row = cursor.fetchone()
                stats[table] = row['count'] if row else 0
            
            # Database file size
            stats['database_size_bytes'] = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
        return stats
    
    def vacuum(self) -> None:
        """Optimize the database by running VACUUM."""
        with self._get_connection() as conn:
            cursor = self._get_cursor(conn)
            cursor.execute("VACUUM")
            logger.info("Database vacuumed successfully")
    
    def close(self) -> None:
        """Close any open database connections."""
        if self._connection:
            self._connection.close()
            self._connection = None