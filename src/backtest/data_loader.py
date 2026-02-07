"""
Historical data loader module for backtesting.

This module provides functionality to load historical OHLCV data
from various sources (CSV, SQLite, API).
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import polars as pl

from src.data.data_cache import OHLCV
from src.backtest.config import BacktestConfig

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """
    Load historical OHLCV data for backtesting.
    
    This class supports loading data from:
    - CSV files
    - SQLite databases
    - API (placeholder for future implementation)
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the historical data loader.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.data_source = config.data_source
        self.data_dir = Path(config.data_dir)
        
        logger.info(
            f"HistoricalDataLoader initialized: source={self.data_source}, "
            f"dir={self.data_dir}"
        )
    
    def load_data(self) -> Dict[str, List[OHLCV]]:
        """
        Load historical data for all configured symbols.
        
        Returns:
            Dictionary mapping symbols to OHLCV data
        """
        logger.info(f"Loading historical data for {len(self.config.symbols)} symbols")
        
        if self.data_source == 'csv':
            return self._load_from_csv()
        elif self.data_source == 'sqlite':
            return self._load_from_sqlite()
        elif self.data_source == 'api':
            return self._fetch_from_api()
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
    
    def _load_from_csv(self) -> Dict[str, List[OHLCV]]:
        """
        Load historical data from CSV files.
        
        Expected CSV format:
        timestamp,open,high,low,close,volume
        
        Returns:
            Dictionary mapping symbols to OHLCV data
        """
        historical_data = {}
        
        # Check if year-specific symbol files are configured
        symbol_files = getattr(self.config, 'symbol_files', {})
        year = getattr(self.config, 'data_year', None)
        
        for symbol in self.config.symbols:
            # Convert symbol to base filename (e.g., BTC/USD -> BTC_USD)
            symbol_base = symbol.replace('/', '_')
            
            # Try year-specific file first if configured
            filepath = None
            if symbol_base in symbol_files:
                filepath = self.data_dir / symbol_files[symbol_base]
            elif year:
                # Try year-specific filename pattern: BTC_USDT_2023_15m.csv
                year_filename = f"{symbol_base}_{year}_{self.config.timeframe}.csv"
                filepath = self.data_dir / year_filename
            
            # Fallback to default patterns
            if not filepath or not filepath.exists():
                filename_base = f"{symbol_base}.csv"
                filename_with_timeframe = f"{symbol_base}_{self.config.timeframe}.csv"
                
                filepath = self.data_dir / filename_base
                if not filepath.exists():
                    filepath = self.data_dir / filename_with_timeframe
            
            if not filepath.exists():
                logger.warning(f"CSV file not found: {filepath}")
                continue
            
            try:
                candles = self._parse_csv(filepath, symbol)
                historical_data[symbol] = candles
                logger.info(
                    f"Loaded {len(candles)} candles for {symbol} from {filepath.name}"
                )
            except Exception as e:
                logger.error(f"Error parsing CSV file {filepath}: {e}")
                raise
        
        return historical_data
    
    def _parse_csv(self, filepath: Path, symbol: str) -> List[OHLCV]:
        """
        Parse a CSV file and return OHLCV data.
        
        Expected CSV format:
        timestamp,open,high,low,close,volume
        
        Args:
            filepath: Path to CSV file
            symbol: Trading pair symbol
            
        Returns:
            List of OHLCV objects
        """
        timeframe = self.config.timeframe

        try:
            df = pl.read_csv(
                filepath,
                schema_overrides={
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to read CSV {filepath} with polars: {e}")
            return []

        df = df.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False)
        )

        null_count = df["timestamp"].null_count()
        if null_count > 0:
            logger.warning(f"Skipping {null_count} rows with invalid timestamps in {filepath}")
            df = df.filter(pl.col("timestamp").is_not_null())

        df = df.sort("timestamp")

        timestamps = df["timestamp"].to_list()
        opens = df["open"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        closes = df["close"].to_list()
        volumes = df["volume"].to_list()

        candles = [
            OHLCV(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            )
            for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes)
        ]

        return candles
    
    def _load_from_sqlite(self) -> Dict[str, List[OHLCV]]:
        """
        Load historical data from SQLite database.
        
        Expected table schema:
        CREATE TABLE ohlcv (
            symbol TEXT,
            timestamp DATETIME,
            timeframe TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        
        Returns:
            Dictionary mapping symbols to OHLCV data
        """
        historical_data = {}
        db_path = self.data_dir / "historical_data.db"
        
        if not db_path.exists():
            logger.error(f"SQLite database not found: {db_path}")
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for symbol in self.config.symbols:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp
                """
                
                cursor.execute(query, (symbol, self.config.timeframe))
                rows = cursor.fetchall()
                
                candles = []
                timeframe = self.config.timeframe
                for row in rows:
                    candle = OHLCV(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=datetime.fromisoformat(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5])
                    )
                    candles.append(candle)
                
                historical_data[symbol] = candles
                logger.info(f"Loaded {len(candles)} candles for {symbol} from SQLite")
            
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            raise
        
        return historical_data
    
    def _fetch_from_api(self) -> Dict[str, List[OHLCV]]:
        """
        Fetch historical data from API.
        
        This is a placeholder for future implementation.
        In a real implementation, this would fetch data from
        an exchange API like CCXT.
        
        Returns:
            Dictionary mapping symbols to OHLCV data
        """
        logger.warning("API data source not implemented yet")
        raise NotImplementedError("API data source not implemented")
    
    def validate_data(self, data: Dict[str, List[OHLCV]]) -> bool:
        """
        Validate loaded historical data.
        
        Args:
            data: Dictionary mapping symbols to OHLCV data
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data is invalid
        """
        # Check if we have data for all symbols
        missing_symbols = set(self.config.symbols) - set(data.keys())
        if missing_symbols:
            raise ValueError(f"Missing data for symbols: {missing_symbols}")
        
        # Validate each symbol's data
        for symbol, candles in data.items():
            if not candles:
                raise ValueError(f"No data for symbol: {symbol}")
            
            # Check date range
            first_candle = candles[0]
            last_candle = candles[-1]
            
            # Make config dates timezone-naive for comparison
            start_date = self.config.start_date.replace(tzinfo=None) if self.config.start_date.tzinfo else self.config.start_date
            end_date = self.config.end_date.replace(tzinfo=None) if self.config.end_date.tzinfo else self.config.end_date
            
            if first_candle.timestamp > start_date:
                logger.warning(
                    f"Data for {symbol} starts after backtest start date: "
                    f"{first_candle.timestamp} > {start_date}"
                )
            
            if last_candle.timestamp < end_date:
                logger.warning(
                    f"Data for {symbol} ends before backtest end date: "
                    f"{last_candle.timestamp} < {end_date}"
                )
            
            # Check for gaps in data
            expected_diff = self._get_expected_time_diff()
            for i in range(1, len(candles)):
                time_diff = candles[i].timestamp - candles[i-1].timestamp
                
                time_diff_seconds = time_diff.total_seconds()
                
                if time_diff_seconds > expected_diff * 2:
                    logger.warning(
                        f"Large gap in data for {symbol} at {candles[i].timestamp}"
                    )
        
        logger.info("Data validation passed")
        return True
    
    def _get_expected_time_diff(self) -> float:
        """
        Get expected time difference between candles.
        
        Returns:
            Time difference in seconds
        """
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        minutes = timeframe_minutes.get(self.config.timeframe, 15)
        return minutes * 60
    
    def filter_by_date_range(
        self,
        data: Dict[str, List[OHLCV]]
    ) -> Dict[str, List[OHLCV]]:
        """
        Filter data by backtest date range.
        
        Args:
            data: Dictionary mapping symbols to OHLCV data
            
        Returns:
            Filtered data dictionary
        """
        filtered_data = {}
        
        # Make config dates timezone-naive for comparison
        start_date = self.config.start_date.replace(tzinfo=None) if self.config.start_date.tzinfo else self.config.start_date
        end_date = self.config.end_date.replace(tzinfo=None) if self.config.end_date.tzinfo else self.config.end_date
        
        for symbol, candles in data.items():
            filtered = [
                c for c in candles
                if start_date <= c.timestamp <= end_date
            ]
            filtered_data[symbol] = filtered
        
        return filtered_data
    
    def get_stats(self, data: Dict[str, List[OHLCV]]) -> Dict[str, Any]:
        """
        Get statistics about loaded data.
        
        Args:
            data: Dictionary mapping symbols to OHLCV data
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'symbols': len(data),
            'total_candles': sum(len(candles) for candles in data.values()),
            'date_range': {}
        }
        
        for symbol, candles in data.items():
            if candles:
                stats['date_range'][symbol] = {
                    'start': candles[0].timestamp.isoformat(),
                    'end': candles[-1].timestamp.isoformat(),
                    'count': len(candles)
                }
        
        return stats
