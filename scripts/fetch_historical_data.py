#!/usr/bin/env python3
"""
Historical Data Fetcher Script

This script fetches historical OHLCV data from the exchange for backtesting.
It supports fetching data for multiple symbols, timeframes, and date ranges.

Usage:
    python scripts/fetch_historical_data.py [options]

Options:
    --start-date START_DATE    Start date in ISO format (default: 2025-01-01)
    --end-date END_DATE        End date in ISO format (default: 2025-12-31)
    --symbols SYMBOLS         Comma-separated list of symbols (default: BTC/USD,ETH/USD,SOL/USD)
    --timeframe TIMEFRAME     Candle timeframe (default: 15m)
    --output-dir OUTPUT_DIR   Output directory for CSV files (default: data/backtest)
    --config CONFIG           Path to config file (default: config/config.yaml)
    --force                   Force re-fetch even if data exists
"""

import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager
from src.exchange.exchange_client import ExchangeClient
from src.utils.logger import setup_logging


logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetch historical OHLCV data from exchange and save to CSV files.
    """
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        timeframe: str,
        output_dir: Path,
        config_path: Optional[Path] = None
    ):
        """
        Initialize the data fetcher.
        
        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching
            symbols: List of trading symbols
            timeframe: Candle timeframe
            output_dir: Directory to save CSV files
            config_path: Path to config file
        """
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.timeframe = timeframe
        self.output_dir = output_dir
        self.config_path = config_path
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange client
        self.client: Optional[ExchangeClient] = None
        
        logger.info(
            f"HistoricalDataFetcher initialized: "
            f"start={start_date.isoformat()}, end={end_date.isoformat()}, "
            f"symbols={symbols}, timeframe={timeframe}"
        )
    
    async def connect(self) -> None:
        """Connect to the exchange."""
        try:
            # Load config
            if self.config_path and self.config_path.exists():
                config_manager = ConfigManager(self.config_path)
                config = config_manager.load_config()
                self.client = ExchangeClient.from_config(config)
            else:
                # Create client with default config - use mainnet for real historical data
                from src.exchange.exchange_client import ExchangeClientConfig
                client_config = ExchangeClientConfig(
                    exchange_id='delta',
                    sandbox=False,  # Use mainnet for real historical data
                    testnet=False   # Use mainnet for real historical data
                )
                self.client = ExchangeClient(client_config)
            
            await self.client.connect()
            logger.info("Connected to exchange")
            
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise
    
    async def close(self) -> None:
        """Close the exchange connection."""
        if self.client:
            await self.client.close()
            logger.info("Exchange connection closed")
    
    async def fetch_symbol_data(self, symbol: str) -> List[List[float]]:
        """
        Fetch OHLCV data for a single symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of OHLCV candles [timestamp, open, high, low, close, volume]
        """
        logger.info(f"Fetching data for {symbol}...")
        
        all_candles = []
        current_timestamp = int(self.start_date.timestamp() * 1000)
        end_timestamp = int(self.end_date.timestamp() * 1000)
        
        # Calculate batch size based on timeframe
        # Most exchanges limit to 500-1000 candles per request
        batch_size = 500
        
        while current_timestamp < end_timestamp:
            try:
                # Fetch a batch of candles
                candles = await self.client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    since=current_timestamp,
                    limit=batch_size
                )
                
                if not candles:
                    logger.warning(f"No more candles for {symbol} at {current_timestamp}")
                    break
                
                all_candles.extend(candles)
                
                # Update timestamp to the last candle's timestamp + 1ms
                last_timestamp = candles[-1][0]
                current_timestamp = last_timestamp + 1
                
                # Log progress
                logger.info(
                    f"Fetched {len(candles)} candles for {symbol}, "
                    f"total: {len(all_candles)}, "
                    f"last: {datetime.fromtimestamp(last_timestamp / 1000).isoformat()}"
                )
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching candles for {symbol}: {e}")
                raise
        
        logger.info(f"Total candles fetched for {symbol}: {len(all_candles)}")
        return all_candles
    
    def save_to_csv(self, symbol: str, candles: List[List[float]]) -> Path:
        """
        Save OHLCV data to CSV file.
        
        Args:
            symbol: Trading symbol
            candles: List of OHLCV candles
            
        Returns:
            Path to saved CSV file
        """
        # Convert symbol to filename (e.g., BTC/USD -> BTC_USD_15m.csv)
        filename = f"{symbol.replace('/', '_')}_{self.timeframe}.csv"
        filepath = self.output_dir / filename
        
        logger.info(f"Saving {len(candles)} candles to {filepath}")
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Write data
            for candle in candles:
                timestamp_ms, open_price, high, low, close, volume = candle
                
                # Convert timestamp to ISO format
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                timestamp_str = timestamp.isoformat()
                
                writer.writerow([
                    timestamp_str,
                    open_price,
                    high,
                    low,
                    close,
                    volume
                ])
        
        logger.info(f"Saved data to {filepath}")
        return filepath
    
    def validate_data(self, symbol: str, candles: List[List[float]]) -> bool:
        """
        Validate fetched data for completeness and gaps.
        
        Args:
            symbol: Trading symbol
            candles: List of OHLCV candles
            
        Returns:
            True if data is valid
        """
        if not candles:
            logger.error(f"No candles fetched for {symbol}")
            return False
        
        # Check date range
        first_timestamp = candles[0][0]
        last_timestamp = candles[-1][0]
        
        first_date = datetime.fromtimestamp(first_timestamp / 1000)
        last_date = datetime.fromtimestamp(last_timestamp / 1000)
        
        logger.info(
            f"Data range for {symbol}: {first_date.isoformat()} to {last_date.isoformat()}"
        )
        
        # Check if data covers the requested range
        if first_date > self.start_date:
            logger.warning(
                f"Data for {symbol} starts after requested start date: "
                f"{first_date.isoformat()} > {self.start_date.isoformat()}"
            )
        
        if last_date < self.end_date:
            logger.warning(
                f"Data for {symbol} ends before requested end date: "
                f"{last_date.isoformat()} < {self.end_date.isoformat()}"
            )
        
        # Check for gaps
        expected_interval_ms = self._get_timeframe_ms()
        gaps = []
        
        for i in range(1, len(candles)):
            time_diff = candles[i][0] - candles[i-1][0]
            
            if time_diff > expected_interval_ms * 2:
                gap_start = datetime.fromtimestamp(candles[i-1][0] / 1000)
                gap_end = datetime.fromtimestamp(candles[i][0] / 1000)
                gaps.append((gap_start, gap_end, time_diff))
        
        if gaps:
            logger.warning(f"Found {len(gaps)} gaps in data for {symbol}:")
            for gap_start, gap_end, time_diff in gaps[:5]:  # Show first 5 gaps
                logger.warning(
                    f"  Gap: {gap_start.isoformat()} to {gap_end.isoformat()} "
                    f"({time_diff / 1000 / 60:.1f} minutes)"
                )
        else:
            logger.info(f"No gaps found in data for {symbol}")
        
        return True
    
    def _get_timeframe_ms(self) -> int:
        """
        Get timeframe in milliseconds.
        
        Returns:
            Timeframe in milliseconds
        """
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '12h': 720,
            '1d': 1440
        }
        
        minutes = timeframe_minutes.get(self.timeframe, 15)
        return minutes * 60 * 1000
    
    def data_exists(self, symbol: str) -> bool:
        """
        Check if data file already exists for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if data file exists
        """
        filename = f"{symbol.replace('/', '_')}_{self.timeframe}.csv"
        filepath = self.output_dir / filename
        return filepath.exists()
    
    async def fetch_all(self, force: bool = False) -> None:
        """
        Fetch data for all symbols.
        
        Args:
            force: Force re-fetch even if data exists
        """
        logger.info(f"Starting data fetch for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                # Check if data already exists
                if not force and self.data_exists(symbol):
                    logger.info(f"Data already exists for {symbol}, skipping. Use --force to re-fetch.")
                    continue
                
                # Fetch data
                candles = await self.fetch_symbol_data(symbol)
                
                # Validate data
                if not self.validate_data(symbol, candles):
                    logger.error(f"Data validation failed for {symbol}")
                    continue
                
                # Save to CSV
                self.save_to_csv(symbol, candles)
                
                logger.info(f"Successfully fetched and saved data for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        logger.info("Data fetch completed")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fetch historical OHLCV data for backtesting'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-01-01',
        help='Start date in ISO format (default: 2025-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-12-31',
        help='End date in ISO format (default: 2025-12-31)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default='BTC/USDT,ETH/USDT,SOL/USDT',
        help='Comma-separated list of symbols (default: BTC/USDT,ETH/USDT,SOL/USDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        help='Candle timeframe (default: 15m)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/backtest',
        help='Output directory for CSV files (default: data/backtest)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-fetch even if data exists'
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging({'logging': {'level': 'INFO', 'console': True}})
    
    # Parse dates
    try:
        start_date = datetime.fromisoformat(args.start_date)
        end_date = datetime.fromisoformat(args.end_date)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Parse output directory
    output_dir = Path(args.output_dir)
    
    # Parse config path
    config_path = Path(args.config) if args.config else None
    
    # Create fetcher
    fetcher = HistoricalDataFetcher(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        timeframe=args.timeframe,
        output_dir=output_dir,
        config_path=config_path
    )
    
    try:
        # Connect to exchange
        await fetcher.connect()
        
        # Fetch all data
        await fetcher.fetch_all(force=args.force)
        
        logger.info("Historical data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data fetch: {e}")
        sys.exit(1)
        
    finally:
        # Close connection
        await fetcher.close()


if __name__ == '__main__':
    asyncio.run(main())
