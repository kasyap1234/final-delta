#!/usr/bin/env python3
"""
Fetch historical data for all years and save to separate files.

This script fetches data for 2023, 2024, and 2025 and saves them to separate
files for each year to avoid overwriting.
"""

import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange.exchange_client import ExchangeClient, ExchangeClientConfig
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


async def fetch_year_data(
    client: ExchangeClient,
    symbol: str,
    timeframe: str,
    year: int,
    output_dir: Path
) -> None:
    """Fetch data for a specific year."""
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)
    
    logger.info(f"Fetching {symbol} {timeframe} data for {year}...")
    
    all_candles = []
    current_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    batch_size = 500
    
    while current_timestamp < end_timestamp:
        try:
            candles = await client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_timestamp,
                limit=batch_size
            )
            
            if not candles:
                logger.warning(f"No more candles for {symbol} at {current_timestamp}")
                break
            
            all_candles.extend(candles)
            
            last_timestamp = candles[-1][0]
            current_timestamp = last_timestamp + 1
            
            if len(all_candles) % 5000 == 0:
                logger.info(
                    f"Fetched {len(candles)} candles for {symbol}, "
                    f"total: {len(all_candles)}, "
                    f"last: {datetime.fromtimestamp(last_timestamp / 1000).isoformat()}"
                )
            
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            raise
    
    logger.info(f"Total candles fetched for {symbol} {year}: {len(all_candles)}")
    
    # Save to year-specific file
    filename = f"{symbol.replace('/', '_')}_{year}_{timeframe}.csv"
    filepath = output_dir / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        for candle in all_candles:
            timestamp_ms, open_price, high, low, close, volume = candle
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            writer.writerow([
                timestamp.isoformat(),
                open_price,
                high,
                low,
                close,
                volume
            ])
    
    logger.info(f"Saved data to {filepath}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch historical data for all years'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default='BTC/USDT,ETH/USDT,SOL/USDT',
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        help='Candle timeframe'
    )
    parser.add_argument(
        '--years',
        type=str,
        default='2023,2024,2025',
        help='Comma-separated list of years'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/backtest',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    setup_logging({'logging': {'level': 'INFO', 'console': True}})
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    years = [int(y.strip()) for y in args.years.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create exchange client
    client_config = ExchangeClientConfig(
        exchange_id='delta',
        sandbox=False,
        testnet=False
    )
    client = ExchangeClient(client_config)
    
    try:
        await client.connect()
        logger.info("Connected to exchange")
        
        for symbol in symbols:
            for year in years:
                try:
                    await fetch_year_data(
                        client,
                        symbol,
                        args.timeframe,
                        year,
                        output_dir
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} for {year}: {e}")
                    continue
        
        logger.info("Data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data fetch: {e}")
        sys.exit(1)
        
    finally:
        await client.close()
        logger.info("Exchange connection closed")


if __name__ == '__main__':
    asyncio.run(main())
