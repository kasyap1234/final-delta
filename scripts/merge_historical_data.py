#!/usr/bin/env python3
"""
Merge historical data from multiple years into single CSV files.

This script combines yearly data files into single files for backtesting.
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_csv_file(filepath: Path) -> List[List]:
    """Parse a CSV file and return rows."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 6:
                try:
                    timestamp = datetime.fromisoformat(row[0])
                    rows.append([
                        timestamp,
                        float(row[1]),  # open
                        float(row[2]),  # high
                        float(row[3]),  # low
                        float(row[4]),  # close
                        float(row[5])   # volume
                    ])
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid row in {filepath}: {e}")
                    continue
    return rows


def merge_data_files(
    data_dir: Path,
    symbol: str,
    timeframe: str,
    years: List[int]
) -> List[List]:
    """Merge data files for multiple years."""
    all_rows = []
    
    for year in years:
        # Look for year-specific files
        year_file = data_dir / f"{symbol.replace('/', '_')}_{year}_{timeframe}.csv"
        if year_file.exists():
            logger.info(f"Loading {year_file}")
            rows = parse_csv_file(year_file)
            all_rows.extend(rows)
        else:
            logger.warning(f"File not found: {year_file}")
    
    # Sort by timestamp
    all_rows.sort(key=lambda x: x[0])
    
    # Remove duplicates based on timestamp
    seen_timestamps = set()
    unique_rows = []
    for row in all_rows:
        ts = row[0]
        if ts not in seen_timestamps:
            seen_timestamps.add(ts)
            unique_rows.append(row)
    
    return unique_rows


def save_merged_data(
    rows: List[List],
    output_path: Path
) -> None:
    """Save merged data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        for row in rows:
            writer.writerow([
                row[0].isoformat(),
                row[1],
                row[2],
                row[3],
                row[4],
                row[5]
            ])
    
    logger.info(f"Saved {len(rows)} rows to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Merge historical data from multiple years'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/backtest',
        help='Directory containing data files'
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
        help='Timeframe of data'
    )
    parser.add_argument(
        '--years',
        type=str,
        default='2023,2024,2025',
        help='Comma-separated list of years to merge'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/backtest',
        help='Output directory for merged files'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    symbols = [s.strip() for s in args.symbols.split(',')]
    years = [int(y.strip()) for y in args.years.split(',')]
    
    logger.info(f"Merging data for symbols: {symbols}, years: {years}")
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        # Merge data
        rows = merge_data_files(data_dir, symbol, args.timeframe, years)
        
        if rows:
            # Save merged file
            output_file = output_dir / f"{symbol.replace('/', '_')}_{args.timeframe}.csv"
            save_merged_data(rows, output_file)
            
            logger.info(
                f"Merged data for {symbol}: {len(rows)} rows, "
                f"from {rows[0][0]} to {rows[-1][0]}"
            )
        else:
            logger.warning(f"No data found for {symbol}")
    
    logger.info("Merge completed")


if __name__ == '__main__':
    main()
