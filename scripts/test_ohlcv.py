#!/usr/bin/env python3
"""Test fetching OHLCV data."""
import asyncio
import ccxt.async_support as ccxt
from datetime import datetime

async def main():
    ex = ccxt.delta({'sandbox': True})
    await ex.load_markets()
    
    # Try different symbol formats
    symbols_to_try = ['BTC/USDT', 'BTC/USD:BTC', 'ETH/USDT', 'ETH/USD:USDC']
    
    for symbol in symbols_to_try:
        if symbol in ex.markets:
            print(f"\n=== Testing {symbol} ===")
            try:
                # Fetch last 10 candles
                candles = await ex.fetch_ohlcv(symbol, '15m', limit=10)
                print(f"Success! Got {len(candles)} candles")
                if candles:
                    print(f"First candle: {candles[0]}")
                    print(f"Last candle: {candles[-1]}")
                    # Convert timestamp to date
                    ts = candles[0][0]
                    dt = datetime.fromtimestamp(ts / 1000)
                    print(f"First candle date: {dt.isoformat()}")
            except Exception as e:
                print(f"Error: {e}")
    
    await ex.close()

if __name__ == '__main__':
    asyncio.run(main())
