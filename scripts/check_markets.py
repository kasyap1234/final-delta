#!/usr/bin/env python3
"""Check available markets on Delta Exchange."""
import asyncio
import ccxt.async_support as ccxt

async def main():
    ex = ccxt.delta({'sandbox': True})
    await ex.load_markets()
    
    print("=== All Markets ===")
    for m in sorted(ex.markets.keys())[:50]:
        print(m)
    
    print("\n=== BTC Markets ===")
    for m in sorted(ex.markets.keys()):
        if 'BTC' in m:
            print(m)
    
    print("\n=== ETH Markets ===")
    for m in sorted(ex.markets.keys()):
        if 'ETH' in m:
            print(m)
    
    await ex.close()

if __name__ == '__main__':
    asyncio.run(main())
