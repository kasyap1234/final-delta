#!/usr/bin/env python3
"""
Performance Report Generator for Delta Trading Bot

Generates a comprehensive trading performance report from the database.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def connect_db(db_path: str = "data/trading_bot.db") -> sqlite3.Connection:
    """Connect to the trading database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_total_trades(conn: sqlite3.Connection) -> int:
    """Get total number of closed trades."""
    cursor = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
    return cursor.fetchone()[0]


def get_win_rate(conn: sqlite3.Connection) -> float:
    """Calculate win rate percentage."""
    cursor = conn.execute("""
        SELECT 
            COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
            COUNT(*) as total
        FROM trades 
        WHERE status = 'closed'
    """)
    row = cursor.fetchone()
    if row['total'] == 0:
        return 0.0
    return (row['wins'] / row['total']) * 100


def get_total_pnl(conn: sqlite3.Connection) -> float:
    """Get total profit/loss."""
    cursor = conn.execute("""
        SELECT COALESCE(SUM(pnl), 0) as total_pnl
        FROM trades 
        WHERE status = 'closed'
    """)
    return cursor.fetchone()['total_pnl']


def get_average_trade(conn: sqlite3.Connection) -> Dict:
    """Get average trade statistics."""
    cursor = conn.execute("""
        SELECT 
            AVG(pnl) as avg_pnl,
            AVG(pnl_percent) as avg_return,
            AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
            AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss
        FROM trades 
        WHERE status = 'closed'
    """)
    row = cursor.fetchone()
    return {
        'avg_pnl': row['avg_pnl'] or 0,
        'avg_return': row['avg_return'] or 0,
        'avg_win': row['avg_win'] or 0,
        'avg_loss': row['avg_loss'] or 0
    }


def get_symbol_performance(conn: sqlite3.Connection) -> List[Dict]:
    """Get performance breakdown by symbol."""
    cursor = conn.execute("""
        SELECT 
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl
        FROM trades
        WHERE status = 'closed'
        GROUP BY symbol
        ORDER BY total_pnl DESC
    """)
    return [dict(row) for row in cursor.fetchall()]


def get_daily_performance(conn: sqlite3.Connection, days: int = 30) -> List[Dict]:
    """Get daily P&L for the last N days."""
    cursor = conn.execute("""
        SELECT 
            DATE(entry_time) as date,
            COUNT(*) as trades,
            SUM(pnl) as total_pnl,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
        FROM trades
        WHERE status = 'closed'
            AND entry_time >= date('now', '-{} days')
        GROUP BY DATE(entry_time)
        ORDER BY date DESC
    """.format(days))
    return [dict(row) for row in cursor.fetchall()]


def get_open_positions(conn: sqlite3.Connection) -> List[Dict]:
    """Get currently open positions."""
    cursor = conn.execute("""
        SELECT 
            symbol,
            side,
            size,
            entry_price,
            current_price,
            unrealized_pnl,
            stop_loss,
            take_profit,
            opened_at
        FROM positions
        WHERE status = 'open'
        ORDER BY opened_at DESC
    """)
    return [dict(row) for row in cursor.fetchall()]


def get_hedge_performance(conn: sqlite3.Connection) -> List[Dict]:
    """Get hedge performance summary."""
    cursor = conn.execute("""
        SELECT 
            h.primary_symbol,
            h.hedge_symbol,
            h.correlation_at_hedge,
            h.pnl as hedge_pnl,
            p.pnl as primary_pnl,
            (h.pnl + COALESCE(p.pnl, 0)) as combined_pnl,
            h.opened_at,
            h.closed_at,
            h.status
        FROM hedges h
        LEFT JOIN positions p ON h.primary_position_id = p.position_id
        ORDER BY h.opened_at DESC
        LIMIT 10
    """)
    return [dict(row) for row in cursor.fetchall()]


def print_report():
    """Generate and print the performance report."""
    db_path = "data/trading_bot.db"
    
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        print("Run the bot first to generate trading data.")
        return
    
    conn = connect_db(db_path)
    
    print("=" * 70)
    print("DELTA TRADING BOT - PERFORMANCE REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Overall Statistics
    print("-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)
    
    total_trades = get_total_trades(conn)
    win_rate = get_win_rate(conn)
    total_pnl = get_total_pnl(conn)
    avg_stats = get_average_trade(conn)
    
    print(f"Total Trades:      {total_trades}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"Total P&L:         ${total_pnl:,.2f}")
    print(f"Average P&L:       ${avg_stats['avg_pnl']:,.2f}")
    print(f"Average Return:    {avg_stats['avg_return']:.2f}%")
    
    if avg_stats['avg_win'] and avg_stats['avg_loss']:
        profit_factor = abs(avg_stats['avg_win'] / avg_stats['avg_loss']) if avg_stats['avg_loss'] != 0 else 0
        print(f"Avg Win:           ${avg_stats['avg_win']:,.2f}")
        print(f"Avg Loss:          ${avg_stats['avg_loss']:,.2f}")
        print(f"Profit Factor:     {profit_factor:.2f}")
    
    print()
    
    # Symbol Performance
    print("-" * 70)
    print("PERFORMANCE BY SYMBOL")
    print("-" * 70)
    
    symbol_perf = get_symbol_performance(conn)
    if symbol_perf:
        print(f"{'Symbol':<12} {'Trades':<8} {'Wins':<6} {'Win%':<8} {'Total P&L':<15} {'Avg P&L':<12}")
        print("-" * 70)
        for row in symbol_perf:
            print(f"{row['symbol']:<12} {row['trades']:<8} {row['wins']:<6} "
                  f"{row['win_rate']:<8} ${row['total_pnl']:>12,.2f} ${row['avg_pnl']:>10,.2f}")
    else:
        print("No closed trades found.")
    
    print()
    
    # Open Positions
    print("-" * 70)
    print("OPEN POSITIONS")
    print("-" * 70)
    
    open_positions = get_open_positions(conn)
    if open_positions:
        print(f"{'Symbol':<12} {'Side':<8} {'Size':<10} {'Entry':<12} {'Current':<12} {'P&L':<12}")
        print("-" * 70)
        for pos in open_positions:
            print(f"{pos['symbol']:<12} {pos['side']:<8} {pos['size']:<10.4f} "
                  f"${pos['entry_price']:<11.2f} ${pos['current_price']:<11.2f} "
                  f"${pos['unrealized_pnl']:>10,.2f}")
    else:
        print("No open positions.")
    
    print()
    
    # Hedge Performance
    print("-" * 70)
    print("RECENT HEDGE ACTIVITY")
    print("-" * 70)
    
    hedge_perf = get_hedge_performance(conn)
    if hedge_perf:
        print(f"{'Primary':<12} {'Hedge':<12} {'Correlation':<12} {'Combined P&L':<15} {'Status':<10}")
        print("-" * 70)
        for hedge in hedge_perf:
            print(f"{hedge['primary_symbol']:<12} {hedge['hedge_symbol']:<12} "
                  f"{hedge['correlation_at_hedge']:<12.2f} ${hedge['combined_pnl']:>12,.2f} "
                  f"{hedge['status']:<10}")
    else:
        print("No hedge activity found.")
    
    print()
    
    # Daily Performance
    print("-" * 70)
    print("DAILY PERFORMANCE (Last 7 Days)")
    print("-" * 70)
    
    daily_perf = get_daily_performance(conn, 7)
    if daily_perf:
        print(f"{'Date':<12} {'Trades':<8} {'Wins':<6} {'P&L':<15}")
        print("-" * 70)
        for day in daily_perf:
            print(f"{day['date']:<12} {day['trades']:<8} {day['wins']:<6} ${day['total_pnl']:>12,.2f}")
    else:
        print("No daily data available.")
    
    print()
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)
    
    conn.close()


if __name__ == "__main__":
    print_report()
