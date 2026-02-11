#!/usr/bin/env python3
"""
Diagnostic script to analyze trade data and identify root causes of losses.
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Any
import json

def load_trade_data(year: str) -> pl.DataFrame:
    """Load trade data for a given year."""
    path = f"backtest_results/{year}/trades.csv"
    if not Path(path).exists():
        path = f"backtest_results/{year}_test/trades.csv"
    df = pl.read_csv(path)
    return df.with_columns(pl.lit(year).alias("year"))

def analyze_stop_loss_distance(df: pl.DataFrame) -> Dict[str, Any]:
    """Analyze stop loss distance as percentage of entry price."""
    df = df.filter(pl.col("reason") == "stop_loss")
    
    # Calculate stop loss distance
    df = df.with_columns([
        ((pl.col("entry_price") - pl.col("exit_price")).abs() / pl.col("entry_price") * 100).alias("sl_distance_pct")
    ])
    
    return {
        "avg_sl_distance_pct": df["sl_distance_pct"].mean(),
        "median_sl_distance_pct": df["sl_distance_pct"].median(),
        "max_sl_distance_pct": df["sl_distance_pct"].max(),
        "min_sl_distance_pct": df["sl_distance_pct"].min(),
    }

def analyze_trade_duration(df: pl.DataFrame) -> Dict[str, Any]:
    """Analyze trade duration."""
    df = df.with_columns([
        (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False))
    ])
    
    # We need entry timestamp, but it's not in the CSV. Let's use what we have.
    return {
        "total_trades": len(df),
    }

def analyze_pnl_by_exit_reason(df: pl.DataFrame) -> Dict[str, Any]:
    """Analyze PnL by exit reason."""
    result = {}
    for reason in df["reason"].unique():
        reason_df = df.filter(pl.col("reason") == reason)
        result[reason] = {
            "count": len(reason_df),
            "total_pnl": float(reason_df["pnl"].sum()),
            "avg_pnl": float(reason_df["pnl"].mean()),
            "win_rate": float((reason_df["pnl"] > 0).sum() / len(reason_df) * 100) if len(reason_df) > 0 else 0,
        }
    return result

def analyze_symbol_performance(df: pl.DataFrame) -> Dict[str, Any]:
    """Analyze performance by symbol."""
    result = {}
    for symbol in df["symbol"].unique():
        symbol_df = df.filter(pl.col("symbol") == symbol)
        result[symbol] = {
            "total_trades": len(symbol_df),
            "total_pnl": float(symbol_df["pnl"].sum()),
            "avg_pnl": float(symbol_df["pnl"].mean()),
            "total_fees": float(symbol_df["fees"].sum()),
            "win_rate": float((symbol_df["pnl"] > 0).sum() / len(symbol_df) * 100) if len(symbol_df) > 0 else 0,
            "avg_win": float(symbol_df.filter(pl.col("pnl") > 0)["pnl"].mean()) if (symbol_df["pnl"] > 0).sum() > 0 else 0,
            "avg_loss": float(symbol_df.filter(pl.col("pnl") < 0)["pnl"].mean()) if (symbol_df["pnl"] < 0).sum() > 0 else 0,
        }
    return result

def main():
    """Main analysis function."""
    years = ["2023", "2024", "2025"]
    all_data = []
    
    for year in years:
        try:
            df = load_trade_data(year)
            all_data.append(df)
            print(f"\n{'='*60}")
            print(f"YEAR: {year}")
            print(f"{'='*60}")
            
            # Basic stats
            print(f"\nTotal Trades: {len(df)}")
            print(f"Total PnL: ${df['pnl'].sum():.2f}")
            print(f"Avg PnL: ${df['pnl'].mean():.2f}")
            print(f"Win Rate: {(df['pnl'] > 0).sum() / len(df) * 100:.2f}%")
            print(f"Total Fees: ${df['fees'].sum():.2f}")
            
            # PnL by exit reason
            print(f"\n--- PnL by Exit Reason ---")
            pnl_by_reason = analyze_pnl_by_exit_reason(df)
            for reason, stats in sorted(pnl_by_reason.items(), key=lambda x: x[1]["count"], reverse=True):
                print(f"{reason}: {stats['count']} trades, PnL=${stats['total_pnl']:.2f}, Avg=${stats['avg_pnl']:.2f}, Win Rate={stats['win_rate']:.1f}%")
            
            # Symbol performance
            print(f"\n--- Symbol Performance ---")
            symbol_perf = analyze_symbol_performance(df)
            for symbol, stats in sorted(symbol_perf.items()):
                print(f"{symbol}: {stats['total_trades']} trades, PnL=${stats['total_pnl']:.2f}, Win Rate={stats['win_rate']:.1f}%, Avg Win=${stats['avg_win']:.2f}, Avg Loss=${stats['avg_loss']:.2f}")
            
            # Stop loss distance
            print(f"\n--- Stop Loss Distance ---")
            sl_stats = analyze_stop_loss_distance(df)
            print(f"Avg SL Distance: {sl_stats['avg_sl_distance_pct']:.2f}%")
            print(f"Median SL Distance: {sl_stats['median_sl_distance_pct']:.2f}%")
            print(f"Max SL Distance: {sl_stats['max_sl_distance_pct']:.2f}%")
            print(f"Min SL Distance: {sl_stats['min_sl_distance_pct']:.2f}%")
            
        except Exception as e:
            print(f"Error loading {year}: {e}")
    
    # Combined analysis
    if all_data:
        combined = pl.concat(all_data)
        print(f"\n{'='*60}")
        print(f"COMBINED ANALYSIS (2023-2025)")
        print(f"{'='*60}")
        print(f"\nTotal Trades: {len(combined)}")
        print(f"Total PnL: ${combined['pnl'].sum():.2f}")
        print(f"Avg PnL: ${combined['pnl'].mean():.2f}")
        print(f"Win Rate: {(combined['pnl'] > 0).sum() / len(combined) * 100:.2f}%")
        print(f"Total Fees: ${combined['fees'].sum():.2f}")
        
        # PnL by exit reason (combined)
        print(f"\n--- PnL by Exit Reason (Combined) ---")
        pnl_by_reason = analyze_pnl_by_exit_reason(combined)
        for reason, stats in sorted(pnl_by_reason.items(), key=lambda x: x[1]["count"], reverse=True):
            print(f"{reason}: {stats['count']} trades ({stats['count']/len(combined)*100:.1f}%), PnL=${stats['total_pnl']:.2f}, Avg=${stats['avg_pnl']:.2f}, Win Rate={stats['win_rate']:.1f}%")
        
        # Symbol performance (combined)
        print(f"\n--- Symbol Performance (Combined) ---")
        symbol_perf = analyze_symbol_performance(combined)
        for symbol, stats in sorted(symbol_perf.items()):
            print(f"{symbol}: {stats['total_trades']} trades, PnL=${stats['total_pnl']:.2f}, Win Rate={stats['win_rate']:.1f}%, Avg Win=${stats['avg_win']:.2f}, Avg Loss=${stats['avg_loss']:.2f}")

if __name__ == "__main__":
    main()
