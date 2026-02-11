"""
Trade logger module for backtesting.

This module provides trade logging functionality for backtesting system.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Log trades to CSV and JSON files.
    
    This class handles:
    - Exporting trades to CSV
    - Exporting trades to JSON
    - Appending to existing logs
    """
    
    def __init__(self, output_dir: str = "backtest_results"):
        """
        Initialize trade logger.
        
        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TradeLogger initialized: output_dir={self.output_dir}")
    
    def export_to_csv(
        self,
        trades: List[Dict[str, Any]],
        filename: str = "trades.csv"
    ) -> str:
        """
        Export trades to CSV file.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename
            
        Returns:
            Path to the created file
        """
        filepath = self.output_dir / filename
        
        if not trades:
            logger.warning("No trades to export")
            return str(filepath)
        
        # Get all unique keys from trades
        fieldnames = set()
        for trade in trades:
            fieldnames.update(trade.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)
        
        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return str(filepath)
    
    def export_to_json(
        self,
        trades: List[Dict[str, Any]],
        filename: str = "trades.json"
    ) -> str:
        """
        Export trades to JSON file.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename
            
        Returns:
            Path to the created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        
        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return str(filepath)
    
    def append_to_csv(
        self,
        trade: Dict[str, Any],
        filename: str = "trades.csv"
    ) -> None:
        """
        Append a single trade to CSV file.
        
        Args:
            trade: Trade dictionary to append
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        # Check if file exists to determine if we need a header
        file_exists = filepath.exists()
        
        # Get all keys from trade
        fieldnames = sorted(trade.keys())
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(trade)
        
        logger.debug(f"Appended trade to {filepath}")
    
    def export_summary(
        self,
        trades: List[Dict[str, Any]],
        filename: str = "trade_summary.txt"
    ) -> str:
        """
        Export a human-readable trade summary.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename
            
        Returns:
            Path to the created file
        """
        filepath = self.output_dir / filename
        
        if not trades:
            summary = "No trades to summarize."
        else:
            # Calculate summary statistics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            total_fees = sum(t.get('fees', 0) for t in trades)
            
            avg_win = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
            avg_win = avg_win / winning_trades if winning_trades > 0 else 0.0
            
            avg_loss = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0)
            avg_loss = avg_loss / losing_trades if losing_trades > 0 else 0.0
            
            summary = []
            summary.append("=" * 60)
            summary.append("TRADE SUMMARY")
            summary.append("=" * 60)
            summary.append(f"Total Trades: {total_trades}")
            summary.append(f"Winning Trades: {winning_trades}")
            summary.append(f"Losing Trades: {losing_trades}")
            summary.append(f"Win Rate: {win_rate*100:.2f}%")
            summary.append(f"Total P&L: ${total_pnl:.2f}")
            summary.append(f"Total Fees: ${total_fees:.2f}")
            summary.append(f"Net P&L: ${total_pnl - total_fees:.2f}")
            summary.append(f"Avg Win: ${avg_win:.2f}")
            summary.append(f"Avg Loss: ${avg_loss:.2f}")
            summary.append("=" * 60)
            
            summary = "\n".join(summary)
        
        with open(filepath, 'w') as f:
            f.write(summary)
        
        logger.info(f"Exported trade summary to {filepath}")
        return str(filepath)
    
    def get_output_dir(self) -> Path:
        """
        Get the output directory.
        
        Returns:
            Path to output directory
        """
        return self.output_dir
    
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set the output directory.
        
        Args:
            output_dir: New output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory changed to: {self.output_dir}")
