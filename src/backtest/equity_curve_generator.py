"""
Equity curve generator module for backtesting.

This module provides equity curve visualization functionality for backtesting system.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class EquityCurveGenerator:
    """
    Generate equity curve visualizations and exports.
    
    This class handles:
    - Exporting equity curve to CSV
    - Exporting equity curve to JSON
    - Generating equity curve statistics
    """
    
    def __init__(self, output_dir: str = "backtest_results"):
        """
        Initialize equity curve generator.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"EquityCurveGenerator initialized: output_dir={self.output_dir}")
    
    def export_to_csv(
        self,
        equity_curve: List[Dict[str, Any]],
        filename: str = "equity_curve.csv"
    ) -> str:
        """
        Export equity curve to CSV file.
        
        Args:
            equity_curve: List of equity point dictionaries
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        if not equity_curve:
            logger.warning("No equity curve data to export")
            return str(filepath)
        
        # Get all unique keys from equity points
        fieldnames = set()
        for point in equity_curve:
            fieldnames.update(point.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(equity_curve)
        
        logger.info(f"Exported {len(equity_curve)} equity points to {filepath}")
        return str(filepath)
    
    def export_to_json(
        self,
        equity_curve: List[Dict[str, Any]],
        filename: str = "equity_curve.json"
    ) -> str:
        """
        Export equity curve to JSON file.
        
        Args:
            equity_curve: List of equity point dictionaries
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(equity_curve, f, indent=2, default=str)
        
        logger.info(f"Exported {len(equity_curve)} equity points to {filepath}")
        return str(filepath)
    
    def export_drawdown_curve(
        self,
        equity_curve: List[Dict[str, Any]],
        filename: str = "drawdown_curve.csv"
    ) -> str:
        """
        Export drawdown curve to CSV file.
        
        Args:
            equity_curve: List of equity point dictionaries
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        if not equity_curve:
            logger.warning("No equity curve data to export")
            return str(filepath)
        
        # Calculate drawdowns
        drawdowns = []
        peak = equity_curve[0].get('equity', 0)
        
        for point in equity_curve:
            equity = point.get('equity', 0)
            
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            
            drawdowns.append({
                'timestamp': point.get('timestamp'),
                'equity': equity,
                'peak': peak,
                'drawdown': drawdown,
                'drawdown_pct': drawdown * 100
            })
        
        # Export to CSV
        fieldnames = ['timestamp', 'equity', 'peak', 'drawdown', 'drawdown_pct']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(drawdowns)
        
        logger.info(f"Exported {len(drawdowns)} drawdown points to {filepath}")
        return str(filepath)
    
    def export_returns_curve(
        self,
        equity_curve: List[Dict[str, Any]],
        filename: str = "returns_curve.csv"
    ) -> str:
        """
        Export returns curve to CSV file.
        
        Args:
            equity_curve: List of equity point dictionaries
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        if len(equity_curve) < 2:
            logger.warning("Not enough equity curve data to calculate returns")
            return str(filepath)
        
        # Calculate returns
        returns = []
        
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1].get('equity', 0)
            curr_equity = equity_curve[i].get('equity', 0)
            
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
            else:
                ret = 0.0
            
            returns.append({
                'timestamp': equity_curve[i].get('timestamp'),
                'equity': curr_equity,
                'return': ret,
                'return_pct': ret * 100
            })
        
        # Export to CSV
        fieldnames = ['timestamp', 'equity', 'return', 'return_pct']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(returns)
        
        logger.info(f"Exported {len(returns)} return points to {filepath}")
        return str(filepath)
    
    def generate_statistics(
        self,
        equity_curve: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate equity curve statistics.
        
        Args:
            equity_curve: List of equity point dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not equity_curve:
            return {}
        
        equity_values = [p.get('equity', 0) for p in equity_curve]
        
        initial_equity = equity_values[0]
        final_equity = equity_values[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate peak and drawdown
        peak = initial_equity
        max_drawdown = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_values)):
            prev_equity = equity_values[i-1]
            curr_equity = equity_values[i]
            
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)
        
        stats = {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'absolute_return': final_equity - initial_equity,
            'peak_equity': peak,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_points': len(equity_curve)
        }
        
        if returns:
            try:
                import numpy as np
                stats['avg_return'] = float(np.mean(returns))
                stats['std_return'] = float(np.std(returns))
                stats['min_return'] = float(np.min(returns))
                stats['max_return'] = float(np.max(returns))
            except ImportError:
                pass
        
        return stats
    
    def export_summary(
        self,
        equity_curve: List[Dict[str, Any]],
        filename: str = "equity_summary.txt"
    ) -> str:
        """
        Export a human-readable equity summary.
        
        Args:
            equity_curve: List of equity point dictionaries
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        stats = self.generate_statistics(equity_curve)
        
        summary = []
        summary.append("=" * 60)
        summary.append("EQUITY CURVE SUMMARY")
        summary.append("=" * 60)
        
        if stats:
            summary.append(f"Initial Equity: ${stats.get('initial_equity', 0):.2f}")
            summary.append(f"Final Equity: ${stats.get('final_equity', 0):.2f}")
            summary.append(f"Total Return: {stats.get('total_return_pct', 0):.2f}%")
            summary.append(f"Absolute Return: ${stats.get('absolute_return', 0):.2f}")
            summary.append(f"Peak Equity: ${stats.get('peak_equity', 0):.2f}")
            summary.append(f"Max Drawdown: {stats.get('max_drawdown_pct', 0):.2f}%")
            summary.append(f"Number of Points: {stats.get('num_points', 0)}")
            
            if 'avg_return' in stats:
                summary.append(f"\nAverage Return: {stats['avg_return']*100:.4f}%")
                summary.append(f"Std Return: {stats['std_return']*100:.4f}%")
                summary.append(f"Min Return: {stats['min_return']*100:.4f}%")
                summary.append(f"Max Return: {stats['max_return']*100:.4f}%")
        else:
            summary.append("No equity curve data available.")
        
        summary.append("=" * 60)
        
        with open(filepath, 'w') as f:
            f.write("\n".join(summary))
        
        logger.info(f"Exported equity summary to {filepath}")
        return str(filepath)
    
    def get_output_dir(self) -> Path:
        """
        Get output directory.
        
        Returns:
            Path to output directory
        """
        return self.output_dir
    
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set output directory.
        
        Args:
            output_dir: New output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory changed to: {self.output_dir}")
