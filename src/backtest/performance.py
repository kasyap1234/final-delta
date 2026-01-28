"""
Performance calculator module for backtesting.

This module provides performance metrics calculation for backtesting system.
"""

from typing import Dict, List, Optional, Any
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("NumPy not available, some calculations will be limited")

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """
    Calculate performance metrics for backtesting results.
    
    This class handles:
    - Return metrics (total, average, etc.)
    - Risk metrics (Sharpe, Sortino, drawdown)
    - Trade statistics (win rate, profit factor)
    - Comprehensive performance reports
    """
    
    def __init__(self, equity_curve: List[float], trade_history: List[Dict[str, Any]]):
        """
        Initialize performance calculator.
        
        Args:
            equity_curve: List of equity values over time
            trade_history: List of trade records
        """
        self.equity_curve = equity_curve
        self.trade_history = trade_history
        
        logger.info(
            f"PerformanceCalculator initialized: {len(equity_curve)} equity points, "
            f"{len(trade_history)} trades"
        )
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        metrics = {}
        
        # Return metrics
        metrics.update(self.calculate_return_metrics())
        
        # Risk metrics
        metrics.update(self.calculate_risk_metrics())
        
        # Trade metrics
        metrics.update(self.calculate_trade_metrics())
        
        # Drawdown metrics
        metrics.update(self.calculate_drawdown_metrics())
        
        return metrics
    
    def calculate_return_metrics(self) -> Dict[str, Any]:
        """
        Calculate return metrics.
        
        Returns:
            Dictionary with return metrics
        """
        if not self.equity_curve:
            return {}
        
        initial_equity = self.equity_curve[0]
        final_equity = self.equity_curve[-1]
        
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate returns
        returns = self._calculate_returns()
        
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'absolute_return': final_equity - initial_equity
        }
        
        if returns and HAS_NUMPY:
            metrics['avg_return'] = float(np.mean(returns))
            metrics['std_return'] = float(np.std(returns))
            metrics['median_return'] = float(np.median(returns))
            metrics['min_return'] = float(np.min(returns))
            metrics['max_return'] = float(np.max(returns))
        
        return metrics
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        if not self.equity_curve:
            return {}
        
        returns = self._calculate_returns()
        
        metrics = {}
        
        if returns and HAS_NUMPY:
            # Sharpe ratio (annualized)
            # Assuming 15m candles, 96 per day, 35040 per year
            periods_per_year = 35040
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                sharpe_ratio = (avg_return / std_return) * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0.0
            
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns)
                if downside_deviation > 0:
                    sortino_ratio = (avg_return / downside_deviation) * np.sqrt(periods_per_year)
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = float('inf') if avg_return > 0 else 0.0
            
            metrics['sortino_ratio'] = sortino_ratio
        
        return metrics
    
    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """
        Calculate trade statistics.
        
        Returns:
            Dictionary with trade metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Extract P&L from trades
        trade_pnls = []
        for trade in self.trade_history:
            pnl = trade.get('realized_pnl', 0)
            if pnl != 0:  # Only count completed trades
                trade_pnls.append(pnl)
        
        if not trade_pnls:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        total_trades = len(trade_pnls)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor
        }
    
    def calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """
        Calculate drawdown metrics.
        
        Returns:
            Dictionary with drawdown metrics
        """
        if not self.equity_curve:
            return {}
        
        peak = self.equity_curve[0]
        max_drawdown = 0.0
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        
        drawdowns = []
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
                current_drawdown_duration = 0
            else:
                drawdown = (peak - equity) / peak if peak > 0 else 0.0
                drawdowns.append(drawdown)
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                
                current_drawdown_duration += 1
                if current_drawdown_duration > max_drawdown_duration:
                    max_drawdown_duration = current_drawdown_duration
        
        avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_pct': avg_drawdown * 100,
            'max_drawdown_duration': max_drawdown_duration,
            'num_drawdowns': len(drawdowns)
        }
    
    def calculate_calmar_ratio(self) -> Dict[str, Any]:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Returns:
            Dictionary with Calmar ratio
        """
        return_metrics = self.calculate_return_metrics()
        drawdown_metrics = self.calculate_drawdown_metrics()
        
        annual_return = return_metrics.get('total_return', 0.0)
        max_drawdown = drawdown_metrics.get('max_drawdown', 0.0)
        
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = float('inf') if annual_return > 0 else 0.0
        
        return {
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_returns(self) -> List[float]:
        """
        Calculate returns from equity curve.
        
        Returns:
            List of returns
        """
        if len(self.equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1]
            curr_equity = self.equity_curve[i]
            
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return returns
    
    def generate_summary(self) -> str:
        """
        Generate a human-readable performance summary.
        
        Returns:
            Formatted summary string
        """
        metrics = self.calculate_all_metrics()
        
        summary = []
        summary.append("=" * 60)
        summary.append("BACKTEST PERFORMANCE SUMMARY")
        summary.append("=" * 60)
        
        # Return metrics
        summary.append("\n--- Return Metrics ---")
        summary.append(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        summary.append(f"Initial Equity: ${metrics.get('initial_equity', 0):.2f}")
        summary.append(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
        summary.append(f"Absolute Return: ${metrics.get('absolute_return', 0):.2f}")
        
        if 'avg_return' in metrics:
            summary.append(f"Avg Return: {metrics['avg_return']*100:.4f}%")
            summary.append(f"Std Return: {metrics['std_return']*100:.4f}%")
        
        # Risk metrics
        summary.append("\n--- Risk Metrics ---")
        if 'sharpe_ratio' in metrics:
            summary.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        if 'sortino_ratio' in metrics:
            summary.append(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        
        summary.append(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        summary.append(f"Avg Drawdown: {metrics.get('avg_drawdown_pct', 0):.2f}%")
        
        # Trade metrics
        summary.append("\n--- Trade Metrics ---")
        summary.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        summary.append(f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
        summary.append(f"Avg Win: ${metrics.get('avg_win', 0):.2f}")
        summary.append(f"Avg Loss: ${metrics.get('avg_loss', 0):.2f}")
        summary.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)
