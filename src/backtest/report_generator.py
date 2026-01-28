"""
Report generator module for backtesting.

This module provides HTML and JSON report generation for backtesting system.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive backtest reports.
    
    This class handles:
    - Generating HTML performance reports
    - Generating JSON reports
    - Creating visual summaries
    """
    
    def __init__(self, output_dir: str = "backtest_results"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ReportGenerator initialized: output_dir={self.output_dir}")
    
    def generate_html_report(
        self,
        results: Dict[str, Any],
        filename: str = "backtest_report.html"
    ) -> str:
        """
        Generate an HTML performance report.
        
        Args:
            results: Backtest results dictionary
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        html = self._generate_html_content(results)
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated HTML report: {filepath}")
        return str(filepath)
    
    def generate_json_report(
        self,
        results: Dict[str, Any],
        filename: str = "backtest_report.json"
    ) -> str:
        """
        Generate a JSON performance report.
        
        Args:
            results: Backtest results dictionary
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Generated JSON report: {filepath}")
        return str(filepath)
    
    def _generate_html_content(self, results: Dict[str, Any]) -> str:
        """
        Generate HTML content for the report.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            HTML content string
        """
        performance = results.get('performance', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric.positive {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric.negative {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Performance Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Period:</strong> {results.get('start_date', 'N/A')} to {results.get('end_date', 'N/A')}</p>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="metric {'positive' if results.get('total_return', 0) >= 0 else 'negative'}">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{results.get('total_return', 0):.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Final Equity</div>
                <div class="metric-value">${results.get('final_equity', 0):.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{results.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric negative">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{results.get('max_drawdown', 0):.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{results.get('win_rate', 0):.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{results.get('total_trades', 0)}</div>
            </div>
        </div>
        
        <h2>Return Metrics</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Initial Balance</td>
                    <td>${results.get('initial_balance', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Final Equity</td>
                    <td>${results.get('final_equity', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td>{results.get('total_return', 0):.2%}</td>
                </tr>
                <tr>
                    <td>Absolute Return</td>
                    <td>${results.get('final_equity', 0) - results.get('initial_balance', 0):.2f}</td>
                </tr>
            </table>
        </div>
        
        <h2>Risk Metrics</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{results.get('sharpe_ratio', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>{results.get('max_drawdown', 0):.2%}</td>
                </tr>
            </table>
        </div>
        
        <h2>Trade Metrics</h2>
        <div class="section">
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{results.get('total_trades', 0)}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{results.get('win_rate', 0):.2%}</td>
                </tr>
                <tr>
                    <td>Total Fees</td>
                    <td>${results.get('total_fees', 0):.2f}</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by Trading Bot Backtesting System</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def generate_text_report(
        self,
        results: Dict[str, Any],
        filename: str = "backtest_report.txt"
    ) -> str:
        """
        Generate a text performance report.
        
        Args:
            results: Backtest results dictionary
            filename: Output filename
            
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST PERFORMANCE REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Period: {results.get('start_date', 'N/A')} to {results.get('end_date', 'N/A')}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Total Return: {results.get('total_return', 0):.2%}")
        lines.append(f"Final Equity: ${results.get('final_equity', 0):.2f}")
        lines.append(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        lines.append(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        lines.append(f"Win Rate: {results.get('win_rate', 0):.2%}")
        lines.append(f"Total Trades: {results.get('total_trades', 0)}")
        lines.append(f"Total Fees: ${results.get('total_fees', 0):.2f}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("RETURN METRICS")
        lines.append("-" * 70)
        lines.append(f"Initial Balance: ${results.get('initial_balance', 0):.2f}")
        lines.append(f"Final Equity: ${results.get('final_equity', 0):.2f}")
        lines.append(f"Total Return: {results.get('total_return', 0):.2%}")
        lines.append(f"Absolute Return: ${results.get('final_equity', 0) - results.get('initial_balance', 0):.2f}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("RISK METRICS")
        lines.append("-" * 70)
        lines.append(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        lines.append(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("TRADE METRICS")
        lines.append("-" * 70)
        lines.append(f"Total Trades: {results.get('total_trades', 0)}")
        lines.append(f"Win Rate: {results.get('win_rate', 0):.2%}")
        lines.append(f"Total Fees: ${results.get('total_fees', 0):.2f}")
        lines.append("")
        
        lines.append("=" * 70)
        
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Generated text report: {filepath}")
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
