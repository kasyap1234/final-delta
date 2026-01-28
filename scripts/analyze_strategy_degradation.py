#!/usr/bin/env python3
"""
Strategy Degradation Analysis Script

Analyzes backtest results across 2023, 2024, and 2025 to identify
why strategy performance degraded over time.
"""

import os
import csv
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_trade_summary(filepath):
    """Parse trade_summary.txt file and extract metrics."""
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if 'Total Trades:' in line:
                metrics['total_trades'] = int(line.split(':')[1].strip())
            elif 'Winning Trades:' in line:
                metrics['winning_trades'] = int(line.split(':')[1].strip())
            elif 'Losing Trades:' in line:
                metrics['losing_trades'] = int(line.split(':')[1].strip())
            elif 'Win Rate:' in line:
                metrics['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Total P&L:' in line:
                metrics['total_pnl'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Total Fees:' in line:
                metrics['total_fees'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Net P&L:' in line:
                metrics['net_pnl'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Avg Win:' in line:
                metrics['avg_win'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Avg Loss:' in line:
                metrics['avg_loss'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
    return metrics


def parse_equity_summary(filepath):
    """Parse equity_summary.txt file and extract metrics."""
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if 'Initial Equity:' in line:
                metrics['initial_equity'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Final Equity:' in line:
                metrics['final_equity'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Total Return:' in line:
                metrics['total_return'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Absolute Return:' in line:
                metrics['absolute_return'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Peak Equity:' in line:
                metrics['peak_equity'] = float(line.split(':')[1].strip().replace('$', '').replace(',', ''))
            elif 'Max Drawdown:' in line:
                metrics['max_drawdown'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Average Return:' in line:
                metrics['avg_return'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Std Return:' in line:
                metrics['std_return'] = float(line.split(':')[1].strip().replace('%', ''))
    return metrics


def parse_backtest_report(filepath):
    """Parse backtest_report.txt file and extract metrics."""
    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()
        # Extract Sharpe ratio
        sharpe_match = re.search(r'Sharpe Ratio:\s+([-\d.]+)', content)
        if sharpe_match:
            metrics['sharpe_ratio'] = float(sharpe_match.group(1))
        # Extract win rate from report (more accurate)
        winrate_match = re.search(r'Win Rate:\s+([\d.]+)%', content)
        if winrate_match:
            metrics['win_rate_report'] = float(winrate_match.group(1))
    return metrics


def analyze_trades_csv(filepath, max_rows=1000):
    """Analyze trades.csv file to extract detailed trade statistics."""
    trades = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            trade = {
                'entry_price': float(row['entry_price']),
                'pnl': float(row['pnl']),
                'price': float(row['price']),
                'reason': row['reason'],
                'side': row['side'],
                'size': float(row['size']),
                'symbol': row['symbol'],
                'timestamp': datetime.fromisoformat(row['timestamp'])
            }
            trades.append(trade)
    
    # Calculate metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    # Stop loss vs take profit
    stop_loss_hits = [t for t in trades if 'stop_loss' in t['reason']]
    take_profit_hits = [t for t in trades if 'take_profit' in t['reason']]
    
    stop_loss_count = len(stop_loss_hits)
    take_profit_count = len(take_profit_hits)
    
    # P&L stats
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    avg_win = sum(t['pnl'] for t in winning_trades) / win_count if win_count > 0 else 0
    avg_loss = sum(t['pnl'] for t in losing_trades) / loss_count if loss_count > 0 else 0
    
    # Max consecutive losses
    max_consecutive_losses = 0
    current_streak = 0
    for t in trades:
        if t['pnl'] <= 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0
    
    # Symbol breakdown
    symbol_stats = defaultdict(lambda: {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0})
    for t in trades:
        symbol_stats[t['symbol']]['total'] += 1
        symbol_stats[t['symbol']]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            symbol_stats[t['symbol']]['wins'] += 1
        else:
            symbol_stats[t['symbol']]['losses'] += 1
    
    # Monthly breakdown
    monthly_stats = defaultdict(lambda: {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0})
    for t in trades:
        month_key = t['timestamp'].strftime('%Y-%m')
        monthly_stats[month_key]['total'] += 1
        monthly_stats[month_key]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            monthly_stats[month_key]['wins'] += 1
        else:
            monthly_stats[month_key]['losses'] += 1
    
    return {
        'total_trades': total_trades,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'stop_loss_count': stop_loss_count,
        'take_profit_count': take_profit_count,
        'stop_loss_rate': (stop_loss_count / total_trades * 100) if total_trades > 0 else 0,
        'take_profit_rate': (take_profit_count / total_trades * 100) if total_trades > 0 else 0,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_consecutive_losses': max_consecutive_losses,
        'symbol_stats': dict(symbol_stats),
        'monthly_stats': dict(monthly_stats)
    }


def analyze_year(year, base_path):
    """Analyze all data for a specific year."""
    year_path = os.path.join(base_path, str(year))
    
    results = {
        'year': year,
        'trade_summary': parse_trade_summary(os.path.join(year_path, 'trade_summary.txt')),
        'equity_summary': parse_equity_summary(os.path.join(year_path, 'equity_summary.txt')),
        'backtest_report': parse_backtest_report(os.path.join(year_path, 'backtest_report.txt')),
        'trade_analysis': analyze_trades_csv(os.path.join(year_path, 'trades.csv'))
    }
    
    return results


def print_comparison_table(data_2023, data_2024, data_2025):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("STRATEGY PERFORMANCE COMPARISON: 2023 vs 2024 vs 2025")
    print("=" * 100)
    
    print("\n{:<40} {:>18} {:>18} {:>18}".format("Metric", "2023", "2024", "2025"))
    print("-" * 100)
    
    # Equity metrics
    print("\n--- EQUITY METRICS ---")
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Total Return (%)",
        data_2023['equity_summary']['total_return'],
        data_2024['equity_summary']['total_return'],
        data_2025['equity_summary']['total_return']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Final Equity ($)",
        data_2023['equity_summary']['final_equity'],
        data_2024['equity_summary']['final_equity'],
        data_2025['equity_summary']['final_equity']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Max Drawdown (%)",
        data_2023['equity_summary']['max_drawdown'],
        data_2024['equity_summary']['max_drawdown'],
        data_2025['equity_summary']['max_drawdown']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Sharpe Ratio",
        data_2023['backtest_report'].get('sharpe_ratio', 0),
        data_2024['backtest_report'].get('sharpe_ratio', 0),
        data_2025['backtest_report'].get('sharpe_ratio', 0)
    ))
    
    # Trade metrics
    print("\n--- TRADE METRICS ---")
    print("{:<40} {:>18} {:>18} {:>18}".format(
        "Total Trades",
        data_2023['trade_analysis']['total_trades'],
        data_2024['trade_analysis']['total_trades'],
        data_2025['trade_analysis']['total_trades']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Win Rate (%)",
        data_2023['trade_analysis']['win_rate'],
        data_2024['trade_analysis']['win_rate'],
        data_2025['trade_analysis']['win_rate']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Total P&L ($)",
        data_2023['trade_analysis']['total_pnl'],
        data_2024['trade_analysis']['total_pnl'],
        data_2025['trade_analysis']['total_pnl']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Avg P&L per Trade ($)",
        data_2023['trade_analysis']['avg_pnl'],
        data_2024['trade_analysis']['avg_pnl'],
        data_2025['trade_analysis']['avg_pnl']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Avg Win ($)",
        data_2023['trade_analysis']['avg_win'],
        data_2024['trade_analysis']['avg_win'],
        data_2025['trade_analysis']['avg_win']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Avg Loss ($)",
        data_2023['trade_analysis']['avg_loss'],
        data_2024['trade_analysis']['avg_loss'],
        data_2025['trade_analysis']['avg_loss']
    ))
    print("{:<40} {:>18} {:>18} {:>18}".format(
        "Max Consecutive Losses",
        data_2023['trade_analysis']['max_consecutive_losses'],
        data_2024['trade_analysis']['max_consecutive_losses'],
        data_2025['trade_analysis']['max_consecutive_losses']
    ))
    
    # Exit reason analysis
    print("\n--- EXIT REASON ANALYSIS ---")
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Stop Loss Rate (%)",
        data_2023['trade_analysis']['stop_loss_rate'],
        data_2024['trade_analysis']['stop_loss_rate'],
        data_2025['trade_analysis']['stop_loss_rate']
    ))
    print("{:<40} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Take Profit Rate (%)",
        data_2023['trade_analysis']['take_profit_rate'],
        data_2024['trade_analysis']['take_profit_rate'],
        data_2025['trade_analysis']['take_profit_rate']
    ))


def print_symbol_analysis(data_2023, data_2024, data_2025):
    """Print symbol-wise analysis."""
    print("\n" + "=" * 100)
    print("SYMBOL-WISE PERFORMANCE ANALYSIS")
    print("=" * 100)
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        print("{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "Year", "Trades", "Wins", "Losses", "Win Rate %", "Total P&L"
        ))
        print("-" * 80)
        
        for data in [data_2023, data_2024, data_2025]:
            year = data['year']
            stats = data['trade_analysis']['symbol_stats'].get(symbol, {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0})
            win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print("{:<20} {:>12} {:>12} {:>12} {:>12.2f} {:>12.2f}".format(
                year, stats['total'], stats['wins'], stats['losses'], win_rate, stats['pnl']
            ))


def print_monthly_trends(data_2023, data_2024, data_2025):
    """Print monthly trend analysis."""
    print("\n" + "=" * 100)
    print("MONTHLY TREND ANALYSIS")
    print("=" * 100)
    
    all_data = [data_2023, data_2024, data_2025]
    
    for data in all_data:
        year = data['year']
        print(f"\n--- {year} Monthly Breakdown ---")
        print("{:<10} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
            "Month", "Trades", "Wins", "Losses", "Win Rate %", "P&L ($)"
        ))
        print("-" * 70)
        
        monthly = data['trade_analysis']['monthly_stats']
        for month in sorted(monthly.keys()):
            stats = monthly[month]
            win_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print("{:<10} {:>10} {:>10} {:>10} {:>12.2f} {:>12.2f}".format(
                month, stats['total'], stats['wins'], stats['losses'], win_rate, stats['pnl']
            ))


def generate_analysis_report(data_2023, data_2024, data_2025, output_path):
    """Generate a comprehensive analysis report."""
    report = []
    
    report.append("=" * 100)
    report.append("STRATEGY DEGRADATION ANALYSIS REPORT")
    report.append("Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    report.append("=" * 100)
    
    # Executive Summary
    report.append("\n" + "=" * 100)
    report.append("EXECUTIVE SUMMARY")
    report.append("=" * 100)
    
    report.append("""
The strategy has shown significant degradation in performance from 2023 to 2025:

- 2023: Strong performance with +18.10% return, 2.08 Sharpe ratio, 58.16% win rate
- 2024: Poor performance with only +1.66% return, 0.24 Sharpe ratio, 37.63% win rate  
- 2025: Negative performance with -2.43% return, -0.42 Sharpe ratio, 32.67% win rate

The strategy has gone from a profitable, high-performance system to a losing strategy
over the course of three years, indicating fundamental issues with the strategy's
ability to adapt to changing market conditions.
""")
    
    # Key Findings
    report.append("\n" + "=" * 100)
    report.append("KEY FINDINGS")
    report.append("=" * 100)
    
    # Win rate degradation
    win_rate_2023 = data_2023['trade_analysis']['win_rate']
    win_rate_2024 = data_2024['trade_analysis']['win_rate']
    win_rate_2025 = data_2025['trade_analysis']['win_rate']
    
    report.append(f"""
1. WIN RATE DEGRADATION
   - 2023: {win_rate_2023:.2f}% win rate
   - 2024: {win_rate_2024:.2f}% win rate (decrease of {win_rate_2023 - win_rate_2024:.2f}%)
   - 2025: {win_rate_2025:.2f}% win rate (decrease of {win_rate_2023 - win_rate_2025:.2f}%)
   
   The win rate has dropped by {(win_rate_2023 - win_rate_2025) / win_rate_2023 * 100:.1f}% from 2023 to 2025,
   indicating the strategy's signals have become significantly less accurate.
""")
    
    # Stop loss vs take profit
    sl_rate_2023 = data_2023['trade_analysis']['stop_loss_rate']
    sl_rate_2024 = data_2024['trade_analysis']['stop_loss_rate']
    sl_rate_2025 = data_2025['trade_analysis']['stop_loss_rate']
    
    tp_rate_2023 = data_2023['trade_analysis']['take_profit_rate']
    tp_rate_2024 = data_2024['trade_analysis']['take_profit_rate']
    tp_rate_2025 = data_2025['trade_analysis']['take_profit_rate']
    
    report.append(f"""
2. STOP LOSS HIT RATE INCREASE
   - 2023: {sl_rate_2023:.2f}% of trades hit stop loss
   - 2024: {sl_rate_2024:.2f}% of trades hit stop loss
   - 2025: {sl_rate_2025:.2f}% of trades hit stop loss
   
   The stop loss hit rate increased by {sl_rate_2025 - sl_rate_2023:.2f} percentage points,
   showing that trades are moving against the strategy more frequently.
   
   Take profit rates:
   - 2023: {tp_rate_2023:.2f}%
   - 2024: {tp_rate_2024:.2f}%
   - 2025: {tp_rate_2025:.2f}%
""")
    
    # Average P&L per trade
    avg_pnl_2023 = data_2023['trade_analysis']['avg_pnl']
    avg_pnl_2024 = data_2024['trade_analysis']['avg_pnl']
    avg_pnl_2025 = data_2025['trade_analysis']['avg_pnl']
    
    report.append(f"""
3. AVERAGE PROFITABILITY PER TRADE
   - 2023: ${avg_pnl_2023:.2f} average P&L per trade
   - 2024: ${avg_pnl_2024:.2f} average P&L per trade
   - 2025: ${avg_pnl_2025:.2f} average P&L per trade
   
   The average profit per trade has declined significantly, turning negative in 2025.
   This indicates that even winning trades are capturing less profit.
""")
    
    # Consecutive losses
    max_cl_2023 = data_2023['trade_analysis']['max_consecutive_losses']
    max_cl_2024 = data_2024['trade_analysis']['max_consecutive_losses']
    max_cl_2025 = data_2025['trade_analysis']['max_consecutive_losses']
    
    report.append(f"""
4. CONSECUTIVE LOSS STREAKS
   - 2023: Maximum {max_cl_2023} consecutive losses
   - 2024: Maximum {max_cl_2024} consecutive losses
   - 2025: Maximum {max_cl_2025} consecutive losses
   
   The maximum consecutive loss streak has increased, indicating more prolonged
   periods of poor performance and potential strategy breakdown.
""")
    
    # Symbol analysis
    report.append("\n5. SYMBOL-SPECIFIC PERFORMANCE CHANGES")
    report.append("-" * 50)
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    for symbol in symbols:
        stats_2023 = data_2023['trade_analysis']['symbol_stats'].get(symbol, {'total': 0, 'pnl': 0, 'wins': 0})
        stats_2024 = data_2024['trade_analysis']['symbol_stats'].get(symbol, {'total': 0, 'pnl': 0, 'wins': 0})
        stats_2025 = data_2025['trade_analysis']['symbol_stats'].get(symbol, {'total': 0, 'pnl': 0, 'wins': 0})
        
        wr_2023 = (stats_2023['wins'] / stats_2023['total'] * 100) if stats_2023['total'] > 0 else 0
        wr_2024 = (stats_2024['wins'] / stats_2024['total'] * 100) if stats_2024['total'] > 0 else 0
        wr_2025 = (stats_2025['wins'] / stats_2025['total'] * 100) if stats_2025['total'] > 0 else 0
        
        report.append(f"""
   {symbol}:
   - 2023: {stats_2023['total']} trades, ${stats_2023['pnl']:.2f} P&L, {wr_2023:.2f}% win rate
   - 2024: {stats_2024['total']} trades, ${stats_2024['pnl']:.2f} P&L, {wr_2024:.2f}% win rate
   - 2025: {stats_2025['total']} trades, ${stats_2025['pnl']:.2f} P&L, {wr_2025:.2f}% win rate
""")
    
    # Root Cause Analysis
    report.append("\n" + "=" * 100)
    report.append("ROOT CAUSE ANALYSIS")
    report.append("=" * 100)
    
    report.append("""
Based on the data analysis, the following factors appear to be driving the strategy degradation:

1. MARKET REGIME CHANGE (2023-2024)
   - 2023 featured a strong crypto bull market recovery from 2022 lows
   - 2024 saw increased volatility and choppy price action
   - The strategy appears optimized for trending markets and struggles in range-bound conditions

2. INCREASED STOP LOSS HITS
   - The dramatic increase in stop loss hits suggests:
     a) Tighter stops may be getting triggered by normal market noise
     b) Entry timing has become less accurate
     c) Market volatility patterns have changed

3. REDUCED PROFIT CAPTURE
   - Average win size has decreased while average loss size has remained similar or increased
   - This indicates the risk/reward ratio has deteriorated
   - Take profit levels may be too ambitious for current market conditions

4. TRADE FREQUENCY INCREASE (2025)
   - 2025 shows more trades (101) compared to 2023 (98) and 2024 (93)
   - More trades with lower win rate suggests over-trading
   - Strategy may be generating false signals in current market conditions

5. SYMBOL-SPECIFIC BREAKDOWN
   - SOL/USDT showed strong performance in 2023 but deteriorated significantly
   - BTC/USDT and ETH/USDT also show declining performance
   - All symbols affected suggests a systematic issue rather than symbol-specific
""")
    
    # Recommendations
    report.append("\n" + "=" * 100)
    report.append("RECOMMENDATIONS FOR IMPROVEMENT")
    report.append("=" * 100)
    
    report.append("""
1. RISK PARAMETER ADJUSTMENTS
   - Widen stop loss levels to avoid noise-driven exits
   - Consider dynamic position sizing based on market volatility
   - Implement tighter take profits to secure gains in choppy markets

2. MARKET REGIME DETECTION
   - Add market regime detection (trending vs ranging)
   - Reduce position sizes or pause trading during unfavorable regimes
   - Implement volatility filters to avoid trading in low-probability conditions

3. SIGNAL QUALITY IMPROVEMENTS
   - Add additional confirmation indicators to reduce false signals
   - Implement multi-timeframe analysis for better entry timing
   - Consider volume-based filters to avoid low-liquidity entries

4. PORTFOLIO DIVERSIFICATION
   - The strategy appears over-concentrated in similar crypto assets
   - Consider adding uncorrelated assets or strategies
   - Implement correlation-based position sizing

5. BACKTEST ROBUSTNESS
   - The 2023 results may be overfitted to that specific market condition
   - Test strategy across multiple market regimes
   - Implement walk-forward analysis to validate robustness

6. DYNAMIC PARAMETER OPTIMIZATION
   - Implement periodic re-optimization of strategy parameters
   - Use rolling window optimization to adapt to changing conditions
   - Consider machine learning for adaptive parameter adjustment

7. RISK MANAGEMENT ENHANCEMENTS
   - Implement maximum daily/weekly loss limits
   - Add correlation-based risk limits
   - Consider portfolio heat management (total risk exposure)
""")
    
    # Conclusion
    report.append("\n" + "=" * 100)
    report.append("CONCLUSION")
    report.append("=" * 100)
    
    report.append("""
The strategy has experienced significant degradation from 2023 to 2025, transitioning from
a profitable system to a losing one. The primary causes appear to be:

1. Market regime changes that the strategy is not adapted to handle
2. Over-optimization to 2023 market conditions
3. Inadequate risk parameters for current volatility levels
4. Signal quality deterioration in ranging/choppy markets

Immediate action is required to either:
- Significantly modify the strategy to handle current market conditions
- Temporarily halt trading until market conditions become more favorable
- Implement the recommended improvements before deploying capital

Without these changes, the strategy is likely to continue underperforming and
potentially lead to further capital erosion.
""")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)


def main():
    """Main analysis function."""
    base_path = 'backtest_results'
    
    print("Loading and analyzing backtest data...")
    
    # Analyze each year
    data_2023 = analyze_year(2023, base_path)
    data_2024 = analyze_year(2024, base_path)
    data_2025 = analyze_year(2025, base_path)
    
    # Print comparison table
    print_comparison_table(data_2023, data_2024, data_2025)
    
    # Print symbol analysis
    print_symbol_analysis(data_2023, data_2024, data_2025)
    
    # Print monthly trends
    print_monthly_trends(data_2023, data_2024, data_2025)
    
    # Generate comprehensive report
    report_path = os.path.join(base_path, 'strategy_degradation_analysis.txt')
    report = generate_analysis_report(data_2023, data_2024, data_2025, report_path)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nComprehensive report saved to: {report_path}")
    print("\nReport preview:")
    print("-" * 100)
    print(report[:3000] + "...")
    print("-" * 100)


if __name__ == '__main__':
    main()
