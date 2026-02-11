#!/usr/bin/env python3
"""
Compute benchmark comparisons and walk-forward analysis for the redesigned strategy.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def read_price_data(csv_path: str) -> List[Dict]:
    """Read price data from CSV file."""
    prices = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append({
                'timestamp': row['timestamp'],
                'close': float(row['close'])
            })
    return prices


def compute_buy_and_hold_return(prices: List[Dict]) -> float:
    """Compute buy-and-hold return from price data."""
    if len(prices) < 2:
        return 0.0
    start_price = prices[0]['close']
    end_price = prices[-1]['close']
    return (end_price - start_price) / start_price


def compute_equal_weight_bnh_return(btc_prices: List[Dict], eth_prices: List[Dict], sol_prices: List[Dict]) -> float:
    """Compute equal-weight buy-and-hold return for BTC/ETH/SOL portfolio."""
    btc_return = compute_buy_and_hold_return(btc_prices)
    eth_return = compute_buy_and_hold_return(eth_prices)
    sol_return = compute_buy_and_hold_return(sol_prices)
    return (btc_return + eth_return + sol_return) / 3.0


def read_backtest_results(report_path: str) -> Dict:
    """Read backtest results from JSON report."""
    with open(report_path, 'r') as f:
        return json.load(f)


def compute_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio from returns."""
    if len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 24 * 4)  # Annualized for 15m


def compute_max_drawdown(equity_curve: List[Dict]) -> float:
    """Compute maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    equity_values = [point['equity'] for point in equity_curve]
    peak = equity_values[0]
    max_dd = 0.0
    for value in equity_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main():
    print("=" * 80)
    print("BENCHMARK COMPARISON AND WALK-FORWARD ANALYSIS")
    print("=" * 80)

    # Data directory
    data_dir = Path("data/backtest")
    results_dir = Path("backtest_results")

    # Years to analyze
    years = [2023, 2024, 2025]

    # Store results
    all_results = {}

    for year in years:
        print(f"\n{'=' * 80}")
        print(f"YEAR {year} ANALYSIS")
        print(f"{'=' * 80}")

        # Read price data
        btc_prices = read_price_data(str(data_dir / f"BTC_USDT_{year}_15m.csv"))
        eth_prices = read_price_data(str(data_dir / f"ETH_USDT_{year}_15m.csv"))
        sol_prices = read_price_data(str(data_dir / f"SOL_USDT_{year}_15m.csv"))

        # Compute benchmark returns
        btc_bnh = compute_buy_and_hold_return(btc_prices)
        eth_bnh = compute_buy_and_hold_return(eth_prices)
        sol_bnh = compute_buy_and_hold_return(sol_prices)
        equal_weight_bnh = compute_equal_weight_bnh_return(btc_prices, eth_prices, sol_prices)

        print(f"\nBenchmark (Buy-and-Hold) Returns:")
        print(f"  BTC: {btc_bnh:.2%}")
        print(f"  ETH: {eth_bnh:.2%}")
        print(f"  SOL: {sol_bnh:.2%}")
        print(f"  Equal-Weight Portfolio: {equal_weight_bnh:.2%}")

        # Read backtest results
        if year == 2023:
            result_dir = results_dir / "2023"
        elif year == 2024:
            result_dir = results_dir / "2024_test"
        else:
            result_dir = results_dir / "2025_test"

        report_path = result_dir / "backtest_report.json"
        if not report_path.exists():
            print(f"  Warning: Backtest report not found for {year}")
            continue

        backtest_results = read_backtest_results(str(report_path))

        # Extract strategy metrics
        strategy_return = backtest_results.get('total_return', 0.0)
        strategy_sharpe = backtest_results.get('sharpe_ratio', 0.0)
        strategy_max_dd = backtest_results.get('max_drawdown', 0.0)
        strategy_win_rate = backtest_results.get('win_rate', 0.0)
        strategy_trades = backtest_results.get('total_trades', 0)

        print(f"\nStrategy Performance:")
        print(f"  Total Return: {strategy_return:.2%}")
        print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"  Max Drawdown: {strategy_max_dd:.2%}")
        print(f"  Win Rate: {strategy_win_rate:.2%}")
        print(f"  Total Trades: {strategy_trades}")

        # Compute relative performance
        relative_return = strategy_return - equal_weight_bnh
        print(f"\nRelative Performance vs Equal-Weight B&H:")
        print(f"  Return Difference: {relative_return:+.2%}")

        # Store results
        all_results[year] = {
            'benchmark': {
                'btc': btc_bnh,
                'eth': eth_bnh,
                'sol': sol_bnh,
                'equal_weight': equal_weight_bnh
            },
            'strategy': {
                'return': strategy_return,
                'sharpe': strategy_sharpe,
                'max_drawdown': strategy_max_dd,
                'win_rate': strategy_win_rate,
                'trades': strategy_trades
            },
            'relative': {
                'return_diff': relative_return
            }
        }

    # Walk-forward analysis
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD ANALYSIS")
    print(f"{'=' * 80}")

    print("\nYear-over-Year Performance:")
    print(f"{'Year':<6} {'Strategy Return':<18} {'Benchmark Return':<18} {'Relative':<12} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 80)

    for year in years:
        if year not in all_results:
            continue
        r = all_results[year]
        print(f"{year:<6} {r['strategy']['return']:>16.2%} {r['benchmark']['equal_weight']:>16.2%} {r['relative']['return_diff']:>10.2%} {r['strategy']['sharpe']:>8.2f} {r['strategy']['max_drawdown']:>8.2%}")

    # Stability analysis
    print(f"\n{'=' * 80}")
    print("STABILITY ANALYSIS")
    print(f"{'=' * 80}")

    returns = [all_results[y]['strategy']['return'] for y in years if y in all_results]
    sharpe_ratios = [all_results[y]['strategy']['sharpe'] for y in years if y in all_results]
    max_drawdowns = [all_results[y]['strategy']['max_drawdown'] for y in years if y in all_results]
    win_rates = [all_results[y]['strategy']['win_rate'] for y in years if y in all_results]
    trade_counts = [all_results[y]['strategy']['trades'] for y in years if y in all_results]

    if len(returns) > 0:
        print(f"\nStrategy Metrics Statistics:")
        print(f"  Return - Mean: {np.mean(returns):.2%}, Std: {np.std(returns):.2%}, Min: {np.min(returns):.2%}, Max: {np.max(returns):.2%}")
        print(f"  Sharpe - Mean: {np.mean(sharpe_ratios):.2f}, Std: {np.std(sharpe_ratios):.2f}")
        print(f"  Max DD - Mean: {np.mean(max_drawdowns):.2%}, Std: {np.std(max_drawdowns):.2%}")
        print(f"  Win Rate - Mean: {np.mean(win_rates):.2%}, Std: {np.std(win_rates):.2%}")
        print(f"  Trade Count - Mean: {np.mean(trade_counts):.0f}, Std: {np.std(trade_counts):.0f}")

    # Adversarial scenario analysis
    print(f"\n{'=' * 80}")
    print("ADVERSARIAL SCENARIO ANALYSIS (2023)")
    print(f"{'=' * 80}")

    scenarios = {
        'baseline': results_dir / "2023",
        'pessimistic': results_dir / "pessimistic_2023",
        'sensitivity': results_dir / "sensitivity_2023"
    }

    print(f"\n{'Scenario':<15} {'Return':<12} {'Sharpe':<10} {'Max DD':<10} {'Win Rate':<12} {'Trades':<10}")
    print("-" * 80)

    for scenario_name, scenario_dir in scenarios.items():
        report_path = scenario_dir / "backtest_report.json"
        if not report_path.exists():
            continue

        results = read_backtest_results(str(report_path))
        print(f"{scenario_name:<15} {results['total_return']:>10.2%} {results['sharpe_ratio']:>8.2f} {results['max_drawdown']:>8.2%} {results['win_rate']:>10.2%} {results['total_trades']:>8.0f}")

    # Degradation analysis
    print(f"\nDegradation vs Baseline:")
    baseline = read_backtest_results(str(scenarios['baseline'] / "backtest_report.json"))

    for scenario_name, scenario_dir in scenarios.items():
        if scenario_name == 'baseline':
            continue

        report_path = scenario_dir / "backtest_report.json"
        if not report_path.exists():
            continue

        results = read_backtest_results(str(report_path))

        return_degradation = results['total_return'] - baseline['total_return']
        sharpe_degradation = results['sharpe_ratio'] - baseline['sharpe_ratio']
        dd_degradation = results['max_drawdown'] - baseline['max_drawdown']

        print(f"\n{scenario_name}:")
        print(f"  Return: {return_degradation:+.2%} (from {baseline['total_return']:.2%} to {results['total_return']:.2%})")
        print(f"  Sharpe: {sharpe_degradation:+.2f} (from {baseline['sharpe_ratio']:.2f} to {results['sharpe_ratio']:.2f})")
        print(f"  Max DD: {dd_degradation:+.2%} (from {baseline['max_drawdown']:.2%} to {results['max_drawdown']:.2%})")

    # Save results to JSON
    output_path = Path("backtest_results/walk_forward_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Analysis complete. Results saved to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
