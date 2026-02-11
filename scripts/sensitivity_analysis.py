#!/usr/bin/env python3
"""
Sensitivity analysis script to test parameter variations.
"""

import yaml
import subprocess
import json
from pathlib import Path
import shutil

def create_sensitivity_config(base_config_path: str, output_path: str, param_changes: dict):
    """Create a modified config for sensitivity testing."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply parameter changes
    for key, value in param_changes.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def run_backtest(config_path: str, output_dir: str):
    """Run a backtest with the given config."""
    # Clean output directory if it exists
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    
    cmd = ["python3", "backtest_main.py", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Load results
    report_path = Path(output_dir) / "backtest_report.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            return json.load(f)
    return None

def main():
    """Run sensitivity analysis."""
    base_config = "config/backtest_2024_fixed.yaml"
    
    # Test variations
    variations = [
        {
            "name": "baseline",
            "changes": {}
        },
        {
            "name": "wider_sl_+10%",
            "changes": {
                "trading_bot.strategy.atr_multiplier": 2.2  # 2.0 * 1.1
            }
        },
        {
            "name": "wider_sl_+20%",
            "changes": {
                "trading_bot.strategy.atr_multiplier": 2.4  # 2.0 * 1.2
            }
        },
        {
            "name": "higher_entry_threshold_+10%",
            "changes": {
                "trading_bot.strategy.rsi_long_threshold": 66.0,  # 60.0 * 1.1
                "trading_bot.strategy.rsi_short_threshold": 36.0,  # 40.0 * 0.9
            }
        },
        {
            "name": "higher_entry_threshold_+20%",
            "changes": {
                "trading_bot.strategy.rsi_long_threshold": 72.0,  # 60.0 * 1.2
                "trading_bot.strategy.rsi_short_threshold": 32.0,  # 40.0 * 0.8
            }
        },
    ]
    
    results = []
    
    for variation in variations:
        print(f"\n{'='*60}")
        print(f"Running: {variation['name']}")
        print(f"{'='*60}")
        
        if variation['changes']:
            config_path = f"config/sensitivity_{variation['name']}.yaml"
            output_dir = f"backtest_results/sensitivity_{variation['name']}"
            create_sensitivity_config(base_config, config_path, variation['changes'])
        else:
            config_path = base_config
            output_dir = "backtest_results/2024"  # Use existing baseline
        
        result = run_backtest(config_path, output_dir)
        
        if result:
            results.append({
                "name": variation['name'],
                "total_return": result.get('total_return', 0),
                "sharpe_ratio": result.get('sharpe_ratio', 0),
                "max_drawdown": result.get('max_drawdown', 0),
                "win_rate": result.get('win_rate', 0),
                "total_trades": result.get('total_trades', 0),
                "total_fees": result.get('total_fees', 0),
            })
            
            print(f"Total Return: {result.get('total_return', 0):.2%}")
            print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {result.get('win_rate', 0):.2%}")
            print(f"Total Trades: {result.get('total_trades', 0)}")
        else:
            print(f"Failed to get results")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Name':<30} {'Return':<10} {'Sharpe':<10} {'DD':<10} {'Win Rate':<10} {'Trades':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30} {r['total_return']:>9.2%} {r['sharpe_ratio']:>9.2f} {r['max_drawdown']:>9.2%} {r['win_rate']:>9.2%} {r['total_trades']:>9}")

if __name__ == "__main__":
    main()
