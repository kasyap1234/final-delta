#!/usr/bin/env python3
"""
Multi-year backtest runner for 2023, 2024, 2025
"""

import subprocess
import json
import os
from datetime import datetime

# Year configurations
years = {
    "2023": {
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-12-31T23:59:59Z",
        "data_files": {
            "BTC/USDT": "data/backtest/BTC_USDT_2023_15m.csv",
            "ETH/USDT": "data/backtest/ETH_USDT_2023_15m.csv",
        },
    },
    "2024": {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-12-31T23:59:59Z",
        "data_files": {
            "BTC/USDT": "data/backtest/BTC_USDT_2024_15m.csv",
            "ETH/USDT": "data/backtest/ETH_USDT_2024_15m.csv",
        },
    },
    "2025": {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-12-31T23:59:59Z",
        "data_files": {
            "BTC/USDT": "data/backtest/BTC_USDT_2025_15m.csv",
            "ETH/USDT": "data/backtest/ETH_USDT_2025_15m.csv",
        },
    },
}

results = {}

for year, config in years.items():
    print(f"\n{'=' * 60}")
    print(f"BACKTESTING YEAR: {year}")
    print(f"Period: {config['start']} to {config['end']}")
    print(f"{'=' * 60}\n")

    # Create temporary config file for this year
    temp_config = f"""
backtest:
  start_date: "{config["start"]}"
  end_date: "{config["end"]}"
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
  timeframe: "15m"
  initial_balance: 10000.0
  initial_currency: "USD"
  output_dir: "backtest_results_{year}"
  save_trade_log: true
  save_equity_curve: true
  generate_report: true

simulation:
  slippage_model: "percentage"
  slippage_percent: 0.01
  maker_fee_percent: 0.02
  taker_fee_percent: 0.06
  latency_ms: 100

data:
  source: "csv"
  data_dir: "data/backtest"
  year: {year}

trading_bot:
  indicators:
    ema_short: 9
    ema_medium: 21
    ema_long: 50
    ema_trend: 200
    rsi_period: 14
    atr_period: 14
  risk:
    max_position_size_percent: 5.0
    max_risk_per_trade_percent: 2.0
    stop_loss_atr_multiplier: 2.0
    take_profit_rr_ratio: 2.0
  hedge:
    enabled: true
    hedge_at_percent: 50.0
    hedge_size_percent: 50.0
"""

    config_file = f"config/backtest_{year}.yaml"
    with open(config_file, "w") as f:
        f.write(temp_config)

    # Run backtest
    try:
        result = subprocess.run(
            ["python3", "backtest_main.py", "--config", config_file],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Parse results
        report_file = f"backtest_results_{year}/backtest_report.json"
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                report = json.load(f)

            # Get trade data
            trades_file = f"backtest_results_{year}/trades.json"
            trades = []
            if os.path.exists(trades_file):
                with open(trades_file, "r") as f:
                    trades = json.load(f)

            total_trades = len(trades)
            wins = len([t for t in trades if t.get("pnl", 0) > 0])
            losses = len([t for t in trades if t.get("pnl", 0) < 0])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(t.get("pnl", 0) for t in trades)
            gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
            gross_loss = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0)
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

            results[year] = {
                "total_return": report.get("total_return", 0),
                "sharpe_ratio": report.get("sharpe_ratio", 0),
                "max_drawdown": report.get("max_drawdown", 0),
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_pnl": total_pnl,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "final_equity": report.get("final_equity", 10000),
            }

            print(f"✅ {year} backtest completed successfully")
            print(
                f"   Trades: {total_trades} | Win Rate: {win_rate:.2f}% | P&L: ${total_pnl:.2f}"
            )
        else:
            print(f"❌ {year} backtest failed - no report generated")
            results[year] = {"error": "No report generated"}

    except subprocess.TimeoutExpired:
        print(f"❌ {year} backtest timed out")
        results[year] = {"error": "Timeout"}
    except Exception as e:
        print(f"❌ {year} backtest error: {e}")
        results[year] = {"error": str(e)}

# Print summary
print(f"\n{'=' * 80}")
print("MULTI-YEAR BACKTEST SUMMARY")
print(f"{'=' * 80}\n")

print(
    f"{'Year':<8} {'Return':<10} {'Sharpe':<8} {'Max DD':<10} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'P.Factor':<10}"
)
print("-" * 80)

for year in ["2023", "2024", "2025"]:
    if year in results and "error" not in results[year]:
        r = results[year]
        print(
            f"{year:<8} {r['total_return']:<10.2f} {r['sharpe_ratio']:<8.2f} {r['max_drawdown']:<10.2f} {r['total_trades']:<8} {r['win_rate']:<8.2f} ${r['total_pnl']:<11.2f} {r['profit_factor']:<10.2f}"
        )
    else:
        print(f"{year:<8} {'ERROR':<10}")

print(f"\n{'=' * 80}")

# Save results to file
with open("backtest_results/multi_year_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to: backtest_results/multi_year_results.json")
