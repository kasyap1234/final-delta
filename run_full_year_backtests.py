#!/usr/bin/env python3
"""
Full year backtest runner for 2023, 2024, 2025
"""

import subprocess
import json
import os
import sys
from datetime import datetime

# Full year configurations
years = {
    "2023": {
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-12-31T23:59:59Z",
        "desc": "Bear Market Recovery",
    },
    "2024": {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-12-31T23:59:59Z",
        "desc": "Bull Market",
    },
    "2025": {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-12-31T23:59:59Z",
        "desc": "Volatile/Mixed",
    },
}

all_results = {}

for year, config in years.items():
    print(f"\n{'=' * 80}")
    print(f"FULL YEAR BACKTEST: {year} ({config['desc']})")
    print(f"Period: {config['start']} to {config['end']}")
    print(f"{'=' * 80}\n")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"This will take approximately 10-15 minutes...")
    print(f"")

    # Create config file
    config_content = f"""backtest:
  start_date: "{config["start"]}"
  end_date: "{config["end"]}"
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
  timeframe: "15m"
  initial_balance: 10000.0
  initial_currency: "USD"
  output_dir: "backtest_results_{year}_full"
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

    config_file = f"config/backtest_{year}_full.yaml"
    with open(config_file, "w") as f:
        f.write(config_content)

    # Run backtest with extended timeout
    try:
        start_time = datetime.now()

        result = subprocess.run(
            ["python3", "backtest_main.py", "--config", config_file],
            capture_output=True,
            text=True,
            timeout=900,  # 15 minutes max per backtest
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print(f"Completed in {duration:.1f} minutes")

        # Parse results
        report_file = f"backtest_results_{year}_full/backtest_report.json"
        trades_file = f"backtest_results_{year}_full/trades.json"

        if os.path.exists(report_file) and os.path.exists(trades_file):
            with open(report_file, "r") as f:
                report = json.load(f)
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

            all_results[year] = {
                "year": year,
                "description": config["desc"],
                "total_trades": total_trades,
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": profit_factor,
                "return_pct": (total_pnl / 10000) * 100,
                "sharpe_ratio": report.get("sharpe_ratio", 0),
                "max_drawdown": report.get("max_drawdown", 0),
                "final_equity": report.get("final_equity", 10000),
                "duration_minutes": duration,
            }

            print(
                f"✅ {year} FULL YEAR: {total_trades} trades, {win_rate:.1f}% win rate, ${total_pnl:.2f} P&L"
            )
        else:
            print(f"❌ {year}: No results generated")
            all_results[year] = {
                "error": "No report generated",
                "description": config["desc"],
            }

    except subprocess.TimeoutExpired:
        print(f"⏱️ {year}: Timeout after 15 minutes")
        all_results[year] = {"error": "Timeout", "description": config["desc"]}
    except Exception as e:
        print(f"❌ {year}: Error - {e}")
        all_results[year] = {"error": str(e), "description": config["desc"]}

# Print comprehensive summary
print(f"\n{'=' * 100}")
print("FULL YEAR BACKTEST RESULTS SUMMARY (2023-2025)")
print(f"{'=' * 100}\n")

print(
    f"{'Year':<8} {'Market':<20} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'Return':<10} {'Sharpe':<8} {'Max DD':<10} {'P.Factor':<10}"
)
print("-" * 100)

for year in ["2023", "2024", "2025"]:
    if year in all_results and "error" not in all_results[year]:
        r = all_results[year]
        print(
            f"{year:<8} {r['description']:<20} {r['total_trades']:<8} {r['win_rate']:<8.1f} ${r['total_pnl']:<11.2f} {r['return_pct']:<10.2f} {r['sharpe_ratio']:<8.2f} {r['max_drawdown']:<10.2f} {r['profit_factor']:<10.2f}"
        )
    else:
        print(f"{year:<8} {'ERROR':<20}")

print(f"\n{'=' * 100}")

# Calculate totals
valid_results = [r for r in all_results.values() if "error" not in r]
if valid_results:
    total_trades_all = sum(r["total_trades"] for r in valid_results)
    total_pnl_all = sum(r["total_pnl"] for r in valid_results)
    avg_win_rate = sum(r["win_rate"] for r in valid_results) / len(valid_results)
    avg_return = sum(r["return_pct"] for r in valid_results) / len(valid_results)

    print(f"\n📊 COMBINED STATISTICS (3 Years):")
    print(f"   Total Trades: {total_trades_all}")
    print(f"   Total P&L: ${total_pnl_all:.2f}")
    print(f"   Average Win Rate: {avg_win_rate:.1f}%")
    print(f"   Average Annual Return: {avg_return:.2f}%")
    print(f"   Total Return (compounded): {(total_pnl_all / 10000) * 100:.2f}%")

print(f"\n{'=' * 100}")

# Save results
os.makedirs("backtest_results", exist_ok=True)
with open("backtest_results/full_year_2023_2024_2025.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n✅ Results saved to: backtest_results/full_year_2023_2024_2025.json")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
