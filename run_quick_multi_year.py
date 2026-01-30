#!/usr/bin/env python3
"""
Quick multi-year backtest - Q1 of each year (3 months)
"""

import subprocess
import json
import os

# Test Q1 (Jan-Mar) of each year for quicker results
years = {
    "2023": {"start": "2023-01-01T00:00:00Z", "end": "2023-03-31T23:59:59Z"},
    "2024": {"start": "2024-01-01T00:00:00Z", "end": "2024-03-31T23:59:59Z"},
    "2025": {"start": "2025-01-01T00:00:00Z", "end": "2025-03-31T23:59:59Z"},
}

results = {}

for year, dates in years.items():
    print(f"\n{'=' * 60}")
    print(f"BACKTESTING Q1 {year}")
    print(f"{'=' * 60}\n")

    # Create config
    config_content = f"""backtest:
  start_date: "{dates["start"]}"
  end_date: "{dates["end"]}"
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
  timeframe: "15m"
  initial_balance: 10000.0
  initial_currency: "USD"
  output_dir: "backtest_results_q1_{year}"
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

    config_file = f"config/backtest_q1_{year}.yaml"
    with open(config_file, "w") as f:
        f.write(config_content)

    # Run backtest with timeout
    try:
        result = subprocess.run(
            ["python3", "backtest_main.py", "--config", config_file],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes max per backtest
        )

        # Parse results
        report_file = f"backtest_results_q1_{year}/backtest_report.json"
        trades_file = f"backtest_results_q1_{year}/trades.json"

        if os.path.exists(report_file) and os.path.exists(trades_file):
            with open(report_file, "r") as f:
                report = json.load(f)
            with open(trades_file, "r") as f:
                trades = json.load(f)

            total_trades = len(trades)
            wins = len([t for t in trades if t.get("pnl", 0) > 0])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.get("pnl", 0) for t in trades)

            results[year] = {
                "period": f"Q1 {year}",
                "trades": total_trades,
                "win_rate": win_rate,
                "pnl": total_pnl,
                "return_pct": (total_pnl / 10000) * 100,
                "sharpe": report.get("sharpe_ratio", 0),
                "max_dd": report.get("max_drawdown", 0),
            }
            print(
                f"✅ Q1 {year}: {total_trades} trades, {win_rate:.1f}% win rate, ${total_pnl:.2f} P&L"
            )
        else:
            print(f"❌ Q1 {year}: No results generated")

    except subprocess.TimeoutExpired:
        print(f"⏱️ Q1 {year}: Timeout (taking too long)")
    except Exception as e:
        print(f"❌ Q1 {year}: Error - {e}")

# Print summary
print(f"\n{'=' * 80}")
print("MULTI-YEAR BACKTEST SUMMARY (Q1 Results)")
print(f"{'=' * 80}\n")
print(
    f"{'Year':<8} {'Period':<12} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'Return':<10} {'Sharpe':<8}"
)
print("-" * 80)

for year in ["2023", "2024", "2025"]:
    if year in results:
        r = results[year]
        print(
            f"{year:<8} {r['period']:<12} {r['trades']:<8} {r['win_rate']:<8.1f} ${r['pnl']:<11.2f} {r['return_pct']:<10.2f} {r['sharpe']:<8.2f}"
        )
    else:
        print(
            f"{year:<8} {'N/A':<12} {'N/A':<8} {'N/A':<8} {'N/A':<12} {'N/A':<10} {'N/A':<8}"
        )

print(f"\n{'=' * 80}")

# Save results
with open("backtest_results/multi_year_q1_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: backtest_results/multi_year_q1_results.json")
print(
    "\nNote: These are Q1 (3-month) results. Full year backtests take 10-15 minutes each."
)
