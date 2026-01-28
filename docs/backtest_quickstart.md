# Backtest Quick Start Guide

This guide will help you quickly set up and run backtests for the Delta Trading Bot using historical data from 2025.

## Prerequisites

Before running backtests, ensure you have:

1. **Python 3.8+** installed
2. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Credentials** (optional for data fetching):
   - If you want to fetch real historical data from Delta Exchange, set up your API credentials in `.env`:
     ```
     DELTA_API_KEY=your_api_key
     DELTA_API_SECRET=your_api_secret
     ```

## Directory Structure

The backtest system uses the following directory structure:

```
data/
└── backtest/
    ├── BTC_USD_15m.csv      # Historical OHLCV data files
    ├── ETH_USD_15m.csv
    ├── SOL_USD_15m.csv
    ├── reports/             # Backtest reports (HTML, JSON, text)
    ├── trades/              # Trade logs (CSV, JSON)
    └── equity/              # Equity curves (CSV, JSON)
```

## Quick Start

### Option 1: Run Everything Automatically

The easiest way to run a backtest is to use the quick start script:

```bash
./scripts/run_backtest.sh
```

This script will:
1. Check if historical data exists
2. Fetch data if needed (from Delta Exchange API)
3. Run the backtest
4. Generate reports

### Option 2: Manual Steps

If you prefer more control, you can run each step manually:

#### Step 1: Fetch Historical Data

Fetch OHLCV data for 2025:

```bash
python3 scripts/fetch_historical_data.py
```

Customize the fetch with options:

```bash
# Fetch for specific date range
python3 scripts/fetch_historical_data.py \
    --start-date 2025-01-01 \
    --end-date 2025-12-31

# Fetch for specific symbols
python3 scripts/fetch_historical_data.py \
    --symbols BTC/USD,ETH/USD

# Fetch for different timeframe
python3 scripts/fetch_historical_data.py \
    --timeframe 1h

# Force re-fetch even if data exists
python3 scripts/fetch_historical_data.py --force
```

#### Step 2: Run Backtest

Run the backtest using the fetched data:

```bash
python3 backtest_main.py
```

Or with a custom config:

```bash
python3 backtest_main.py --config config/backtest.yaml
```

Enable verbose logging:

```bash
python3 backtest_main.py --verbose
```

## Configuration

### Backtest Configuration

Edit [`config/backtest.yaml`](../config/backtest.yaml) to customize your backtest:

```yaml
backtest:
  # Date range for backtest
  start_date: "2025-01-01T00:00:00Z"
  end_date: "2025-12-31T23:59:59Z"
  
  # Trading symbols
  symbols:
    - "BTC/USD"
    - "ETH/USD"
    - "SOL/USD"
  
  # Timeframe for candles
  timeframe: "15m"
  
  # Initial account state
  initial_balance: 10000.0
  initial_currency: "USD"
  
  # Output settings
  output_dir: "backtest_results"
  save_trade_log: true
  save_equity_curve: true
  generate_report: true

simulation:
  # Slippage model: percentage, fixed, or none
  slippage_model: "percentage"
  slippage_percent: 0.01
  
  # Trading fees
  maker_fee_percent: 0.02
  taker_fee_percent: 0.06
  
  # Simulated order latency in milliseconds
  latency_ms: 100
```

### Trading Bot Configuration

The trading strategy and risk management settings are in [`config/config.yaml`](../config/config.yaml):

```yaml
strategy:
  ema_fast: 9
  ema_medium: 21
  ema_slow: 50
  ema_trend: 200
  rsi_period: 14
  atr_period: 14

risk_management:
  account_balance: 10000.0
  risk_per_trade_percent: 1.0
  max_risk_per_trade_percent: 2.0
  take_profit_r_ratio: 2.0
```

## Understanding the Output

### Backtest Results

After running a backtest, you'll see a summary:

```
============================================================
BACKTEST RESULTS
============================================================
Total Return: 15.23%
Sharpe Ratio: 1.45
Max Drawdown: -8.34%
Win Rate: 62.50%
Total Trades: 48
Total Fees: $123.45
Final Equity: $11523.00
Initial Balance: $10000.00
============================================================
```

### Generated Reports

The backtest generates several output files in the output directory:

1. **Trade Logs** (`trades/`):
   - `trades.csv` - Detailed trade history
   - `trades.json` - Trade data in JSON format
   - `trades_summary.txt` - Trade statistics summary

2. **Equity Curves** (`equity/`):
   - `equity_curve.csv` - Equity over time
   - `equity_curve.json` - Equity data in JSON format
   - `drawdown_curve.csv` - Drawdown over time
   - `returns_curve.csv` - Returns over time
   - `equity_summary.txt` - Equity statistics

3. **Reports** (`reports/`):
   - `backtest_report.html` - Interactive HTML report
   - `backtest_report.json` - Report data in JSON format
   - `backtest_report.txt` - Text-based report

## Data Format

### CSV Format for Historical Data

Historical data files should be in CSV format with the following columns:

```csv
timestamp,open,high,low,close,volume
2025-01-01T00:00:00,42500.0,42800.0,42400.0,42750.0,123.45
2025-01-01T00:15:00,42750.0,42900.0,42650.0,42800.0,98.76
...
```

- **timestamp**: ISO 8601 format (e.g., `2025-01-01T00:00:00`)
- **open**: Opening price
- **high**: Highest price
- **low**: Lowest price
- **close**: Closing price
- **volume**: Trading volume

## Troubleshooting

### Missing Dependencies

If you get import errors, install dependencies:

```bash
pip install -r requirements.txt
```

### API Connection Issues

If data fetching fails:

1. Check your API credentials in `.env`
2. Ensure you have internet connectivity
3. Verify Delta Exchange API is accessible
4. Check rate limits (you may need to add delays)

### Data Validation Warnings

If you see warnings about data gaps:

- This is normal for historical data
- Gaps may occur during exchange maintenance
- The backtest will still run with available data

### No Data Found

If the backtest reports no data:

1. Verify data files exist in `data/backtest/`
2. Check the file names match the expected format: `{SYMBOL}_{TIMEFRAME}.csv`
3. Ensure the date range in config matches your data

## Advanced Usage

### Custom Date Ranges

Test specific periods:

```bash
# Test Q1 2025
python3 scripts/fetch_historical_data.py \
    --start-date 2025-01-01 \
    --end-date 2025-03-31

# Test a specific month
python3 scripts/fetch_historical_data.py \
    --start-date 2025-06-01 \
    --end-date 2025-06-30
```

### Multiple Timeframes

Compare performance across timeframes:

```bash
# 1-hour timeframe
python3 scripts/fetch_historical_data.py --timeframe 1h
python3 backtest_main.py

# 4-hour timeframe
python3 scripts/fetch_historical_data.py --timeframe 4h --force
python3 backtest_main.py
```

### Custom Symbols

Test different trading pairs:

```bash
python3 scripts/fetch_historical_data.py \
    --symbols BTC/USD,ETH/USD,LTC/USD
```

## Next Steps

1. **Review Results**: Analyze the generated reports to understand performance
2. **Optimize Parameters**: Adjust strategy parameters in the config
3. **Compare Periods**: Run backtests on different time periods
4. **Validate Strategy**: Ensure results are consistent across periods

## Additional Resources

- [Backtest Design Document](backtest_design.md)
- [Backtest Implementation Plan](backtest_implementation_plan.md)
- [Architecture Documentation](architecture.md)

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review the error messages carefully
3. Ensure all dependencies are installed
4. Verify configuration files are valid
