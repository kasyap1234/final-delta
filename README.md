# Delta Exchange India Trading Bot

A sophisticated, production-ready cryptocurrency trading bot designed for Delta Exchange India. This bot implements advanced technical analysis, automated risk management, and an intelligent hedging mechanism to protect positions.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Strategy Summary](#strategy-summary)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Strategy Details](#strategy-details)
- [Database](#database)
- [Troubleshooting](#troubleshooting)
- [Safety & Disclaimer](#safety--disclaimer)

## Overview

This trading bot is designed to trade cryptocurrency perpetual contracts on Delta Exchange India. It uses a multi-timeframe technical analysis approach combined with intelligent risk management and an automatic hedging mechanism to protect capital during adverse market conditions.

### Architecture

The bot follows a modular architecture with clear separation of concerns:

- **Exchange Module**: Handles all exchange interactions via CCXT
- **Data Module**: Manages WebSocket streams and data caching
- **Indicators Module**: Technical analysis and signal generation
- **Risk Module**: Position sizing and portfolio risk management
- **Execution Module**: Order execution and management
- **Hedge Module**: Automatic hedging of losing positions
- **Database Module**: Trade journaling and performance tracking

## Key Features

### Core Capabilities

- **Multi-Symbol Trading**: Monitor and trade multiple cryptocurrency pairs simultaneously
- **Real-time Data**: WebSocket integration for live market data
- **Technical Analysis**: EMA, RSI, ATR, and pivot point analysis
- **Automated Execution**: Market and limit order execution with retry logic
- **Position Management**: Automatic stop-loss and take-profit handling

### Risk Management

- **Position Sizing**: Risk-based position sizing (configurable percentage per trade)
- **Portfolio Limits**: Maximum position count and exposure limits
- **Stop-Loss**: ATR-based dynamic stop-loss calculation
- **Take-Profit**: Risk:Reward ratio-based profit targets

### Hedge Mechanism

The bot includes a sophisticated hedging system:

- **Trigger Condition**: Activates when a position moves 50% toward stop-loss
- **Hedge Selection**: Automatically selects the best hedge pair based on correlation analysis
- **Partial Hedge**: Configurable hedge size (default: 50% of original position)
- **Chunked Execution**: Splits hedge orders to minimize market impact

### Safety Features

- **Circuit Breaker**: Automatically pauses trading after consecutive failures
- **Sandbox Mode**: Test all functionality without risking real capital
- **Graceful Shutdown**: Proper position handling on bot termination
- **Comprehensive Logging**: Detailed logs for debugging and auditing

## Strategy Summary

### Entry Conditions

The bot generates trading signals based on a confluence of technical indicators:

1. **EMA Trend Alignment**: Fast EMA above medium EMA indicates bullish trend
2. **RSI Confirmation**: RSI above threshold confirms momentum
3. **Support/Resistance**: Entry near support (long) or resistance (short)
4. **Volume Confirmation**: Above-average volume validates the signal

### Exit Conditions

- **Take Profit**: Price reaches risk:reward target (default 2:1)
- **Stop Loss**: Price hits ATR-based stop-loss level
- **Signal Reversal**: Opposing signal with sufficient strength

### Hedge Activation

When a position moves 50% toward its stop-loss:

1. Calculate correlations between the losing symbol and other markets
2. Select the most correlated symbol moving in the opposite direction
3. Open a hedge position (50% size of original) in the opposite direction
4. Monitor both positions and close hedge when original recovers or stops out

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB recommended
- **Internet**: Stable connection required for WebSocket data

### Account Requirements

1. **Delta Exchange India Account**: [Sign up here](https://www.delta.exchange)
2. **API Keys**: Generate from your account settings
   - Requires `Trading` permissions
   - IP whitelisting recommended for security
3. **Trading Capital**: Minimum recommended ₹10,000 for meaningful risk management

### Knowledge Prerequisites

- Basic understanding of cryptocurrency trading
- Familiarity with technical indicators (EMA, RSI, ATR)
- Understanding of risk management principles
- Python basics (for customization)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/delta-trading-bot.git
cd delta-trading-bot
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Or install as a package
pip install -e .
```

### Step 4: Configure Environment

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API credentials:
   ```bash
   DELTA_API_KEY=your_actual_api_key
   DELTA_API_SECRET=your_actual_api_secret
   ```

3. Copy and customize the configuration:
   ```bash
   cp config/config.yaml config/config.yaml
   # Edit config.yaml with your preferred settings
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for live trading
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here

# Trading mode
DELTA_SANDBOX=true          # Set to false for live trading
DELTA_TESTNET=true          # Use testnet for testing

# Account settings
DELTA_ACCOUNT_BALANCE=10000.0

# Logging
DELTA_LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Configuration File Structure

The main configuration is in `config/config.yaml`:

```yaml
# Exchange Settings
exchange:
  exchange_id: "delta"
  api_key: null              # Use env vars instead
  api_secret: null           # Use env vars instead
  sandbox: true              # No real trades when true
  testnet: true              # Use testnet

# Trading Settings
trading:
  timeframe: "15m"           # Candlestick timeframe
  symbols:
    - "BTC/USD"
    - "ETH/USD"
    - "SOL/USD"
  max_positions: 5           # Maximum concurrent positions

# Strategy Parameters
strategy:
  ema_fast: 9
  ema_medium: 21
  ema_slow: 50
  ema_trend: 200
  rsi_period: 14
  rsi_long_threshold: 60.0
  rsi_short_threshold: 40.0
  atr_period: 14
  atr_multiplier: 2.0

# Risk Management
risk_management:
  account_balance: 10000.0
  risk_per_trade_percent: 1.0    # Risk 1% per trade
  max_risk_per_trade_percent: 2.0
  take_profit_r_ratio: 2.0       # 2:1 risk:reward

# Hedge Configuration
hedge:
  hedge_trigger_percent: 50.0    # Trigger at 50% toward SL
  hedge_size_ratio: 0.5          # 50% of original position
  hedge_chunks: 3                # Split into 3 orders
  correlation_lookback_days: 30
```

### Trading Parameters Explained

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `timeframe` | Candlestick period for analysis | 15m - 1h |
| `risk_per_trade_percent` | Max risk per trade | 1.0% - 2.0% |
| `take_profit_r_ratio` | Risk:Reward ratio | 2.0 - 3.0 |
| `hedge_trigger_percent` | When to activate hedge | 40.0 - 60.0 |
| `hedge_size_ratio` | Hedge position size | 0.3 - 0.7 |

## Usage

### Running the Bot

#### Basic Usage

```bash
# Run with default configuration
python main.py

# Run with custom config
python main.py --config config/config.yaml

# Run in sandbox mode (recommended for testing)
python main.py --sandbox

# Run with specific symbols
python main.py --symbols BTC/USD,ETH/USD

# Run with debug logging
python main.py --log-level DEBUG
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c` | Path to config file | config/config.yaml |
| `--sandbox, -s` | Run in sandbox mode | False |
| `--symbols` | Comma-separated symbols | From config |
| `--log-level` | Logging level | INFO |
| `--version, -v` | Show version | - |

### Monitoring

The bot provides several ways to monitor its operation:

#### Console Output

Real-time status is displayed in the console:

```
2024-01-27 10:30:15 - INFO - Bot initialized successfully
2024-01-27 10:30:15 - INFO - Starting trading bot...
2024-01-27 10:30:16 - INFO - WebSocket connected
2024-01-27 10:30:20 - INFO - Signal: BUY BTC/USD @ 42500.50 (strength: 75.5)
```

#### Log Files

Logs are written to `logs/trading_bot.log`:

```bash
# View logs in real-time
tail -f logs/trading_bot.log

# Search for specific events
grep "ERROR" logs/trading_bot.log
grep "Position opened" logs/trading_bot.log
```

#### Database Queries

Query the SQLite database for trade history:

```bash
# Open database
sqlite3 data/trading_bot.db

# View recent trades
SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;

# View open positions
SELECT * FROM positions WHERE status = 'open';

# Calculate P&L
SELECT SUM(pnl) as total_pnl FROM trades WHERE status = 'closed';
```

### Stopping the Bot

Press `Ctrl+C` for graceful shutdown:

1. Bot stops accepting new signals
2. Existing positions remain open (optional: close on exit)
3. All pending orders are cancelled
4. Database connections closed
5. Log files finalized

## Strategy Details

### Entry Conditions

A long entry is triggered when ALL of the following conditions are met:

1. **Trend Condition**: Fast EMA (9) > Medium EMA (21) > Slow EMA (50)
2. **Momentum Condition**: RSI (14) > 60 (bullish momentum)
3. **Value Condition**: Price near support level (pivot low)
4. **Volume Condition**: Volume > Average volume (20 periods)

For short entries, conditions are reversed.

### Exit Conditions

#### Take Profit

Calculated as: `Entry Price ± (Stop Loss Distance × Risk:Reward Ratio)`

Example:
- Entry: $42,500
- Stop Loss: $41,500 (distance = $1,000)
- Risk:Reward = 2:1
- Take Profit: $42,500 + ($1,000 × 2) = $44,500

#### Stop Loss

ATR-based stop loss:
- Long: `Entry - (ATR × Multiplier)`
- Short: `Entry + (ATR × Multiplier)`

Default multiplier is 2.0, meaning stop is placed 2 ATRs away from entry.

### Hedge Mechanism

#### When Hedge Activates

When a position's unrealized loss reaches 50% of the stop-loss distance:

```
Example:
- Long position at $42,500
- Stop loss at $41,500 (max loss = $1,000)
- Hedge triggers when unrealized loss = $500 (50% of $1,000)
- Current price = $42,000
```

#### Hedge Selection Process

1. Calculate 30-day correlation between losing symbol and all other symbols
2. Filter for symbols with correlation > 0.7
3. Select the symbol with strongest inverse momentum
4. Open hedge position in opposite direction

#### Hedge Execution

- Size: 50% of original position (configurable)
- Split into 3 chunks to minimize slippage
- Placed as market orders for immediate execution
- Monitored alongside original position

#### Hedge Closure

The hedge is closed when:
- Original position hits take profit (hedge closed immediately)
- Original position hits stop loss (hedge becomes primary position)
- Original position recovers (hedge closed at break-even or small profit)

### Risk Management

#### Position Sizing

Position size is calculated as:

```
Risk Amount = Account Balance × Risk Percentage
Stop Distance = Entry Price - Stop Loss Price
Position Size = Risk Amount / Stop Distance
```

Example:
- Account: ₹100,000
- Risk: 1% = ₹1,000
- Entry: $42,500, Stop: $41,500 (distance = $1,000)
- Position Size = ₹1,000 / $1,000 = 1 unit

#### Portfolio Limits

- Maximum positions: 5 (configurable)
- Maximum risk per trade: 2% of account
- No new positions if total risk > 10% of account

## Database

### Schema Overview

The bot uses SQLite for data persistence with the following tables:

#### trades
Stores all trade executions:
- `trade_id`: Unique identifier
- `symbol`: Trading pair
- `side`: buy/sell/long/short
- `entry_price`, `exit_price`: Execution prices
- `quantity`: Position size
- `pnl`, `pnl_percent`: Profit/loss
- `status`: open/closed/cancelled

#### positions
Tracks open and closed positions:
- `position_id`: Unique identifier
- `symbol`, `side`: Market and direction
- `size`, `entry_price`: Position details
- `stop_loss`, `take_profit`: Exit levels
- `unrealized_pnl`, `realized_pnl`: P&L tracking
- `status`: open/closed/liquidated

#### signals
Records generated trading signals:
- `signal_id`: Unique identifier
- `symbol`, `signal_type`: Market and signal
- `strength`: Signal strength (0-100)
- `indicators`: JSON with indicator values
- `executed`: Whether signal resulted in trade

#### hedges
Hedge position records:
- `hedge_id`: Unique identifier
- `primary_position_id`: Original position
- `hedge_position_id`: Hedge position
- `correlation_at_hedge`: Correlation when opened
- `pnl`: Combined P&L

### Querying Trades

```sql
-- Daily P&L summary
SELECT 
    DATE(entry_time) as date,
    COUNT(*) as trades,
    SUM(pnl) as total_pnl,
    AVG(pnl_percent) as avg_return
FROM trades 
WHERE status = 'closed'
GROUP BY DATE(entry_time)
ORDER BY date DESC;

-- Win rate by symbol
SELECT 
    symbol,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
    SUM(pnl) as total_pnl
FROM trades
WHERE status = 'closed'
GROUP BY symbol;

-- Recent hedge performance
SELECT 
    h.hedge_id,
    h.primary_symbol,
    h.hedge_symbol,
    h.correlation_at_hedge,
    h.pnl as hedge_pnl,
    p.pnl as primary_pnl,
    (h.pnl + COALESCE(p.pnl, 0)) as combined_pnl
FROM hedges h
JOIN positions p ON h.primary_position_id = p.position_id
ORDER BY h.opened_at DESC
LIMIT 10;
```

### Performance Reports

Generate performance reports using the database:

```bash
# Export trades to CSV
sqlite3 -header -csv data/trading_bot.db "SELECT * FROM trades" > trades.csv

# Monthly summary
sqlite3 data/trading_bot.db < scripts/monthly_report.sql
```

## Troubleshooting

### Common Issues

#### Issue: "API Key not found"

**Solution**:
```bash
# Check .env file exists and is properly formatted
cat .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('DELTA_API_KEY'))"

# Ensure python-dotenv is installed
pip install python-dotenv
```

#### Issue: "WebSocket connection failed"

**Solution**:
- Check internet connection
- Verify Delta Exchange is not under maintenance
- Check firewall settings (ports 443, 80)
- Try restarting the bot

#### Issue: "Insufficient balance"

**Solution**:
- Verify account balance in Delta Exchange
- Check if funds are in the correct wallet (trading vs. main)
- Reduce position sizes in config
- Ensure `account_balance` in config matches actual balance

#### Issue: "Rate limit exceeded"

**Solution**:
- Bot has built-in rate limiting, but if exceeded:
- Reduce number of trading pairs
- Increase `check_interval` in config
- Wait a few minutes before restarting

#### Issue: "Position not opening"

**Solution**:
- Check logs for risk management rejections
- Verify `max_positions` limit not reached
- Check if symbol is available on Delta Exchange
- Ensure sufficient margin for position

### Logs Location

```
logs/
├── trading_bot.log          # Main log file
├── trading_bot.error.log    # Error-only log
└── archive/                 # Rotated logs
```

### Debug Mode

Enable debug logging for detailed information:

```bash
python main.py --log-level DEBUG
```

Or set in `.env`:
```bash
DELTA_LOG_LEVEL=DEBUG
```

Debug mode logs:
- All API requests/responses
- WebSocket messages
- Indicator calculations
- Risk check details
- Order execution flow

### Getting Help

1. Check logs: `logs/trading_bot.log`
2. Review configuration: `config/config.yaml`
3. Verify API credentials in Delta Exchange dashboard
4. Test in sandbox mode first
5. Open an issue on GitHub with logs attached

## Safety & Disclaimer

### Trading Risks

**Cryptocurrency trading involves substantial risk of loss.**

- Past performance does not guarantee future results
- The bot can lose money; only trade with capital you can afford to lose
- Technical issues (internet, exchange downtime) can result in losses
- Market conditions can change rapidly and unpredictably

### Test Mode Recommendation

**ALWAYS test thoroughly in sandbox/testnet mode before live trading:**

1. Run with `--sandbox` flag for at least 1 week
2. Verify all functionality works as expected
3. Test the hedge mechanism with simulated losing trades
4. Practice graceful shutdowns and restarts
5. Only then switch to live trading with small amounts

### Security Best Practices

1. **API Keys**: Use IP whitelisting; never share keys
2. **Withdrawal Permissions**: Do NOT enable withdrawal permissions on API keys
3. **Environment Variables**: Never commit `.env` files
4. **Server Security**: Use firewall; keep system updated
5. **Monitoring**: Regularly check logs and account activity

### Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**By using this software, you acknowledge that:**
- You understand the risks of cryptocurrency trading
- You are solely responsible for any trading decisions
- You will not hold the developers liable for any losses
- You will use this software at your own risk

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Trading! 🚀**

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/yourusername/delta-trading-bot).