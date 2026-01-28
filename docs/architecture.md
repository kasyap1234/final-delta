# Delta Exchange India Trading Bot - Architecture Document

## 1. System Overview

This document outlines the architecture for a Python-based cryptocurrency trading bot designed for Delta Exchange India. The bot implements a trend-following strategy with an advanced hedge mechanism for risk management.

### Key Features
- **Strategy**: Trend-following with EMA crossovers, RSI confirmation, and ATR-based risk management
- **Timeframe**: 15-minute candlesticks
- **Exchange**: Delta Exchange India via CCXT Pro
- **Data Feed**: WebSocket for real-time market data
- **Order Type**: Post-only limit orders to minimize fees
- **Risk Management**: Dynamic position sizing, ATR-based stop losses, and correlation-based hedging

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING BOT SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Config        │    │   Database      │    │   Logging       │         │
│  │   Manager       │    │   Manager       │    │   Service       │         │
│  │   (YAML/JSON)   │    │   (SQLite)      │    │   (Structured)  │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                  │
│           └──────────────────────┼──────────────────────┘                  │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Core Trading Engine                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │   State     │  │   Signal    │  │   Risk      │  │  Order    │  │   │
│  │  │   Manager   │  │   Generator │  │   Manager   │  │  Manager  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│           ┌──────────────────────┼──────────────────────┐                  │
│           │                      │                      │                  │
│           ▼                      ▼                      ▼                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Market Data   │    │   Hedge         │    │   Position      │         │
│  │   Handler       │    │   Engine        │    │   Tracker       │         │
│  │   (WebSocket)   │    │   (Correlation) │    │   (P&L Monitor) │         │
│  └────────┬────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                │
│           ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CCXT Pro Exchange Interface                      │   │
│  │                         (Delta Exchange India)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Delta Exchange India API                         │   │
│  │              (REST + WebSocket for Orders & Market Data)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Breakdown and Responsibilities

### 3.1 Core Modules

#### `config_manager.py`
**Responsibilities:**
- Load and validate configuration from YAML/JSON files
- Provide centralized access to all configurable parameters
- Support environment-specific configurations (dev/prod)
- Validate configuration integrity on startup

**Key Methods:**
- `load_config(path: str) -> ConfigDict`
- `get_trading_params() -> TradingConfig`
- `get_risk_params() -> RiskConfig`
- `get_hedge_params() -> HedgeConfig`
- `validate_config() -> bool`

---

#### `exchange_client.py`
**Responsibilities:**
- Initialize and manage CCXT Pro exchange connection
- Handle authentication and API key management
- Implement rate limiting and retry logic
- Provide unified interface for all exchange operations

**Key Methods:**
- `initialize() -> None`
- `watch_ohlcv(symbol: str, timeframe: str) -> AsyncGenerator`
- `watch_ticker(symbol: str) -> AsyncGenerator`
- `create_limit_order(symbol: str, side: str, amount: float, price: float, params: dict) -> Order`
- `cancel_order(order_id: str, symbol: str) -> Order`
- `fetch_balance() -> Balance`
- `fetch_positions(symbol: str = None) -> List[Position]`

---

#### `market_data_handler.py`
**Responsibilities:**
- Subscribe to real-time market data via WebSocket
- Maintain OHLCV data buffers for multiple symbols
- Calculate technical indicators (EMA, RSI, ATR, Pivot Points)
- Provide clean data feeds to signal generator

**Key Methods:**
- `subscribe(symbols: List[str], timeframe: str) -> None`
- `get_ohlcv(symbol: str, limit: int = 200) -> DataFrame`
- `get_latest_price(symbol: str) -> float`
- `calculate_indicators(symbol: str) -> IndicatorSet`
- `update_correlation_matrix(symbols: List[str], lookback: int) -> CorrelationMatrix`

---

#### `signal_generator.py`
**Responsibilities:**
- Generate trading signals based on technical indicators
- Implement long and short entry/exit logic
- Filter signals based on trend conditions and RSI thresholds
- Check for resistance/support levels before entry

**Key Methods:**
- `generate_signal(symbol: str, indicators: IndicatorSet) -> Signal`
- `check_long_entry(indicators: IndicatorSet) -> bool`
- `check_short_entry(indicators: IndicatorSet) -> bool`
- `check_exit_conditions(position: Position, indicators: IndicatorSet) -> bool`
- `is_near_resistance(price: float, pivot: PivotPoints) -> bool`

---

#### `risk_manager.py`
**Responsibilities:**
- Calculate position sizes based on account balance and risk percentage
- Compute stop-loss and take-profit levels using ATR
- Validate orders against risk limits
- Track open positions and exposure

**Key Methods:**
- `calculate_position_size(balance: float, risk_pct: float, entry: float, stop_loss: float) -> float`
- `calculate_stop_loss(entry: float, atr: float, side: str, multiplier: float = 2.0) -> float`
- `calculate_take_profit(entry: float, stop_loss: float, risk_reward: float = 2.0) -> float`
- `validate_position_risk(position: Position) -> bool`
- `get_account_exposure() -> ExposureReport`

---

#### `order_manager.py`
**Responsibilities:**
- Execute post-only limit orders
- Handle order lifecycle (create, modify, cancel)
- Implement order retry logic with price adjustments
- Track order status and fills

**Key Methods:**
- `place_entry_order(signal: Signal) -> Order`
- `place_stop_loss_order(position: Position) -> Order`
- `place_take_profit_order(position: Position) -> Order`
- `modify_order(order_id: str, new_price: float) -> Order`
- `cancel_pending_orders(symbol: str) -> None`
- `get_order_status(order_id: str) -> OrderStatus`

---

#### `hedge_engine.py`
**Responsibilities:**
- Monitor unrealized P&L for hedge trigger conditions
- Calculate correlations between trading pairs
- Execute hedge positions in chunks
- Manage hedge lifecycle and profit-taking

**Key Methods:**
- `monitor_positions(positions: List[Position]) -> List[HedgeTrigger]`
- `select_hedge_asset(original_symbol: str, correlations: CorrelationMatrix) -> str`
- `execute_hedge_leg(trigger: HedgeTrigger, chunk: int) -> Order`
- `check_hedge_take_profit(hedge: Position) -> bool`
- `close_hedge_position(hedge_id: str) -> Order`
- `calculate_total_pnl(positions: List[Position], hedges: List[Position]) -> float`

---

#### `position_tracker.py`
**Responsibilities:**
- Track all open positions and their metadata
- Calculate unrealized and realized P&L
- Monitor position duration and performance
- Provide position status updates

**Key Methods:**
- `add_position(position: Position) -> None`
- `update_position(position_id: str, updates: dict) -> None`
- `close_position(position_id: str, exit_price: float) -> TradeRecord`
- `get_open_positions(symbol: str = None) -> List[Position]`
- `get_position_pnl(position_id: str) -> PnLReport`

---

#### `state_manager.py`
**Responsibilities:**
- Maintain application state across modules
- Handle state persistence and recovery
- Manage bot lifecycle states (STARTING, RUNNING, STOPPING, ERROR)
- Coordinate graceful shutdown

**Key Methods:**
- `set_state(state: BotState) -> None`
- `get_state() -> BotState`
- `persist_state() -> None`
- `recover_state() -> bool`
- `request_shutdown(reason: str) -> None`

---

#### `database_manager.py`
**Responsibilities:**
- Manage SQLite database connections
- Store trade journal entries
- Record all order executions and modifications
- Provide query interface for trade analysis

**Key Methods:**
- `initialize_database() -> None`
- `log_trade(trade: TradeRecord) -> None`
- `log_signal(signal: Signal) -> None`
- `query_trades(filters: TradeFilters) -> List[TradeRecord]`
- `get_performance_metrics(start_date: datetime, end_date: datetime) -> Metrics`

---

#### `logging_service.py`
**Responsibilities:**
- Configure structured logging with multiple handlers
- Support different log levels (DEBUG, INFO, WARNING, ERROR)
- Write logs to file with rotation
- Send critical alerts via appropriate channels

**Key Methods:**
- `configure_logging(config: LogConfig) -> None`
- `log_trade_event(event: TradeEvent) -> None`
- `log_error(error: Exception, context: dict) -> None`
- `log_performance_metrics(metrics: Metrics) -> None`

---

### 3.2 Data Models

```python
# Core data structures (simplified representations)

class Signal:
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    confidence: float

class Position:
    id: str
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    status: str  # 'open', 'closed', 'hedged'
    unrealized_pnl: float
    hedges: List[str]  # IDs of hedge positions

class IndicatorSet:
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float
    rsi: float
    atr: float
    pivot_points: PivotPoints
    timestamp: datetime

class PivotPoints:
    pivot: float
    resistance_1: float
    resistance_2: float
    support_1: float
    support_2: float

class HedgeTrigger:
    original_position_id: str
    trigger_price: float
    hedge_symbol: str
    hedge_side: str
    hedge_size: float
    chunks_remaining: int
```

---

## 4. Data Flow Description

### 4.1 Main Trading Loop

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Start     │────▶│  Load Config &   │────▶│  Initialize     │
│             │     │  Validate        │     │  Exchange       │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Sleep     │◀────│  Wait for Next   │◀────│  Subscribe to   │
│   (Interval)│     │  Candle          │     │  Market Data    │
└──────┬──────┘     └──────────────────┘     └─────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Calculate  │────▶│  Generate        │────▶│  Check Risk     │
│  Indicators │     │  Signals         │     │  Constraints    │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                              │                       │                       │
                              ▼                       ▼                       ▼
                       ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
                       │  No Signal  │        │  Entry      │        │  Exit       │
                       │  (Continue) │        │  Signal     │        │  Signal     │
                       └─────────────┘        └──────┬──────┘        └──────┬──────┘
                                                     │                      │
                                                     ▼                      ▼
                                              ┌─────────────┐        ┌─────────────┐
                                              │  Place      │        │  Close      │
                                              │  Orders     │        │  Position   │
                                              │  (SL/TP)    │        │             │
                                              └──────┬──────┘        └─────────────┘
                                                     │
                                                     ▼
                                              ┌─────────────┐
                                              │  Log Trade  │
                                              │  to DB      │
                                              └─────────────┘
```

### 4.2 Hedge Mechanism Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HEDGE MONITORING LOOP                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  Check P&L  │────▶│  Loss > 50% SL?  │────▶│  Calculate Correlation  │
│  of Open    │     │                  │     │  Matrix                 │
│  Positions  │     │  (Trigger Check) │     │                         │
└─────────────┘     └──────────────────┘     └─────────────┬───────────┘
                                                           │
                              ┌────────────────────────────┘
                              ▼
                       ┌─────────────┐
                       │  Select     │
                       │  Hedge      │
                       │  Asset      │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐     ┌──────────────────┐
                       │  Split into │────▶│  Execute Chunk 1 │
                       │  3 Chunks   │     │  (1/3 Size)      │
                       └─────────────┘     └────────┬─────────┘
                                                    │
                                                    ▼
                       ┌─────────────┐     ┌──────────────────┐
                       │  Monitor    │◀────│  Wait for Fill   │
                       │  Hedge P&L  │     │                  │
                       └──────┬──────┘     └──────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
       │  Loss       │ │  Profit >   │ │  Loss       │
       │  Widens     │ │  Threshold? │ │  Unchanged  │
       │  (+50% SL)  │ │             │ │             │
       └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
              │               │               │
              ▼               ▼               ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
       │  Execute    │ │  Close      │ │  Continue   │
       │  Additional │ │  Hedge      │ │  Monitoring │
       │  Hedge      │ │  (Take      │ │             │
       │  Leg        │ │  Profit)    │ │             │
       └─────────────┘ └─────────────┘ └─────────────┘
```

### 4.3 Order Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORDER EXECUTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

Signal Generated
       │
       ▼
┌─────────────┐
│  Calculate  │
│  Position   │
│  Size       │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│  Check      │────▶│  Insufficient    │
│  Balance    │     │  Funds?          │
└─────────────┘     │  (Log & Skip)    │
                    └──────────────────┘
       │
       ▼
┌─────────────┐
│  Calculate  │
│  Post-Only  │
│  Price      │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Submit     │────▶│  Rate Limit?     │────▶│  Retry with      │
│  Limit      │     │                  │     │  Backoff         │
│  Order      │     │  (Wait & Retry)  │     │                  │
└─────────────┘     └──────────────────┘     └──────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│  Order      │────▶│  Rejected?       │
│  Accepted?  │     │  (Adjust Price   │
└─────────────┘     │  & Retry)        │
                    └──────────────────┘
       │
       ▼
┌─────────────┐
│  Place      │
│  Stop Loss  │
│  Order      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Place      │
│  Take       │
│  Profit     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Log to     │
│  Database   │
└─────────────┘
```

---

## 5. Database Schema (SQLite)

### 5.1 Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     trades      │       │    signals      │       │   positions     │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │       │ id (PK)         │
│ signal_id (FK)  │◀──────│ timestamp       │       │ trade_id (FK)   │◀──┐
│ position_id(FK) │◀──────│ symbol          │       │ symbol          │   │
│ symbol          │       │ side            │       │ side            │   │
│ side            │       │ entry_price     │       │ entry_price     │   │
│ order_type      │       │ stop_loss       │       │ size            │   │
│ entry_price     │       │ take_profit     │       │ stop_loss       │   │
│ size            │       │ position_size   │       │ take_profit     │   │
│ stop_loss       │       │ confidence      │       │ status          │   │
│ take_profit     │       │ executed        │       │ open_time       │   │
│ exit_price      │       │                 │       │ close_time      │   │
│ pnl             │       └─────────────────┘       │ unrealized_pnl  │   │
│ fees            │                                 │ realized_pnl    │   │
│ timestamp       │                                 └─────────────────┘   │
└─────────────────┘                                                       │
                                                                          │
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐   │
│     orders      │       │     hedges      │       │  correlations   │   │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤   │
│ id (PK)         │       │ id (PK)         │       │ id (PK)         │   │
│ trade_id (FK)   │       │ position_id(FK) │───────┤ timestamp       │   │
│ position_id(FK) │◀──────│ hedge_symbol    │       │ symbol_1        │   │
│ order_type      │       │ hedge_side      │       │ symbol_2        │   │
│ side            │       │ hedge_size      │       │ correlation     │   │
│ price           │       │ entry_price     │       │ lookback_days   │   │
│ size            │       │ status          │       └─────────────────┘   │
│ status          │       │ chunk_number    │                             │
│ exchange_id     │       │ total_chunks    │                             │
│ timestamp       │       │ pnl             │                             │
└─────────────────┘       └─────────────────┘                             │
                                                                          │
┌─────────────────┐       ┌─────────────────┐                             │
│  market_data    │       │  performance    │                             │
├─────────────────┤       ├─────────────────┤                             │
│ id (PK)         │       │ id (PK)         │                             │
│ symbol          │       │ date            │                             │
│ timestamp       │       │ total_trades    │                             │
│ open            │       │ winning_trades  │                             │
│ high            │       │ losing_trades   │                             │
│ low             │       │ total_pnl       │                             │
│ close           │       │ avg_win         │                             │
│ volume          │       │ avg_loss        │                             │
│ ema_9           │       │ win_rate        │                             │
│ ema_21          │       │ profit_factor   │                             │
│ ema_50          │       │ sharpe_ratio    │                             │
│ ema_200         │       │ max_drawdown    │                             │
│ rsi             │       └─────────────────┘                             │
│ atr             │                                                     │
└─────────────────┘                                                     │
                                                                        │
                              ┌─────────────────┐                       │
                              │ balance_history │                       │
                              ├─────────────────┤                       │
                              │ id (PK)         │                       │
                              │ timestamp       │                       │
                              │ total_balance   │                       │
                              │ available       │                       │
                              │ in_positions    │                       │
                              │ in_hedges       │                       │
                              └─────────────────┘                       │
                                                                        │
                              ┌─────────────────┐◀──────────────────────┘
                              │  trade_log      │
                              ├─────────────────┤
                              │ id (PK)         │
                              │ trade_id (FK)   │
                              │ event_type      │
                              │ message         │
                              │ timestamp       │
                              └─────────────────┘
```

### 5.2 Table Definitions

```sql
-- Core trades table
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    position_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    order_type TEXT NOT NULL DEFAULT 'limit',
    entry_price REAL NOT NULL,
    exit_price REAL,
    size REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    pnl REAL,
    fees REAL DEFAULT 0,
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed', 'cancelled')),
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP,
    FOREIGN KEY (signal_id) REFERENCES signals(id),
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

-- Signals generated by strategy
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    position_size REAL NOT NULL,
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    executed BOOLEAN DEFAULT FALSE,
    execution_time TIMESTAMP
);

-- Active and closed positions
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    entry_price REAL NOT NULL,
    size REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed', 'hedged')),
    open_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    close_time TIMESTAMP,
    unrealized_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);

-- Order tracking
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER,
    position_id INTEGER,
    order_type TEXT NOT NULL CHECK(order_type IN ('entry', 'stop_loss', 'take_profit', 'hedge')),
    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    price REAL NOT NULL,
    size REAL NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'filled', 'partial', 'cancelled', 'rejected')),
    exchange_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_time TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES trades(id),
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

-- Hedge positions
CREATE TABLE hedges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    hedge_symbol TEXT NOT NULL,
    hedge_side TEXT NOT NULL CHECK(hedge_side IN ('long', 'short')),
    hedge_size REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed')),
    chunk_number INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL DEFAULT 3,
    pnl REAL DEFAULT 0,
    open_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    close_time TIMESTAMP,
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

-- Market data cache (optional, for backtesting/analysis)
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    ema_9 REAL,
    ema_21 REAL,
    ema_50 REAL,
    ema_200 REAL,
    rsi REAL,
    atr REAL,
    UNIQUE(symbol, timestamp)
);

-- Correlation matrix tracking
CREATE TABLE correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol_1 TEXT NOT NULL,
    symbol_2 TEXT NOT NULL,
    correlation REAL NOT NULL CHECK(correlation >= -1 AND correlation <= 1),
    lookback_days INTEGER NOT NULL DEFAULT 30,
    UNIQUE(timestamp, symbol_1, symbol_2)
);

-- Performance metrics
CREATE TABLE performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    avg_win REAL DEFAULT 0,
    avg_loss REAL DEFAULT 0,
    win_rate REAL DEFAULT 0,
    profit_factor REAL DEFAULT 0,
    sharpe_ratio REAL,
    max_drawdown REAL DEFAULT 0
);

-- Balance history
CREATE TABLE balance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_balance REAL NOT NULL,
    available_balance REAL NOT NULL,
    in_positions REAL DEFAULT 0,
    in_hedges REAL DEFAULT 0
);

-- Trade log for debugging
CREATE TABLE trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER,
    event_type TEXT NOT NULL,
    message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);

-- Indexes for performance
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_time ON trades(entry_time);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timestamp);
CREATE INDEX idx_correlations_time ON correlations(timestamp);
```

---

## 6. Configuration File Structure

### 6.1 Main Configuration File (`config.yaml`)

```yaml
# Delta Exchange India Trading Bot Configuration

# Application Settings
app:
  name: "Delta Trading Bot"
  version: "1.0.0"
  environment: "production"  # development, production
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/trading_bot.log"
  state_file: "data/bot_state.json"

# Exchange Configuration
exchange:
  name: "delta"
  api_key: "${DELTA_API_KEY}"
  api_secret: "${DELTA_API_SECRET}"
  sandbox: false
  enable_rate_limit: true
  timeout: 30000  # milliseconds
  
  # WebSocket Settings
  websocket:
    reconnect_interval: 5000  # milliseconds
    heartbeat_interval: 30000  # milliseconds
    max_reconnect_attempts: 10

# Trading Parameters
trading:
  # Timeframe for analysis
  timeframe: "15m"
  
  # Trading pairs (prioritized for correlation)
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
  
  # Order settings
  order_type: "limit"
  post_only: true
  price_offset_ticks: 1  # Number of ticks to offset for post-only
  
  # Position management
  max_open_positions: 5
  max_positions_per_symbol: 1

# Strategy Configuration
strategy:
  # EMA Periods
  ema:
    fast: 9
    medium: 21
    slow: 50
    trend: 200
  
  # RSI Settings
  rsi:
    period: 14
    overbought: 70
    oversold: 30
    long_threshold: 60
    short_threshold: 40
  
  # ATR Settings
  atr:
    period: 14
    stop_loss_multiplier: 2.0
  
  # Pivot Points
  pivot_points:
    period: 14  # Lookback period for pivot calculation
  
  # Entry Conditions
  entry:
    require_trend_alignment: true
    check_resistance: true
    resistance_buffer_pct: 0.5  # Don't enter if within 0.5% of resistance

# Risk Management
risk:
  # Position sizing
  risk_per_trade_pct: 1.5  # 1-2% of account balance
  max_risk_per_trade_pct: 2.0
  min_risk_per_trade_pct: 1.0
  
  # Risk:Reward Ratio
  risk_reward_ratio: 2.0  # 2:1
  
  # Account limits
  max_daily_loss_pct: 5.0
  max_drawdown_pct: 10.0
  
  # Leverage (if applicable)
  max_leverage: 1

# Hedge Configuration
hedge:
  # Trigger Settings
  trigger_threshold_pct: 50.0  # % of SL distance
  
  # Hedge Sizing
  size_ratio: 0.5  # 50% of original position
  
  # Execution
  num_chunks: 3
  chunk_delay_seconds: 60
  
  # Asset Selection
  correlation_lookback_days: 30
  priority_symbols:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
  
  # Profit Taking
  profit_threshold_pct: 1.0  # Close hedge at 1% profit
  
  # Re-hedging
  rehedge_threshold_pct: 50.0  # Additional 50% of SL for new hedge
  max_hedges_per_position: 3

# Database Configuration
database:
  path: "data/trading.db"
  backup_interval_hours: 24
  retention_days: 365

# Notification Settings (Optional)
notifications:
  enabled: false
  
  # Email notifications
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "${EMAIL_USERNAME}"
    password: "${EMAIL_PASSWORD}"
    to_address: "admin@example.com"
    
  # Webhook notifications
  webhook:
    url: "${WEBHOOK_URL}"
    events:
      - "trade_entry"
      - "trade_exit"
      - "hedge_triggered"
      - "error"

# Performance Monitoring
monitoring:
  metrics_enabled: true
  metrics_interval_seconds: 60
  health_check_interval_seconds: 30
```

### 6.2 Environment Variables (`.env`)

```bash
# Delta Exchange India API Credentials
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here

# Optional: Email Notifications
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Optional: Webhook Notifications
WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

---

## 7. Error Handling Strategy

### 7.1 Error Categories

| Category | Examples | Handling Strategy |
|----------|----------|-------------------|
| **Network** | Connection timeout, DNS failure | Exponential backoff retry, circuit breaker |
| **Exchange** | Rate limit, invalid order, insufficient funds | Specific error handling, user notification |
| **Data** | Missing OHLCV, malformed response | Log and skip, use cached data |
| **Logic** | Division by zero, invalid state | Defensive programming, graceful degradation |
| **System** | Disk full, memory error | Critical alert, graceful shutdown |

### 7.2 Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ERROR HANDLING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

Error Occurred
       │
       ▼
┌─────────────┐
│  Classify   │
│  Error      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ERROR CATEGORIES                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Network   │  │  Exchange   │  │    Data     │  │   System    │    │
│  │    Error    │  │    Error    │  │    Error    │  │    Error    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │           │
│         ▼                ▼                ▼                ▼           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Retry with │  │  Check Error│  │  Log & Use  │  │  Immediate  │    │
│  │  Backoff    │  │  Code       │  │  Cached     │  │  Shutdown   │    │
│  │  (Max 5x)   │  │             │  │  Data       │  │             │    │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘  └─────────────┘    │
│         │                │                                             │
│         │                ▼                                             │
│         │         ┌─────────────┐                                      │
│         │         │ Rate Limit? │                                      │
│         │         └──────┬──────┘                                      │
│         │                │                                             │
│         │       ┌────────┴────────┐                                    │
│         │       │                 │                                    │
│         │       ▼                 ▼                                    │
│         │  ┌─────────┐      ┌─────────┐                                │
│         │  │  Wait   │      │ Invalid │                                │
│         │  │  Retry  │      │  Order? │                                │
│         │  │  After  │      └────┬────┘                                │
│         │  │  Reset  │           │                                     │
│         │  └─────────┘           ▼                                     │
│         │                   ┌─────────┐                                │
│         │                   │  Log &  │                                │
│         │                   │  Skip   │                                │
│         │                   └─────────┘                                │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     CIRCUIT BREAKER PATTERN                     │   │
│  │                                                                 │   │
│  │  Failure Count > Threshold ──▶ Open Circuit ──▶ Wait Cooldown   │   │
│  │         │                                                    │   │
│  │         └───────────────────▶ Half-Open ──▶ Success?          │   │
│  │                                      │                        │   │
│  │                                      ▼                        │   │
│  │                              Close Circuit                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Exception Hierarchy

```python
# Custom exception classes for the trading bot

class TradingBotError(Exception):
    """Base exception for all trading bot errors"""
    pass

# Exchange-related errors
class ExchangeError(TradingBotError):
    """Base class for exchange-related errors"""
    pass

class RateLimitError(ExchangeError):
    """Raised when rate limit is exceeded"""
    pass

class InsufficientFundsError(ExchangeError):
    """Raised when account has insufficient funds"""
    pass

class OrderRejectedError(ExchangeError):
    """Raised when order is rejected by exchange"""
    pass

class ConnectionError(ExchangeError):
    """Raised when connection to exchange fails"""
    pass

# Strategy-related errors
class StrategyError(TradingBotError):
    """Base class for strategy errors"""
    pass

class SignalGenerationError(StrategyError):
    """Raised when signal generation fails"""
    pass

class InvalidIndicatorError(StrategyError):
    """Raised when indicator calculation fails"""
    pass

# Risk management errors
class RiskError(TradingBotError):
    """Base class for risk management errors"""
    pass

class PositionSizeError(RiskError):
    """Raised when position size calculation fails"""
    pass

class RiskLimitExceededError(RiskError):
    """Raised when risk limits are exceeded"""
    pass

# Data-related errors
class DataError(TradingBotError):
    """Base class for data errors"""
    pass

class DataFeedError(DataError):
    """Raised when data feed fails"""
    pass

class CorrelationError(DataError):
    """Raised when correlation calculation fails"""
    pass

# Hedge-related errors
class HedgeError(TradingBotError):
    """Base class for hedge errors"""
    pass

class HedgeExecutionError(HedgeError):
    """Raised when hedge execution fails"""
    pass
```

### 7.4 Retry Strategy

| Error Type | Retry Count | Backoff Strategy | Circuit Breaker |
|------------|-------------|------------------|-----------------|
| Network Timeout | 5 | Exponential (1s, 2s, 4s, 8s, 16s) | Yes (5 failures) |
| Rate Limit | 1 | Wait for reset window | No |
| Order Rejection | 0 | N/A | No |
| Data Feed | 3 | Linear (5s between) | Yes (3 failures) |

---

## 8. State Management Approach

### 8.1 State Machine Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BOT STATE MACHINE                               │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │   INITIAL   │
                         │   STATE     │
                         └──────┬──────┘
                                │
                                │ Initialize
                                ▼
                         ┌─────────────┐
              ┌─────────│   STARTING  │─────────┐
              │         │             │         │
              │         └──────┬──────┘         │
              │                │                │
              │                │ Success        │
              │                ▼                │
              │         ┌─────────────┐         │
              │         │    IDLE     │◀────────┤
              │         │  (Waiting   │         │
              │         │  for next   │         │
              │         │  candle)    │         │
              │         └──────┬──────┘         │
              │                │                │
              │                │ Candle Ready   │
              │                ▼                │
              │         ┌─────────────┐         │
              │         │  ANALYZING  │         │
              │         │  (Calculate │         │
              │         │  indicators)│         │
              │         └──────┬──────┘         │
              │                │                │
              │                ▼                │
              │         ┌─────────────┐         │
              │    ┌───│   SIGNAL    │───┐     │
              │    │    │  CHECK      │   │     │
              │    │    │             │   │     │
              │    │    └──────┬──────┘   │     │
              │    │           │          │     │
              │ No Signal      │          │     │
              │    │      Signal Found    │     │
              │    │           │          │     │
              │    │           ▼          │     │
              │    │    ┌─────────────┐   │     │
              │    └───▶│  EXECUTING  │◀──┘     │
              │         │  (Place     │         │
              │         │   orders)   │         │
              │         └──────┬──────┘         │
              │                │                │
              │                ▼                │
              │         ┌─────────────┐         │
              │         │  MONITORING │─────────┤
              │         │  (Track     │  Wait   │
              │         │   position) │         │
              │         └──────┬──────┘         │
              │                │                │
              │                │ Position       │
              │                │ Closed         │
              │                ▼                │
              │         ┌─────────────┐         │
              └────────▶│  STOPPING   │◀────────┘
                        │  (Cleanup)  │
                        └──────┬──────┘
                               │
                               │ Shutdown
                               ▼
                        ┌─────────────┐
                        │   STOPPED   │
                        └─────────────┘
                               │
                               │ Error
                               ▼
                        ┌─────────────┐
                        │    ERROR    │
                        │    STATE    │
                        └─────────────┘
```

### 8.2 State Persistence

```python
# State persistence structure (JSON)
{
    "version": "1.0.0",
    "last_updated": "2026-01-27T15:15:00Z",
    "bot_state": "MONITORING",
    
    "positions": {
        "open_positions": [
            {
                "id": "pos_001",
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 45000.00,
                "size": 0.1,
                "stop_loss": 44000.00,
                "take_profit": 47000.00,
                "open_time": "2026-01-27T10:00:00Z",
                "hedges": ["hedge_001"]
            }
        ],
        "pending_orders": [
            {
                "id": "ord_001",
                "position_id": "pos_001",
                "type": "stop_loss",
                "status": "open"
            }
        ]
    },
    
    "hedges": {
        "active_hedges": [
            {
                "id": "hedge_001",
                "position_id": "pos_001",
                "symbol": "ETH/USDT",
                "side": "short",
                "size": 0.05,
                "entry_price": 3200.00,
                "chunks_filled": 2,
                "total_chunks": 3
            }
        ]
    },
    
    "market_data": {
        "last_candle_timestamp": "2026-01-27T15:00:00Z",
        "subscribed_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "correlation_matrix": {
            "BTC/USDT": {
                "ETH/USDT": 0.85,
                "SOL/USDT": 0.72
            },
            "ETH/USDT": {
                "SOL/USDT": 0.78
            }
        }
    },
    
    "statistics": {
        "trades_today": 3,
        "daily_pnl": 150.50,
        "total_trades": 150,
        "win_rate": 0.62
    }
}
```

### 8.3 Recovery Procedures

| Scenario | Recovery Action | Data Loss Risk |
|----------|-----------------|----------------|
| Graceful Shutdown | Save state, close positions if configured | None |
| Crash During Trade | Recover state, check exchange positions, reconcile | Minimal |
| WebSocket Disconnect | Reconnect, resubscribe, request missed data | None (with buffer) |
| Database Corruption | Restore from backup, replay from exchange logs | Last few trades |
| API Key Invalid | Alert user, pause trading | None |

---

## 9. Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLASS HIERARCHY                                 │
└─────────────────────────────────────────────────────────────────────────┘

Base Classes:
┌─────────────────────────────────────────────────────────────────────────┐
│  BaseComponent (ABC)                                                    │
│  ├── initialize() -> None                                               │
│  ├── start() -> None                                                    │
│  ├── stop() -> None                                                     │
│  └── health_check() -> HealthStatus                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│  BaseStrategy │         │  BaseRiskMgr  │         │  BaseExchange │
│  (ABC)        │         │  (ABC)        │         │  (ABC)        │
├───────────────┤         ├───────────────┤         ├───────────────┤
│ generate()    │         │ calculate_    │         │ connect()     │
│ validate()    │         │   position_   │         │ disconnect()  │
└───────────────┘         │   size()      │         │ place_order() │
        │                 │ validate()    │         │ cancel_order()│
        │                 └───────────────┘         │ get_balance() │
        │                         │                 └───────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ TrendFollow   │         │ DeltaRiskMgr  │         │ DeltaExchange │
│ Strategy      │         │               │         │ Client        │
├───────────────┤         ├───────────────┤         ├───────────────┤
│ ema_periods   │         │ risk_per_     │         │ ccxt_exchange │
│ rsi_period    │         │   trade_pct   │         │ ws_client     │
│ atr_period    │         │ max_daily_    │         │ rate_limiter  │
│ entry_rules   │         │   loss_pct    │         │ retry_handler │
│ exit_rules    │         │ risk_reward   │         └───────────────┘
└───────────────┘         └───────────────┘

Main Classes:
┌─────────────────────────────────────────────────────────────────────────┐
│  TradingBot (Main Controller)                                         │
│  ├── config_manager: ConfigManager                                    │
│  ├── exchange_client: DeltaExchangeClient                             │
│  ├── market_data: MarketDataHandler                                   │
│  ├── signal_generator: TrendFollowingStrategy                         │
│  ├── risk_manager: DeltaRiskManager                                   │
│  ├── order_manager: OrderManager                                      │
│  ├── hedge_engine: HedgeEngine                                        │
│  ├── position_tracker: PositionTracker                                │
│  ├── state_manager: StateManager                                      │
│  ├── database: DatabaseManager                                        │
│  └── logger: LoggingService                                           │
│                                                                         │
│  Methods:                                                               │
│  ├── initialize() -> None                                               │
│  ├── run() -> None                                                      │
│  ├── stop() -> None                                                     │
│  ├── handle_signal(signal: Signal) -> None                              │
│  └── handle_error(error: Exception) -> None                             │
└─────────────────────────────────────────────────────────────────────────┘

Supporting Classes:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ConfigManager   │  │ StateManager    │  │ DatabaseManager │  │ LoggingService  │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ _config: dict   │  │ _state: BotState│  │ _connection     │  │ _logger         │
│ _env_vars: dict │  │ _persistence    │  │ _cursor         │  │ _handlers       │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ load()          │  │ set_state()     │  │ connect()       │  │ configure()     │
│ get()           │  │ get_state()     │  │ execute()       │  │ log_info()      │
│ validate()      │  │ persist()       │  │ query()         │  │ log_error()     │
│ reload()        │  │ recover()       │  │ backup()        │  │ log_trade()     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ MarketData      │  │ OrderManager    │  │ HedgeEngine     │  │ PositionTracker │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ _ohlcv_buffers  │  │ _pending_orders │  │ _active_hedges  │  │ _positions      │
│ _indicators     │  │ _exchange       │  │ _correlations   │ │ _pnl_cache      │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ subscribe()     │  │ place_entry()   │  │ monitor()       │  │ add_position()  │
│ get_ohlcv()     │  │ place_sl()      │  │ trigger_hedge() │  │ update_pnl()    │
│ calculate_ema() │  │ place_tp()      │  │ execute_chunk() │  │ close_position()│
│ calculate_rsi() │  │ cancel()        │  │ close_hedge()   │  │ get_open()      │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 10. Key Design Decisions

### 10.1 Why CCXT Pro?
- **Unified API**: Single interface for multiple exchanges
- **WebSocket Support**: Native async WebSocket implementation
- **Rate Limiting**: Built-in rate limit handling
- **Active Development**: Well-maintained with extensive documentation

### 10.2 Why SQLite?
- **Zero Configuration**: No separate server required
- **Sufficient for Scale**: Can handle millions of trade records
- **Portability**: Single file database easy to backup/restore
- **Python Native**: Built-in sqlite3 module

### 10.3 Why Async Architecture?
- **WebSocket Efficiency**: Handle multiple concurrent streams
- **Non-blocking I/O**: Continue processing while waiting for exchange
- **Better Resource Usage**: Single-threaded with event loop

### 10.4 Hedge Strategy Rationale
- **Correlation-Based**: Exploits natural relationships between crypto assets
- **Gradual Execution**: Reduces market impact and slippage
- **Dynamic Profit Taking**: Locks in hedge profits while maintaining protection
- **Multiple Hedges**: Scales protection as losses widen

---

## 11. Security Considerations

1. **API Key Management**: Store in environment variables, never in code
2. **IP Whitelisting**: Restrict API access to bot's IP address
3. **Minimal Permissions**: Use trading-only API keys (no withdrawal)
4. **Encryption**: Encrypt sensitive data at rest
5. **Audit Logging**: Log all trading actions for accountability

---

## 12. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Signal Latency | < 5 seconds | From candle close to signal generation |
| Order Execution | < 2 seconds | From signal to order submission |
| WebSocket Reconnect | < 10 seconds | Time to restore data feeds |
| Database Writes | < 100ms | Per transaction log entry |
| Memory Usage | < 512MB | For all active buffers and state |
| CPU Usage | < 10% | Average during normal operation |

---

## 13. Future Enhancements

1. **Multi-Exchange Support**: Extend beyond Delta Exchange
2. **Machine Learning**: Add ML-based signal filtering
3. **Portfolio Optimization**: Dynamic position sizing across pairs
4. **Backtesting Engine**: Historical simulation capability
5. **Web Dashboard**: Real-time monitoring UI
6. **Telegram/Discord Bot**: Trade notifications and control

---

## 14. Appendix: File Structure

```
delta-trading-bot/
├── config/
│   ├── config.yaml           # Main configuration
│   └── .env                  # Environment variables (not in git)
├── src/
│   ├── __init__.py
│   ├── bot.py                # Main trading bot class
│   ├── config_manager.py     # Configuration management
│   ├── exchange_client.py    # CCXT Pro wrapper
│   ├── market_data_handler.py # Data feed management
│   ├── signal_generator.py   # Strategy implementation
│   ├── risk_manager.py       # Risk calculations
│   ├── order_manager.py      # Order execution
│   ├── hedge_engine.py       # Hedge logic
│   ├── position_tracker.py   # Position monitoring
│   ├── state_manager.py      # State persistence
│   ├── database_manager.py   # SQLite interface
│   └── logging_service.py    # Logging configuration
├── data/
│   ├── trading.db            # SQLite database
│   ├── bot_state.json        # State persistence
│   └── backups/              # Database backups
├── logs/
│   └── trading_bot.log       # Application logs
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   └── architecture.md       # This document
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point
└── README.md                 # User documentation
```

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-27*
*Author: Trading Bot Architecture Team*
