# Backtesting System - Implementation Plan

## Document Information
- **Version**: 1.0
- **Date**: 2025-01-27
- **Purpose**: Detailed implementation plan for the backtesting system
- **Based On**: [`backtest_design.md`](backtest_design.md:1)

---

## Overview

This document provides a step-by-step implementation plan for building the backtesting system. The implementation is divided into 7 phases, each with specific tasks, dependencies, and acceptance criteria.

### Implementation Phases

1. **Phase 1: Core Infrastructure** - Foundation classes and configuration
2. **Phase 2: Mock Components** - Mock implementations of exchange, stream, and cache
3. **Phase 3: Backtest Engine** - Main orchestration engine
4. **Phase 4: State Tracking & Metrics** - Account state, trade history, performance metrics
5. **Phase 5: Reporting** - Trade logs, equity curves, reports
6. **Phase 6: Data Preparation** - Fetching and validating 2025 historical data
7. **Phase 7: Testing & Validation** - Unit tests, integration tests, optimization

---

## Phase 1: Core Infrastructure

### Objective
Create the foundational classes and configuration system for the backtesting framework.

### Tasks

#### 1.1 Create Backtest Module Structure
**File**: `src/backtest/__init__.py`

```python
"""
Backtesting module for the trading bot.

This module provides a complete backtesting framework that uses
exactly the same trading logic as the live bot, with mocked
data sources and order execution.
"""

from .backtest_config import BacktestConfig, BacktestConfigManager
from .time_controller import TimeController
from .account_state import AccountState

__all__ = [
    'BacktestConfig',
    'BacktestConfigManager',
    'TimeController',
    'AccountState',
]
```

**Acceptance Criteria**:
- Module structure created
- All necessary imports defined
- Module can be imported without errors

---

#### 1.2 Implement BacktestConfig
**File**: `src/backtest/backtest_config.py`

**Classes to Implement**:
- `BacktestConfig` (dataclass)
- `BacktestConfigManager`
- `SlippageModel` (enum)

**Key Methods**:
- `BacktestConfigManager._load_config()`
- `BacktestConfigManager.get_config()`

**Dependencies**: None

**Estimated Time**: 2 hours

**Acceptance Criteria**:
- Configuration can be loaded from YAML file
- All configuration parameters are accessible
- Default values are properly set
- Configuration validation works

**Testing**:
```python
def test_backtest_config_loading():
    config_manager = BacktestConfigManager("config/backtest_config.yaml")
    config = config_manager.get_config()
    assert config.start_date == datetime(2025, 1, 1)
    assert config.initial_balance == 10000.0
    assert len(config.symbols) > 0
```

---

#### 1.3 Implement TimeController
**File**: `src/backtest/time_controller.py`

**Classes to Implement**:
- `TimeController`

**Key Methods**:
- `__init__(start_date, end_date, timeframe)`
- `advance_to_next_candle() -> datetime`
- `get_current_time() -> datetime`
- `is_complete() -> bool`
- `_parse_timeframe(timeframe) -> int`

**Dependencies**: None

**Estimated Time**: 1 hour

**Acceptance Criteria**:
- Time advances correctly by candle interval
- Current time is always accessible
- Completion detection works
- Supports 15m timeframe

**Testing**:
```python
def test_time_controller():
    controller = TimeController(
        datetime(2025, 1, 1),
        datetime(2025, 1, 2),
        "15m"
    )
    assert controller.get_current_time() == datetime(2025, 1, 1)
    controller.advance_to_next_candle()
    assert controller.get_current_time() == datetime(2025, 1, 1, 0, 15)
```

---

#### 1.4 Implement AccountState
**File**: `src/backtest/account_state.py`

**Classes to Implement**:
- `AccountState` (dataclass)

**Key Methods**:
- `get_balance() -> Balance`
- `get_positions(symbol=None) -> List[Position]`
- `process_fill(order, fill_price, fee)`
- `update_unrealized_pnl(current_prices)`
- `_update_total_balance()`
- `_generate_position_id() -> str`
- `_find_position_to_close(symbol) -> Optional[Position]`

**Dependencies**: 
- `src/risk/portfolio_tracker.py` (for Position class)

**Estimated Time**: 3 hours

**Acceptance Criteria**:
- Balance tracking works correctly
- Positions are opened and closed properly
- P&L is calculated accurately
- Fees are deducted correctly
- Unrealized P&L updates with current prices

**Testing**:
```python
def test_account_state():
    account = AccountState(
        initial_balance=10000.0,
        currency="USDT"
    )
    assert account.available_balance == 10000.0
    
    # Simulate opening a position
    order = Order(id="1", symbol="BTC/USDT", side="buy", 
                  amount=0.1, price=50000.0)
    account.process_fill(order, 50000.0, 10.0)
    assert account.available_balance < 10000.0
    assert len(account.positions) == 1
```

---

### Phase 1 Summary
**Total Estimated Time**: 6 hours
**Dependencies**: None
**Deliverables**:
- `src/backtest/__init__.py`
- `src/backtest/backtest_config.py`
- `src/backtest/time_controller.py`
- `src/backtest/account_state.py`

---

## Phase 2: Mock Components

### Objective
Implement mock versions of exchange, stream manager, and data cache that serve historical data.

### Tasks

#### 2.1 Create Mock Module Structure
**File**: `src/backtest/mock/__init__.py`

```python
"""
Mock components for backtesting.

These components implement the same interfaces as the live components
but serve historical data and simulate order execution.
"""

from .backtest_exchange_client import BacktestExchangeClient
from .backtest_stream_manager import BacktestStreamManager
from .backtest_data_cache import BacktestDataCache
from .backtest_order_simulator import BacktestOrderSimulator

__all__ = [
    'BacktestExchangeClient',
    'BacktestStreamManager',
    'BacktestDataCache',
    'BacktestOrderSimulator',
]
```

---

#### 2.2 Implement BacktestOrderSimulator
**File**: `src/backtest/mock/backtest_order_simulator.py`

**Classes to Implement**:
- `BacktestOrderSimulator`

**Key Methods**:
- `__init__(config, account_state)`
- `submit_order(order)`
- `cancel_order(order_id) -> Order`
- `get_order(order_id) -> Order`
- `process_orders(candle)`
- `_should_fill_order(order, candle) -> bool`
- `_fill_order(order, candle)`
- `_generate_order_id() -> str`

**Dependencies**:
- `src/backtest/backtest_config.py` (BacktestConfig)
- `src/backtest/account_state.py` (AccountState)
- `src/exchange/exchange_client.py` (Order class)

**Estimated Time**: 4 hours

**Acceptance Criteria**:
- Orders are tracked correctly
- Limit orders fill when price crosses
- Slippage is applied correctly
- Fees are calculated accurately
- Order cancellation works
- Partial fills are handled (if enabled)

**Testing**:
```python
def test_order_simulator():
    config = BacktestConfig()
    account = AccountState(10000.0, "USDT")
    simulator = BacktestOrderSimulator(config, account)
    
    # Submit a buy limit order
    order = Order(id="1", symbol="BTC/USDT", side="buy",
                  type="limit", amount=0.1, price=50000.0)
    simulator.submit_order(order)
    
    # Process a candle that should fill the order
    candle = OHLCV(
        symbol="BTC/USDT",
        timestamp=datetime(2025, 1, 1),
        open=50100.0,
        high=50200.0,
        low=49900.0,  # Below order price
        close=50050.0,
        volume=1000.0
    )
    simulator.process_orders(candle)
    
    # Check order was filled
    filled_order = simulator.get_order("1")
    assert filled_order.status == "filled"
```

---

#### 2.3 Implement BacktestDataCache
**File**: `src/backtest/mock/backtest_data_cache.py`

**Classes to Implement**:
- `BacktestDataCache`

**Key Methods**:
- `__init__(historical_data)`
- `update_ohlcv(symbol, candle)`
- `get_ohlcv(symbol, timeframe, limit=200) -> List[OHLCV]`
- `get_latest_price(symbol) -> float`
- `get_ticker(symbol) -> Ticker`
- `set_current_time(current_time)`

**Dependencies**:
- `src/data/data_cache.py` (DataCache base class)
- `src/data/data_cache.py` (OHLCV, Ticker classes)

**Estimated Time**: 2 hours

**Acceptance Criteria**:
- Historical data is stored correctly
- Data is filtered by current time
- Latest price is returned correctly
- Ticker data is generated correctly
- Cache updates work properly

**Testing**:
```python
def test_backtest_data_cache():
    historical_data = {
        "BTC/USDT": [
            OHLCV(symbol="BTC/USDT", timestamp=datetime(2025, 1, 1),
                  open=50000.0, high=50100.0, low=49900.0,
                  close=50050.0, volume=1000.0)
        ]
    }
    cache = BacktestDataCache(historical_data)
    cache.set_current_time(datetime(2025, 1, 1))
    
    price = cache.get_latest_price("BTC/USDT")
    assert price == 50050.0
```

---

#### 2.4 Implement BacktestStreamManager
**File**: `src/backtest/mock/backtest_stream_manager.py`

**Classes to Implement**:
- `BacktestStreamManager`

**Key Methods**:
- `__init__(historical_data, data_cache)`
- `start()`
- `stop()`
- `subscribe_symbols(symbols)`
- `update_time(current_time)`
- `watch_ohlcv(symbol, timeframe)` (async generator)
- `watch_ticker(symbol)` (async generator)

**Dependencies**:
- `src/data/stream_manager.py` (StreamManager base class)
- `src/backtest/mock/backtest_data_cache.py` (BacktestDataCache)

**Estimated Time**: 3 hours

**Acceptance Criteria**:
- Symbols can be subscribed
- Time updates push data to cache
- Async generators yield correct data
- Start/stop methods work

**Testing**:
```python
async def test_backtest_stream_manager():
    historical_data = {
        "BTC/USDT": [
            OHLCV(symbol="BTC/USDT", timestamp=datetime(2025, 1, 1),
                  open=50000.0, high=50100.0, low=49900.0,
                  close=50050.0, volume=1000.0)
        ]
    }
    cache = BacktestDataCache(historical_data)
    stream = BacktestStreamManager(historical_data, cache)
    
    await stream.subscribe_symbols(["BTC/USDT"])
    await stream.update_time(datetime(2025, 1, 1))
    
    # Check data was pushed to cache
    price = cache.get_latest_price("BTC/USDT")
    assert price == 50050.0
```

---

#### 2.5 Implement BacktestExchangeClient
**File**: `src/backtest/mock/backtest_exchange_client.py`

**Classes to Implement**:
- `BacktestExchangeClient`

**Key Methods**:
- `__init__(historical_data, order_simulator, account_state)`
- `fetch_ohlcv(symbol, timeframe, limit=200) -> List[OHLCV]`
- `fetch_ticker(symbol) -> Ticker`
- `fetch_balance() -> Balance`
- `fetch_positions(symbol=None) -> List[Position]`
- `create_order(symbol, side, order_type, amount, price, params) -> Order`
- `cancel_order(order_id, symbol) -> Order`
- `fetch_order(order_id, symbol) -> Order`
- `close()`

**Dependencies**:
- `src/exchange/exchange_client.py` (ExchangeClient base class)
- `src/backtest/mock/backtest_order_simulator.py` (BacktestOrderSimulator)
- `src/backtest/account_state.py` (AccountState)

**Estimated Time**: 3 hours

**Acceptance Criteria**:
- Historical OHLCV data is returned correctly
- Ticker data is generated correctly
- Balance is returned from account state
- Positions are returned from account state
- Orders are submitted to simulator
- Order cancellation works
- Order status can be queried

**Testing**:
```python
async def test_backtest_exchange_client():
    historical_data = {
        "BTC/USDT": [
            OHLCV(symbol="BTC/USDT", timestamp=datetime(2025, 1, 1),
                  open=50000.0, high=50100.0, low=49900.0,
                  close=50050.0, volume=1000.0)
        ]
    }
    config = BacktestConfig()
    account = AccountState(10000.0, "USDT")
    simulator = BacktestOrderSimulator(config, account)
    
    exchange = BacktestExchangeClient(historical_data, simulator, account)
    exchange.current_time = datetime(2025, 1, 1)
    
    # Test fetch_ohlcv
    ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "15m", 200)
    assert len(ohlcv) > 0
    
    # Test fetch_ticker
    ticker = await exchange.fetch_ticker("BTC/USDT")
    assert ticker.last == 50050.0
    
    # Test fetch_balance
    balance = await exchange.fetch_balance()
    assert balance.total == 10000.0
```

---

### Phase 2 Summary
**Total Estimated Time**: 12 hours
**Dependencies**: Phase 1
**Deliverables**:
- `src/backtest/mock/__init__.py`
- `src/backtest/mock/backtest_order_simulator.py`
- `src/backtest/mock/backtest_data_cache.py`
- `src/backtest/mock/backtest_stream_manager.py`
- `src/backtest/mock/backtest_exchange_client.py`

---

## Phase 3: Backtest Engine

### Objective
Implement the main backtest engine that orchestrates the entire backtesting process.

### Tasks

#### 3.1 Implement HistoricalDataLoader
**File**: `src/backtest/historical_data_loader.py`

**Classes to Implement**:
- `HistoricalDataLoader`
- `DataPreprocessor` (static methods)

**Key Methods**:
- `__init__(config)`
- `load_data() -> Dict[str, List[OHLCV]]`
- `_load_from_csv() -> Dict[str, List[OHLCV]]`
- `_parse_csv(filename) -> List[OHLCV]`
- `_load_from_sqlite() -> Dict[str, List[OHLCV]]`
- `_fetch_from_api() -> Dict[str, List[OHLCV]]`
- `validate_data(data) -> bool`

**Dependencies**:
- `src/backtest/backtest_config.py` (BacktestConfig)
- `src/data/data_cache.py` (OHLCV class)

**Estimated Time**: 4 hours

**Acceptance Criteria**:
- Data can be loaded from CSV
- Data can be loaded from SQLite
- Data is parsed correctly
- Data validation works
- Timeframe alignment is checked

**Testing**:
```python
def test_historical_data_loader():
    config = BacktestConfig(
        data_source="csv",
        data_path="data/historical/2025/"
    )
    loader = HistoricalDataLoader(config)
    data = loader.load_data()
    
    assert "BTC/USDT" in data
    assert len(data["BTC/USDT"]) > 0
    assert loader.validate_data(data)
```

---

#### 3.2 Implement BacktestEngine
**File**: `src/backtest/backtest_engine.py`

**Classes to Implement**:
- `BacktestEngine`
- `BacktestResults` (dataclass)

**Key Methods**:
- `__init__(config)`
- `run_backtest() -> BacktestResults`
- `_initialize_components()`
- `_load_historical_data()`
- `_set_initial_account_state()`
- `_main_backtest_loop()`
- `_update_market_data()`
- `_process_pending_orders()`
- `_update_account_state()`
- `_record_equity_point()`
- `_generate_results() -> BacktestResults`

**Dependencies**:
- All Phase 1 and Phase 2 components
- Existing trading bot components (unchanged)

**Estimated Time**: 8 hours

**Acceptance Criteria**:
- All components are initialized correctly
- Historical data is loaded
- Main loop iterates through all candles
- Trading cycles are executed at each candle
- Orders are processed correctly
- Account state is updated
- Equity curve is recorded
- Results are generated correctly

**Testing**:
```python
async def test_backtest_engine():
    config = BacktestConfig(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 2),
        symbols=["BTC/USDT"],
        initial_balance=10000.0
    )
    engine = BacktestEngine(config)
    results = await engine.run_backtest()
    
    assert results is not None
    assert results.metrics.total_trades >= 0
    assert len(results.equity_curve.points) > 0
```

---

### Phase 3 Summary
**Total Estimated Time**: 12 hours
**Dependencies**: Phase 1, Phase 2
**Deliverables**:
- `src/backtest/historical_data_loader.py`
- `src/backtest/backtest_engine.py`

---

## Phase 4: State Tracking & Metrics

### Objective
Implement trade history, equity curve tracking, and performance metrics calculation.

### Tasks

#### 4.1 Implement TradeHistory
**File**: `src/backtest/trade_history.py`

**Classes to Implement**:
- `TradeRecord` (dataclass)
- `TradeHistory`

**Key Methods**:
- `TradeRecord.to_dict() -> dict`
- `TradeHistory.__init__()`
- `TradeHistory.add_trade(position, exit_price, exit_time, exit_reason)`
- `TradeHistory.get_trades(symbol=None) -> List[TradeRecord]`
- `TradeHistory.to_dataframe() -> pd.DataFrame`

**Dependencies**:
- `src/risk/portfolio_tracker.py` (Position class)

**Estimated Time**: 2 hours

**Acceptance Criteria**:
- Trades are recorded correctly
- Trades can be filtered by symbol
- DataFrame conversion works

**Testing**:
```python
def test_trade_history():
    history = TradeHistory()
    position = Position(
        id="1", symbol="BTC/USDT", side="long",
        entry_price=50000.0, size=0.1,
        open_time=datetime(2025, 1, 1)
    )
    
    history.add_trade(position, 51000.0, datetime(2025, 1, 2), "take_profit")
    assert len(history.trades) == 1
    assert history.trades[0].pnl == 100.0
```

---

#### 4.2 Implement EquityCurve
**File**: `src/backtest/equity_curve.py`

**Classes to Implement**:
- `EquityPoint` (dataclass)
- `EquityCurve`

**Key Methods**:
- `EquityCurve.__init__()`
- `EquityCurve.add_point(timestamp, account_state)`
- `EquityCurve.to_dataframe() -> pd.DataFrame`

**Dependencies**:
- `src/backtest/account_state.py` (AccountState)

**Estimated Time**: 1 hour

**Acceptance Criteria**:
- Equity points are recorded correctly
- DataFrame conversion works

**Testing**:
```python
def test_equity_curve():
    curve = EquityCurve()
    account = AccountState(10000.0, "USDT")
    
    curve.add_point(datetime(2025, 1, 1), account)
    assert len(curve.points) == 1
    assert curve.points[0].equity == 10000.0
```

---

#### 4.3 Implement PerformanceCalculator
**File**: `src/backtest/performance_calculator.py`

**Classes to Implement**:
- `PerformanceMetrics` (dataclass)
- `PerformanceCalculator`

**Key Methods**:
- `PerformanceCalculator.__init__(account_state, trade_history, equity_curve)`
- `PerformanceCalculator.calculate_metrics() -> PerformanceMetrics`
- `_calculate_total_return() -> float`
- `_calculate_annualized_return(equity_df) -> float`
- `_calculate_sharpe_ratio(equity_df) -> float`
- `_calculate_sortino_ratio(equity_df) -> float`
- `_calculate_max_drawdown(equity_df) -> float`
- `_calculate_max_drawdown_duration(equity_df) -> timedelta`
- `_calculate_win_rate(trades_df) -> float`
- `_calculate_avg_win(trades_df) -> float`
- `_calculate_avg_loss(trades_df) -> float`
- `_calculate_profit_factor(trades_df) -> float`
- `_calculate_avg_trade(trades_df) -> float`
- `_calculate_avg_exposure_time(trades_df) -> timedelta`
- `_calculate_calmar_ratio() -> float`
- `_calculate_recovery_factor() -> float`

**Dependencies**:
- `src/backtest/account_state.py` (AccountState)
- `src/backtest/trade_history.py` (TradeHistory)
- `src/backtest/equity_curve.py` (EquityCurve)
- pandas, numpy

**Estimated Time**: 4 hours

**Acceptance Criteria**:
- All metrics are calculated correctly
- Edge cases are handled (no trades, zero volatility, etc.)
- Metrics match expected values

**Testing**:
```python
def test_performance_calculator():
    account = AccountState(10000.0, "USDT")
    account.realized_pnl = 1000.0
    account.total_balance = 11000.0
    
    history = TradeHistory()
    curve = EquityCurve()
    
    calculator = PerformanceCalculator(account, history, curve)
    metrics = calculator.calculate_metrics()
    
    assert metrics.total_return == 10.0
    assert metrics.total_pnl == 1000.0
```

---

### Phase 4 Summary
**Total Estimated Time**: 7 hours
**Dependencies**: Phase 1, Phase 2, Phase 3
**Deliverables**:
- `src/backtest/trade_history.py`
- `src/backtest/equity_curve.py`
- `src/backtest/performance_calculator.py`

---

## Phase 5: Reporting

### Objective
Implement trade logging, equity curve visualization, and report generation.

### Tasks

#### 5.1 Implement TradeLogger
**File**: `src/backtest/trade_logger.py`

**Classes to Implement**:
- `TradeLogger`

**Key Methods**:
- `__init__(output_dir)`
- `log_trade(trade)`
- `save_to_csv(filename="trade_log.csv")`
- `save_to_json(filename="trade_log.json")`

**Dependencies**:
- `src/backtest/trade_history.py` (TradeRecord)
- pandas, json

**Estimated Time**: 1 hour

**Acceptance Criteria**:
- Trades are logged correctly
- CSV export works
- JSON export works

**Testing**:
```python
def test_trade_logger():
    logger = TradeLogger("output/")
    trade = TradeRecord(
        trade_id="1", symbol="BTC/USDT", side="long",
        entry_price=50000.0, exit_price=51000.0,
        size=0.1, entry_time=datetime(2025, 1, 1),
        exit_time=datetime(2025, 1, 2), pnl=100.0,
        fees=10.0, duration=timedelta(days=1),
        exit_reason="take_profit"
    )
    
    logger.log_trade(trade)
    logger.save_to_csv()
    
    # Check file exists
    assert os.path.exists("output/trade_log.csv")
```

---

#### 5.2 Implement EquityCurveGenerator
**File**: `src/backtest/equity_curve_generator.py`

**Classes to Implement**:
- `EquityCurveGenerator`

**Key Methods**:
- `__init__(equity_curve, output_dir)`
- `generate_plot(filename="equity_curve.png")`
- `generate_drawdown_plot(filename="drawdown.png")`

**Dependencies**:
- `src/backtest/equity_curve.py` (EquityCurve)
- matplotlib

**Estimated Time**: 2 hours

**Acceptance Criteria**:
- Equity curve plot is generated
- Drawdown plot is generated
- Plots are saved correctly

**Testing**:
```python
def test_equity_curve_generator():
    curve = EquityCurve()
    account = AccountState(10000.0, "USDT")
    curve.add_point(datetime(2025, 1, 1), account)
    
    generator = EquityCurveGenerator(curve, "output/")
    generator.generate_plot()
    
    # Check file exists
    assert os.path.exists("output/equity_curve.png")
```

---

#### 5.3 Implement ReportGenerator
**File**: `src/backtest/report_generator.py`

**Classes to Implement**:
- `ReportGenerator`

**Key Methods**:
- `__init__(metrics, trade_history, equity_curve, output_dir)`
- `generate_html_report(filename="backtest_report.html")`
- `generate_json_report(filename="backtest_report.json")`

**Dependencies**:
- `src/backtest/performance_calculator.py` (PerformanceMetrics)
- `src/backtest/trade_history.py` (TradeHistory)
- `src/backtest/equity_curve.py` (EquityCurve)
- json

**Estimated Time**: 3 hours

**Acceptance Criteria**:
- HTML report is generated
- JSON report is generated
- Reports contain all metrics
- Reports are formatted correctly

**Testing**:
```python
def test_report_generator():
    metrics = PerformanceMetrics(...)
    history = TradeHistory()
    curve = EquityCurve()
    
    generator = ReportGenerator(metrics, history, curve, "output/")
    generator.generate_html_report()
    
    # Check file exists
    assert os.path.exists("output/backtest_report.html")
```

---

### Phase 5 Summary
**Total Estimated Time**: 6 hours
**Dependencies**: Phase 4
**Deliverables**:
- `src/backtest/trade_logger.py`
- `src/backtest/equity_curve_generator.py`
- `src/backtest/report_generator.py`

---

## Phase 6: Data Preparation

### Objective
Fetch, validate, and store 2025 historical data for all trading pairs.

### Tasks

#### 6.1 Create Data Fetching Script
**File**: `scripts/fetch_historical_data.py`

**Functionality**:
- Fetch OHLCV data from Delta Exchange API
- Fetch data for all symbols in config
- Fetch data for entire 2025 year
- Save to CSV format
- Handle rate limits and retries

**Dependencies**:
- ccxt
- `src/config/config_manager.py`

**Estimated Time**: 4 hours

**Acceptance Criteria**:
- Data is fetched for all symbols
- Data covers entire 2025 year
- Data is saved to CSV format
- Rate limits are respected
- Errors are handled gracefully

**Usage**:
```bash
python scripts/fetch_historical_data.py --start 2025-01-01 --end 2025-12-31 --symbols BTC/USDT,ETH/USDT,SOL/USDT
```

---

#### 6.2 Create Data Validation Script
**File**: `scripts/validate_historical_data.py`

**Functionality**:
- Validate CSV files for all symbols
- Check for missing candles
- Check timeframe alignment
- Check data quality (no negative prices, etc.)
- Generate validation report

**Dependencies**:
- pandas

**Estimated Time**: 2 hours

**Acceptance Criteria**:
- All CSV files are validated
- Missing candles are detected
- Timeframe misalignments are detected
- Data quality issues are reported

**Usage**:
```bash
python scripts/validate_historical_data.py --data-dir data/historical/2025/
```

---

#### 6.3 Create Data Preprocessing Script
**File**: `scripts/preprocess_historical_data.py`

**Functionality**:
- Sort data by timestamp
- Fill gaps (optional)
- Remove duplicates
- Calculate pre-computed indicators (optional)
- Save to SQLite (optional)

**Dependencies**:
- pandas
- sqlite3

**Estimated Time**: 2 hours

**Acceptance Criteria**:
- Data is sorted correctly
- Gaps can be filled
- Duplicates are removed
- SQLite database is created (optional)

**Usage**:
```bash
python scripts/preprocess_historical_data.py --input-dir data/historical/2025/ --output-dir data/historical/2025/processed/
```

---

### Phase 6 Summary
**Total Estimated Time**: 8 hours
**Dependencies**: None (can run in parallel with other phases)
**Deliverables**:
- `scripts/fetch_historical_data.py`
- `scripts/validate_historical_data.py`
- `scripts/preprocess_historical_data.py`
- Historical data files in `data/historical/2025/`

---

## Phase 7: Testing & Validation

### Objective
Create comprehensive tests and validate the backtesting system.

### Tasks

#### 7.1 Create Unit Tests
**File**: `tests/backtest/`

**Test Files to Create**:
- `test_backtest_config.py`
- `test_time_controller.py`
- `test_account_state.py`
- `test_order_simulator.py`
- `test_data_cache.py`
- `test_stream_manager.py`
- `test_exchange_client.py`
- `test_trade_history.py`
- `test_equity_curve.py`
- `test_performance_calculator.py`

**Estimated Time**: 8 hours

**Acceptance Criteria**:
- All components have unit tests
- Test coverage > 80%
- All tests pass

---

#### 7.2 Create Integration Tests
**File**: `tests/backtest/test_integration.py`

**Test Scenarios**:
- Full backtest with single symbol
- Full backtest with multiple symbols
- Backtest with no trades
- Backtest with many trades
- Backtest with hedging enabled
- Backtest with hedging disabled

**Estimated Time**: 4 hours

**Acceptance Criteria**:
- All integration tests pass
- Backtest completes without errors
- Results are reasonable

---

#### 7.3 Create Performance Tests
**File**: `tests/backtest/test_performance.py`

**Test Scenarios**:
- Backtest speed (candles per second)
- Memory usage
- Large dataset handling

**Estimated Time**: 2 hours

**Acceptance Criteria**:
- Backtest runs in reasonable time
- Memory usage is acceptable
- Large datasets are handled

---

#### 7.4 Validate Against Live Trading (Optional)
**File**: `scripts/compare_live_vs_backtest.py`

**Functionality**:
- Compare backtest results with live trading results
- Identify discrepancies
- Generate comparison report

**Estimated Time**: 4 hours

**Acceptance Criteria**:
- Comparison report is generated
- Discrepancies are identified
- Backtest behavior matches live behavior

---

### Phase 7 Summary
**Total Estimated Time**: 18 hours
**Dependencies**: All previous phases
**Deliverables**:
- Unit tests in `tests/backtest/`
- Integration tests
- Performance tests
- Comparison script (optional)

---

## Implementation Timeline

### Week 1: Core Infrastructure & Mock Components
- Days 1-2: Phase 1 (Core Infrastructure) - 6 hours
- Days 3-5: Phase 2 (Mock Components) - 12 hours

### Week 2: Backtest Engine & State Tracking
- Days 1-2: Phase 3 (Backtest Engine) - 12 hours
- Days 3-4: Phase 4 (State Tracking & Metrics) - 7 hours

### Week 3: Reporting & Data Preparation
- Days 1-2: Phase 5 (Reporting) - 6 hours
- Days 3-5: Phase 6 (Data Preparation) - 8 hours

### Week 4: Testing & Validation
- Days 1-3: Phase 7 (Testing & Validation) - 18 hours
- Days 4-5: Buffer for bug fixes and improvements

**Total Estimated Time**: 69 hours (~2 weeks of full-time work)

---

## Dependencies & Prerequisites

### External Dependencies
- Python 3.9+
- pandas
- numpy
- matplotlib
- ccxt
- pyyaml
- pytest

### Internal Dependencies
- Existing trading bot components (unchanged)
- Configuration files
- Database schema

### Data Requirements
- 2025 historical OHLCV data for all trading pairs
- Data must be in 15-minute timeframe
- Data must cover entire year (Jan 1 - Dec 31)

---

## Risk Mitigation

### Potential Risks
1. **Data Quality Issues**
   - Mitigation: Comprehensive validation scripts
   - Fallback: Use multiple data sources

2. **Performance Issues**
   - Mitigation: Performance testing and optimization
   - Fallback: Use SQLite instead of CSV for large datasets

3. **Logic Discrepancies**
   - Mitigation: Compare with live trading results
   - Fallback: Detailed logging and debugging

4. **Time Constraints**
   - Mitigation: Prioritize core functionality
   - Fallback: Defer advanced features to later phases

---

## Success Criteria

The backtesting system will be considered successful when:

1. ✅ All components are implemented according to design
2. ✅ Unit tests pass with >80% coverage
3. ✅ Integration tests pass
4. ✅ Backtest runs on 2025 data without errors
5. ✅ Performance metrics are calculated correctly
6. ✅ Reports are generated in HTML and JSON formats
7. ✅ Backtest behavior matches live trading behavior (validated)

---

## Next Steps

1. Review this implementation plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Follow phases sequentially
5. Conduct testing at each phase
6. Deploy and validate

---

**Document End**
