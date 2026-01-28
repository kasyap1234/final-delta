# Trading Bot Backtesting System - Technical Design Document

## Document Information
- **Version**: 1.0
- **Date**: 2025-01-27
- **Purpose**: Design a backtesting system that exactly mirrors the live trading bot behavior
- **Target Year**: 2025 historical data

---

## 1. Executive Summary

This document outlines the design for a comprehensive backtesting system that will execute the trading bot's strategy on 2025 historical data. The backtester uses **exactly the same trading logic** as the live bot, with only the data sources and order execution components mocked to simulate historical market conditions.

### Key Design Principles
1. **Zero Logic Modification**: All trading logic (SignalDetector, IndicatorManager, RiskManager, HedgeManager) remains unchanged
2. **Component Injection**: Mock components implement the same interfaces as live components
3. **Time Control**: Backtester controls time progression through historical data
4. **Realistic Simulation**: Order fills, slippage, and fees modeled accurately
5. **Complete State Tracking**: All positions, orders, and P&L tracked throughout the backtest

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKTESTING SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Backtest      │    │   Historical    │    │   Backtest      │         │
│  │   Engine        │    │   Data Loader   │    │   Config        │         │
│  │   (Time Control)│    │   (2025 Data)   │    │   Manager       │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                  │
│           └──────────────────────┼──────────────────────┘                  │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MOCKED COMPONENTS LAYER                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │Backtest     │  │Backtest     │  │Backtest     │  │Backtest   │  │   │
│  │  │Exchange     │  │Stream       │  │DataCache    │  │Order      │  │   │
│  │  │Client       │  │Manager      │  │             │  │Simulator  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    UNCHANGED TRADING LOGIC                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │   Signal    │  │ Indicator   │  │   Risk      │  │   Hedge    │  │   │
│  │  │   Detector  │  │   Manager   │  │   Manager   │  │   Manager  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │   Position  │  │   Order     │  │   Portfolio │  │   State    │  │   │
│  │  │   Sizer     │  │   Manager   │  │   Tracker   │  │   Manager  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                         │
│                                  ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    BACKTEST RESULTS                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │   Trade     │  │   Equity    │  │Performance  │  │   Report   │  │   │
│  │  │   Log       │  │   Curve     │  │  Metrics    │  │ Generator │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Inheritance Hierarchy

```
ExchangeClient (Base)
    │
    └── BacktestExchangeClient (Mock)

StreamManager (Base)
    │
    └── BacktestStreamManager (Mock)

DataCache (Base)
    │
    └── BacktestDataCache (Mock)

OrderExecutor (Base)
    │
    └── BacktestOrderSimulator (Mock)
```

---

## 3. Backtester Architecture

### 3.1 Class Structure

#### 3.1.1 BacktestEngine

```python
class BacktestEngine:
    """
    Main backtesting engine that orchestrates the backtest.
    
    Responsibilities:
    - Load historical data for 2025
    - Initialize all components (mocked and real)
    - Control time progression through historical data
    - Execute trading cycles at each candle
    - Track all state and metrics
    - Generate final reports
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_time: datetime = None
        self.historical_data: Dict[str, List[OHLCV]] = {}
        self.mock_exchange: BacktestExchangeClient = None
        self.mock_stream: BacktestStreamManager = None
        self.mock_cache: BacktestDataCache = None
        self.order_simulator: BacktestOrderSimulator = None
        
        # Real trading components (unchanged)
        self.trading_bot: TradingBot = None
        self.indicator_manager: IndicatorManager = None
        self.signal_detector: SignalDetector = None
        self.risk_manager: RiskManager = None
        self.hedge_manager: HedgeManager = None
        
        # State tracking
        self.account_state: AccountState = None
        self.trade_history: List[TradeRecord] = []
        self.equity_curve: List[EquityPoint] = []
        self.performance_metrics: PerformanceMetrics = None
    
    async def run_backtest(self) -> BacktestResults:
        """Execute the complete backtest on 2025 data."""
        pass
```

#### 3.1.2 BacktestConfig

```python
@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Date range
    start_date: datetime = datetime(2025, 1, 1)
    end_date: datetime = datetime(2025, 12, 31, 23, 59, 59)
    
    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    
    # Timeframe
    timeframe: str = "15m"
    
    # Initial account state
    initial_balance: float = 10000.0
    initial_currency: str = "USDT"
    
    # Order simulation
    slippage_model: SlippageModel = SlippageModel.PERCENTAGE
    slippage_pct: float = 0.05  # 0.05% slippage
    maker_fee: float = 0.02  # 0.02% maker fee
    taker_fee: float = 0.06  # 0.06% taker fee
    
    # Data source
    data_source: str = "csv"  # csv, sqlite, or api
    data_path: str = "data/historical/2025/"
    
    # Output
    output_dir: str = "backtest_results/"
    save_trade_log: bool = True
    save_equity_curve: bool = True
    generate_report: bool = True
    
    # Trading bot config (same as live)
    trading_bot_config: TradingBotConfig = None
```

### 3.2 Time Control Mechanism

The backtester controls time progression through historical data:

```python
class TimeController:
    """
    Controls time progression during backtesting.
    
    The backtester iterates through historical candles chronologically,
    advancing time to each candle's timestamp and executing the trading
    logic at that point in time.
    """
    
    def __init__(self, start_date: datetime, end_date: datetime, timeframe: str):
        self.current_time = start_date
        self.end_time = end_date
        self.timeframe = timeframe
        self.candle_interval = self._parse_timeframe(timeframe)
    
    def advance_to_next_candle(self) -> datetime:
        """Advance time to the next candle."""
        self.current_time += timedelta(minutes=self.candle_interval)
        return self.current_time
    
    def get_current_time(self) -> datetime:
        """Get the current simulation time."""
        return self.current_time
    
    def is_complete(self) -> bool:
        """Check if backtest is complete."""
        return self.current_time >= self.end_time
```

### 3.3 Backtest Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKTEST EXECUTION FLOW                          │
└─────────────────────────────────────────────────────────────────────────┘

Load Historical Data (2025)
        │
        ▼
┌─────────────┐
│ Initialize  │
│ Components  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Set Initial │
│ Account     │
│ State       │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MAIN BACKTEST LOOP                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐      │
│  │  Get Next   │────▶│  Update Mock     │────▶│  Execute        │      │
│  │  Candle     │     │  Components with  │     │  Trading Cycle  │      │
│  │  (Time)     │     │  Current Data    │     │                 │      │
│  └─────────────┘     └──────────────────┘     └────────┬────────┘      │
│                                                     │                  │
│                                                     ▼                  │
│                                          ┌─────────────────┐            │
│                                          │  Process        │            │
│                                          │  Pending Orders │            │
│                                          │  (Fill/Cancel)  │            │
│                                          └────────┬────────┘            │
│                                                     │                  │
│                                                     ▼                  │
│                                          ┌─────────────────┐            │
│                                          │  Update         │            │
│                                          │  Account State  │            │
│                                          │  (P&L, Balance) │            │
│                                          └────────┬────────┘            │
│                                                     │                  │
│                                                     ▼                  │
│                                          ┌─────────────────┐            │
│                                          │  Record         │            │
│                                          │  Equity Point   │            │
│                                          └─────────────────┘            │
│                                                     │                  │
│                                                     ▼                  │
│                                          ┌─────────────────┐            │
│                                          │  Check if       │            │
│                                          │  End of Data?   │            │
│                                          └────────┬────────┘            │
│                                                     │                  │
│                               ┌─────────────────────┼─────────────────┐ │
│                               │                     │                 │ │
│                               ▼                     ▼                 │ │
│                        ┌─────────────┐      ┌─────────────┐          │ │
│                        │  Continue   │      │  Generate   │          │ │
│                        │  Loop       │      │  Reports    │          │ │
│                        └─────────────┘      └─────────────┘          │ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Mock Components Design

### 4.1 BacktestExchangeClient

```python
class BacktestExchangeClient(ExchangeClient):
    """
    Mock exchange client that serves historical data and simulates order execution.
    
    This class implements the same interface as ExchangeClient but:
    - Returns historical OHLCV data instead of fetching from exchange
    - Simulates order fills based on historical price action
    - Tracks simulated account balance and positions
    - Does not make any network calls
    """
    
    def __init__(self, historical_data: Dict[str, List[OHLCV]], 
                 order_simulator: BacktestOrderSimulator,
                 account_state: AccountState):
        self.historical_data = historical_data
        self.order_simulator = order_simulator
        self.account_state = account_state
        self.current_time: datetime = None
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, 
                         limit: int = 200) -> List[OHLCV]:
        """
        Return historical OHLCV data up to current_time.
        
        The data is sliced from the pre-loaded historical dataset,
        returning only candles that have occurred up to the current
        simulation time.
        """
        data = self.historical_data.get(symbol, [])
        # Filter data up to current_time
        return [candle for candle in data if candle.timestamp <= self.current_time][-limit:]
    
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Return ticker data for the current time.
        
        Uses the latest candle's close price as the current price.
        """
        data = self.historical_data.get(symbol, [])
        latest = [candle for candle in data if candle.timestamp <= self.current_time][-1]
        return Ticker(
            symbol=symbol,
            last=latest.close,
            bid=latest.close * 0.9999,  # Simulated bid
            ask=latest.close * 1.0001,  # Simulated ask
            timestamp=latest.timestamp
        )
    
    async def fetch_balance(self) -> Balance:
        """Return the simulated account balance."""
        return self.account_state.get_balance()
    
    async def fetch_positions(self, symbol: str = None) -> List[Position]:
        """Return simulated open positions."""
        return self.account_state.get_positions(symbol)
    
    async def create_order(self, symbol: str, side: str, order_type: str,
                          amount: float, price: float, params: dict = None) -> Order:
        """
        Create an order and submit to the order simulator.
        
        The order is not immediately filled. The order simulator will
        determine if/when it fills based on historical price action.
        """
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            status='open',
            timestamp=self.current_time
        )
        self.order_simulator.submit_order(order)
        return order
    
    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """Cancel an order in the simulator."""
        return self.order_simulator.cancel_order(order_id)
    
    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        """Get order status from simulator."""
        return self.order_simulator.get_order(order_id)
    
    async def close(self):
        """Cleanup (no-op for backtest)."""
        pass
```

### 4.2 BacktestStreamManager

```python
class BacktestStreamManager(StreamManager):
    """
    Mock stream manager that feeds historical data instead of WebSocket.
    
    Instead of subscribing to real-time WebSocket streams, this class
    provides the same interface but serves data from the pre-loaded
    historical dataset.
    """
    
    def __init__(self, historical_data: Dict[str, List[OHLCV]], 
                 data_cache: BacktestDataCache):
        self.historical_data = historical_data
        self.data_cache = data_cache
        self.current_time: datetime = None
        self.subscribed_symbols: Set[str] = set()
    
    async def start(self):
        """Start the stream (no-op for backtest)."""
        pass
    
    async def stop(self):
        """Stop the stream (no-op for backtest)."""
        pass
    
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols (track for data serving)."""
        self.subscribed_symbols.update(symbols)
    
    async def update_time(self, current_time: datetime):
        """
        Update the current time and push new candle data to cache.
        
        This is called by the backtest engine at each time step to
        simulate new data arriving from the WebSocket.
        """
        self.current_time = current_time
        
        for symbol in self.subscribed_symbols:
            data = self.historical_data.get(symbol, [])
            # Find the candle for current_time
            candle = next(
                (c for c in data if c.timestamp == current_time),
                None
            )
            if candle:
                # Push to cache (simulating WebSocket update)
                await self.data_cache.update_ohlcv(symbol, candle)
    
    async def watch_ohlcv(self, symbol: str, timeframe: str):
        """
        Async generator that yields OHLCV data.
        
        In backtest mode, this yields data as time advances.
        """
        while True:
            data = self.historical_data.get(symbol, [])
            candle = next(
                (c for c in data if c.timestamp == self.current_time),
                None
            )
            if candle:
                yield candle
            await asyncio.sleep(0)  # Yield control
    
    async def watch_ticker(self, symbol: str):
        """
        Async generator that yields ticker data.
        
        In backtest mode, this yields ticker data as time advances.
        """
        while True:
            data = self.historical_data.get(symbol, [])
            candle = next(
                (c for c in data if c.timestamp == self.current_time),
                None
            )
            if candle:
                yield Ticker(
                    symbol=symbol,
                    last=candle.close,
                    bid=candle.close * 0.9999,
                    ask=candle.close * 1.0001,
                    timestamp=candle.timestamp
                )
            await asyncio.sleep(0)
```

### 4.3 BacktestDataCache

```python
class BacktestDataCache(DataCache):
    """
    Mock data cache that serves pre-loaded historical data.
    
    This class implements the same interface as DataCache but serves
    data from the historical dataset instead of caching real-time
    WebSocket data.
    """
    
    def __init__(self, historical_data: Dict[str, List[OHLCV]]):
        self.historical_data = historical_data
        self.current_time: datetime = None
        self._cache: Dict[str, List[OHLCV]] = {}
    
    async def update_ohlcv(self, symbol: str, candle: OHLCV):
        """
        Update cache with new candle data.
        
        Called by BacktestStreamManager to simulate WebSocket updates.
        """
        if symbol not in self._cache:
            self._cache[symbol] = []
        self._cache[symbol].append(candle)
    
    async def get_ohlcv(self, symbol: str, timeframe: str, 
                       limit: int = 200) -> List[OHLCV]:
        """
        Get OHLCV data up to current_time.
        
        Returns data from the historical dataset, filtered by time.
        """
        data = self.historical_data.get(symbol, [])
        # Filter data up to current_time
        filtered = [candle for candle in data if candle.timestamp <= self.current_time]
        return filtered[-limit:]
    
    async def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        data = self.historical_data.get(symbol, [])
        filtered = [candle for candle in data if candle.timestamp <= self.current_time]
        if filtered:
            return filtered[-1].close
        return 0.0
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data for a symbol."""
        price = await self.get_latest_price(symbol)
        return Ticker(
            symbol=symbol,
            last=price,
            bid=price * 0.9999,
            ask=price * 1.0001,
            timestamp=self.current_time
        )
    
    def set_current_time(self, current_time: datetime):
        """Update the current simulation time."""
        self.current_time = current_time
```

### 4.4 BacktestOrderSimulator

```python
class BacktestOrderSimulator:
    """
    Simulates order execution based on historical price action.
    
    This class is responsible for:
    - Tracking all submitted orders
    - Determining when limit orders fill based on price crossing
    - Calculating realistic fill prices with slippage
    - Applying trading fees
    - Handling partial fills
    - Processing order cancellations
    """
    
    def __init__(self, config: BacktestConfig, account_state: AccountState):
        self.config = config
        self.account_state = account_state
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.cancelled_orders: Dict[str, Order] = {}
        self.order_counter = 0
    
    def submit_order(self, order: Order):
        """Submit an order to the simulator."""
        self.pending_orders[order.id] = order
    
    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.status = 'cancelled'
            self.cancelled_orders[order_id] = order
            return order
        raise OrderNotFoundError(f"Order {order_id} not found")
    
    def get_order(self, order_id: str) -> Order:
        """Get order status."""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        if order_id in self.filled_orders:
            return self.filled_orders[order_id]
        if order_id in self.cancelled_orders:
            return self.cancelled_orders[order_id]
        raise OrderNotFoundError(f"Order {order_id} not found")
    
    def process_orders(self, candle: OHLCV):
        """
        Process all pending orders against the current candle.
        
        This is called at each time step to check if any pending orders
        should be filled based on the candle's price action.
        """
        orders_to_fill = []
        
        for order_id, order in list(self.pending_orders.items()):
            if self._should_fill_order(order, candle):
                orders_to_fill.append(order)
        
        for order in orders_to_fill:
            self._fill_order(order, candle)
    
    def _should_fill_order(self, order: Order, candle: OHLCV) -> bool:
        """
        Determine if a limit order should fill.
        
        A buy limit order fills if the price crosses below the order price.
        A sell limit order fills if the price crosses above the order price.
        """
        if order.side == 'buy':
            # Buy limit fills if price goes below order price
            return candle.low <= order.price
        else:  # sell
            # Sell limit fills if price goes above order price
            return candle.high >= order.price
    
    def _fill_order(self, order: Order, candle: OHLCV):
        """
        Fill an order with realistic price and fees.
        
        The fill price is calculated as:
        - Order price (for limit orders)
        - Plus/minus slippage
        - Fees are applied to the fill
        """
        # Calculate fill price with slippage
        if order.side == 'buy':
            # Buy: price + slippage
            fill_price = order.price * (1 + self.config.slippage_pct / 100)
            fee_rate = self.config.maker_fee / 100  # Limit orders are maker
        else:  # sell
            # Sell: price - slippage
            fill_price = order.price * (1 - self.config.slippage_pct / 100)
            fee_rate = self.config.maker_fee / 100
        
        # Calculate fee
        fee_amount = order.amount * fill_price * fee_rate
        
        # Update order
        order.status = 'filled'
        order.filled = order.amount
        order.average = fill_price
        order.fee = fee_amount
        order.filled_time = candle.timestamp
        
        # Move from pending to filled
        self.pending_orders.pop(order.id)
        self.filled_orders[order.id] = order
        
        # Update account state
        self.account_state.process_fill(order, fill_price, fee_amount)
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self.order_counter += 1
        return f"backtest_order_{self.order_counter}"
```

---

## 5. Historical Data Management

### 5.1 Data Fetching Strategy

```python
class HistoricalDataLoader:
    """
    Loads and manages historical OHLCV data for backtesting.
    
    Supports multiple data sources:
    - CSV files (one per symbol)
    - SQLite database
    - API fetch (for initial data collection)
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data: Dict[str, List[OHLCV]] = {}
    
    async def load_data(self) -> Dict[str, List[OHLCV]]:
        """
        Load historical data for all symbols.
        
        Returns a dictionary mapping symbol -> list of OHLCV candles.
        """
        if self.config.data_source == "csv":
            return await self._load_from_csv()
        elif self.config.data_source == "sqlite":
            return await self._load_from_sqlite()
        elif self.config.data_source == "api":
            return await self._fetch_from_api()
        else:
            raise ValueError(f"Unknown data source: {self.config.data_source}")
    
    async def _load_from_csv(self) -> Dict[str, List[OHLCV]]:
        """Load data from CSV files."""
        data = {}
        for symbol in self.config.symbols:
            filename = f"{self.config.data_path}{symbol.replace('/', '_')}_15m.csv"
            data[symbol] = await self._parse_csv(filename)
        return data
    
    async def _parse_csv(self, filename: str) -> List[OHLCV]:
        """Parse a CSV file into OHLCV objects."""
        candles = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append(OHLCV(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                ))
        return candles
    
    async def _load_from_sqlite(self) -> Dict[str, List[OHLCV]]:
        """Load data from SQLite database."""
        data = {}
        conn = sqlite3.connect(f"{self.config.data_path}historical_data.db")
        cursor = conn.cursor()
        
        for symbol in self.config.symbols:
            cursor.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp
            """, (symbol, self.config.timeframe))
            
            candles = []
            for row in cursor.fetchall():
                candles.append(OHLCV(
                    timestamp=datetime.fromisoformat(row[0]),
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5]
                ))
            data[symbol] = candles
        
        conn.close()
        return data
    
    async def _fetch_from_api(self) -> Dict[str, List[OHLCV]]:
        """
        Fetch historical data from exchange API.
        
        This is used to initially populate the historical data store.
        Not used during actual backtesting.
        """
        # Implementation would use CCXT to fetch historical data
        # and save to CSV/SQLite for future use
        pass
    
    def validate_data(self, data: Dict[str, List[OHLCV]]) -> bool:
        """
        Validate that historical data is complete and aligned.
        
        Checks:
        - Data exists for all symbols
        - No gaps in time series
        - Timeframe is correct (15m)
        - Date range covers the backtest period
        """
        for symbol, candles in data.items():
            if not candles:
                raise ValueError(f"No data for symbol {symbol}")
            
            # Check timeframe alignment
            for i in range(1, len(candles)):
                time_diff = candles[i].timestamp - candles[i-1].timestamp
                expected_diff = timedelta(minutes=15)
                if time_diff != expected_diff:
                    logger.warning(
                        f"Time gap detected in {symbol}: "
                        f"{candles[i-1].timestamp} -> {candles[i].timestamp}"
                    )
        
        return True
```

### 5.2 Data Format

#### CSV Format
```csv
timestamp,open,high,low,close,volume
2025-01-01T00:00:00,42000.50,42150.00,41900.00,42050.00,1250.5
2025-01-01T00:15:00,42050.00,42200.00,41950.00,42100.00,980.3
...
```

#### SQLite Schema
```sql
CREATE TABLE ohlcv_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX idx_ohlcv_symbol_time ON ohlcv_data(symbol, timestamp);
```

### 5.3 Data Preprocessing

```python
class DataPreprocessor:
    """
    Preprocesses historical data for backtesting.
    
    Tasks:
    - Ensure data is sorted by timestamp
    - Fill missing candles (optional)
    - Validate data quality
    - Calculate pre-computed indicators (optional)
    """
    
    @staticmethod
    def sort_data(data: Dict[str, List[OHLCV]]) -> Dict[str, List[OHLCV]]:
        """Sort data by timestamp for each symbol."""
        return {
            symbol: sorted(candles, key=lambda c: c.timestamp)
            for symbol, candles in data.items()
        }
    
    @staticmethod
    def filter_by_date_range(data: Dict[str, List[OHLCV]],
                            start_date: datetime,
                            end_date: datetime) -> Dict[str, List[OHLCV]]:
        """Filter data to the specified date range."""
        return {
            symbol: [
                candle for candle in candles
                if start_date <= candle.timestamp <= end_date
            ]
            for symbol, candles in data.items()
        }
    
    @staticmethod
    def validate_timeframe(candles: List[OHLCV], 
                          expected_interval: timedelta) -> bool:
        """Validate that candles are at the expected interval."""
        for i in range(1, len(candles)):
            time_diff = candles[i].timestamp - candles[i-1].timestamp
            if time_diff != expected_interval:
                return False
        return True
```

---

## 6. Order Simulation Logic

### 6.1 Limit Order Fill Simulation

The order simulator determines when limit orders fill based on historical price action:

```python
def _should_fill_order(self, order: Order, candle: OHLCV) -> bool:
    """
    Determine if a limit order should fill based on candle price action.
    
    Buy Limit Order:
    - Fills if candle.low <= order.price
    - This means the market traded down to or below the limit price
    
    Sell Limit Order:
    - Fills if candle.high >= order.price
    - This means the market traded up to or above the limit price
    """
    if order.side == 'buy':
        return candle.low <= order.price
    else:  # sell
        return candle.high >= order.price
```

### 6.2 Slippage Model

```python
class SlippageModel(Enum):
    """Slippage calculation models."""
    PERCENTAGE = "percentage"  # Fixed percentage slippage
    VOLATILITY_BASED = "volatility_based"  # Slippage based on ATR
    ORDER_BOOK_BASED = "order_book_based"  # Simulated order book depth

def calculate_slippage(self, order: Order, candle: OHLCV) -> float:
    """
    Calculate slippage for an order.
    
    Percentage Model:
    - Fixed percentage of order price
    - Default: 0.05%
    
    Volatility-Based Model:
    - Slippage proportional to ATR
    - Higher volatility = more slippage
    """
    if self.config.slippage_model == SlippageModel.PERCENTAGE:
        return self.config.slippage_pct / 100
    
    elif self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
        # Calculate ATR for recent candles
        atr = self._calculate_atr(candle)
        # Slippage as percentage of ATR
        slippage_pct = (atr / candle.close) * 0.5  # 50% of ATR as slippage
        return slippage_pct
    
    return self.config.slippage_pct / 100
```

### 6.3 Trading Fees

```python
def calculate_fees(self, order: Order, fill_price: float) -> float:
    """
    Calculate trading fees for an order.
    
    Maker Fee: 0.02% (limit orders that add liquidity)
    Taker Fee: 0.06% (market orders that take liquidity)
    
    In backtesting, we assume limit orders are maker orders.
    """
    if order.type == 'limit':
        fee_rate = self.config.maker_fee / 100
    else:  # market
        fee_rate = self.config.taker_fee / 100
    
    return order.amount * fill_price * fee_rate
```

### 6.4 Partial Fills

```python
def _fill_order(self, order: Order, candle: OHLCV):
    """
    Fill an order, handling partial fills.
    
    For simplicity in the initial implementation, we assume full fills.
    Partial fills can be added by:
    1. Tracking filled amount separately from order amount
    2. Updating order status to 'partial' when partially filled
    3. Keeping order in pending_orders until fully filled
    """
    # Calculate fillable amount based on volume
    max_fillable = candle.volume * 0.1  # Assume we can fill 10% of volume
    fillable_amount = min(order.amount, max_fillable)
    
    if fillable_amount < order.amount:
        # Partial fill
        order.filled = fillable_amount
        order.status = 'partial'
        # Keep in pending_orders
    else:
        # Full fill
        order.filled = order.amount
        order.status = 'filled'
        # Move to filled_orders
```

### 6.5 Order Cancellation

```python
def cancel_order(self, order_id: str) -> Order:
    """
    Cancel a pending order.
    
    In backtesting, orders can be cancelled at any time before they fill.
    The cancellation is processed immediately.
    """
    if order_id in self.pending_orders:
        order = self.pending_orders.pop(order_id)
        order.status = 'cancelled'
        self.cancelled_orders[order_id] = order
        return order
    raise OrderNotFoundError(f"Order {order_id} not found")
```

---

## 7. State Tracking

### 7.1 Account State

```python
@dataclass
class AccountState:
    """
    Tracks the simulated account state during backtesting.
    
    This includes:
    - Available balance
    - Total balance (including unrealized P&L)
    - Open positions
    - Realized P&L
    - Fees paid
    """
    
    initial_balance: float
    currency: str
    
    # Current state
    available_balance: float
    total_balance: float
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def get_balance(self) -> Balance:
        """Get current balance as Balance object."""
        return Balance(
            total=self.total_balance,
            free=self.available_balance,
            used=self.total_balance - self.available_balance,
            currency=self.currency
        )
    
    def get_positions(self, symbol: str = None) -> List[Position]:
        """Get open positions, optionally filtered by symbol."""
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions
    
    def process_fill(self, order: Order, fill_price: float, fee: float):
        """
        Process an order fill and update account state.
        
        This handles:
        - Opening new positions
        - Closing existing positions
        - Updating balance
        - Calculating P&L
        """
        if order.side == 'buy':
            # Opening long position
            position = Position(
                id=self._generate_position_id(),
                symbol=order.symbol,
                side='long',
                entry_price=fill_price,
                size=order.amount,
                stop_loss=None,
                take_profit=None,
                open_time=order.timestamp,
                status='open'
            )
            self.positions[position.id] = position
            
            # Deduct from available balance
            cost = order.amount * fill_price + fee
            self.available_balance -= cost
            self.total_fees += fee
            
        else:  # sell
            # Closing long position (simplification)
            # Find matching position
            position = self._find_position_to_close(order.symbol)
            if position:
                # Calculate P&L
                pnl = (fill_price - position.entry_price) * position.size - fee
                
                # Update position
                position.exit_price = fill_price
                position.close_time = order.timestamp
                position.status = 'closed'
                position.realized_pnl = pnl
                
                # Update account
                self.realized_pnl += pnl
                self.available_balance += (position.size * fill_price - fee)
                self.total_fees += fee
                self.total_trades += 1
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Remove from open positions
                del self.positions[position.id]
        
        # Update total balance
        self._update_total_balance()
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]):
        """
        Update unrealized P&L for all open positions.
        
        Called at each time step to track unrealized gains/losses.
        """
        total_unrealized = 0.0
        
        for position in self.positions.values():
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                if position.side == 'long':
                    unrealized = (current_price - position.entry_price) * position.size
                else:  # short
                    unrealized = (position.entry_price - current_price) * position.size
                position.unrealized_pnl = unrealized
                total_unrealized += unrealized
        
        self.unrealized_pnl = total_unrealized
        self._update_total_balance()
    
    def _update_total_balance(self):
        """Update total balance including unrealized P&L."""
        self.total_balance = self.available_balance + self.unrealized_pnl
    
    def _generate_position_id(self) -> str:
        """Generate a unique position ID."""
        return f"pos_{len(self.positions)}_{int(time.time())}"
    
    def _find_position_to_close(self, symbol: str) -> Optional[Position]:
        """Find an open position to close."""
        for position in self.positions.values():
            if position.symbol == symbol and position.status == 'open':
                return position
        return None
```

### 7.2 Trade History

```python
@dataclass
class TradeRecord:
    """Record of a completed trade."""
    
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees: float
    duration: timedelta
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal', 'manual'
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'pnl': self.pnl,
            'fees': self.fees,
            'duration_seconds': self.duration.total_seconds(),
            'exit_reason': self.exit_reason
        }

class TradeHistory:
    """Manages the trade history during backtesting."""
    
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.trade_counter = 0
    
    def add_trade(self, position: Position, exit_price: float, 
                 exit_time: datetime, exit_reason: str):
        """Add a completed trade to history."""
        trade = TradeRecord(
            trade_id=f"trade_{self.trade_counter}",
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            entry_time=position.open_time,
            exit_time=exit_time,
            pnl=position.realized_pnl,
            fees=0.0,  # Would be tracked separately
            duration=exit_time - position.open_time,
            exit_reason=exit_reason
        )
        self.trades.append(trade)
        self.trade_counter += 1
    
    def get_trades(self, symbol: str = None) -> List[TradeRecord]:
        """Get trades, optionally filtered by symbol."""
        if symbol:
            return [t for t in self.trades if t.symbol == symbol]
        return self.trades
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade history to pandas DataFrame."""
        return pd.DataFrame([t.to_dict() for t in self.trades])
```

### 7.3 Equity Curve

```python
@dataclass
class EquityPoint:
    """A point on the equity curve."""
    
    timestamp: datetime
    equity: float
    available_balance: float
    unrealized_pnl: float
    open_positions: int

class EquityCurve:
    """Tracks the equity curve during backtesting."""
    
    def __init__(self):
        self.points: List[EquityPoint] = []
    
    def add_point(self, timestamp: datetime, account_state: AccountState):
        """Add a point to the equity curve."""
        point = EquityPoint(
            timestamp=timestamp,
            equity=account_state.total_balance,
            available_balance=account_state.available_balance,
            unrealized_pnl=account_state.unrealized_pnl,
            open_positions=len(account_state.positions)
        )
        self.points.append(point)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to pandas DataFrame."""
        return pd.DataFrame([
            {
                'timestamp': p.timestamp,
                'equity': p.equity,
                'available_balance': p.available_balance,
                'unrealized_pnl': p.unrealized_pnl,
                'open_positions': p.open_positions
            }
            for p in self.points
        ])
```

---

## 8. Performance Metrics

### 8.1 Metrics Calculation

```python
@dataclass
class PerformanceMetrics:
    """Performance metrics for the backtest."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade: float
    
    # Exposure metrics
    avg_exposure_time: timedelta
    max_open_positions: int
    
    # Additional metrics
    calmar_ratio: float
    recovery_factor: float

class PerformanceCalculator:
    """Calculates performance metrics from backtest results."""
    
    def __init__(self, account_state: AccountState, 
                 trade_history: TradeHistory,
                 equity_curve: EquityCurve):
        self.account_state = account_state
        self.trade_history = trade_history
        self.equity_curve = equity_curve
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        equity_df = self.equity_curve.to_dataframe()
        trades_df = self.trade_history.to_dataframe()
        
        return PerformanceMetrics(
            total_return=self._calculate_total_return(),
            annualized_return=self._calculate_annualized_return(equity_df),
            sharpe_ratio=self._calculate_sharpe_ratio(equity_df),
            sortino_ratio=self._calculate_sortino_ratio(equity_df),
            max_drawdown=self._calculate_max_drawdown(equity_df),
            max_drawdown_duration=self._calculate_max_drawdown_duration(equity_df),
            total_trades=len(trades_df),
            winning_trades=len(trades_df[trades_df['pnl'] > 0]),
            losing_trades=len(trades_df[trades_df['pnl'] < 0]),
            win_rate=self._calculate_win_rate(trades_df),
            total_pnl=self.account_state.realized_pnl,
            avg_win=self._calculate_avg_win(trades_df),
            avg_loss=self._calculate_avg_loss(trades_df),
            profit_factor=self._calculate_profit_factor(trades_df),
            avg_trade=self._calculate_avg_trade(trades_df),
            avg_exposure_time=self._calculate_avg_exposure_time(trades_df),
            max_open_positions=max(equity_df['open_positions']),
            calmar_ratio=self._calculate_calmar_ratio(),
            recovery_factor=self._calculate_recovery_factor()
        )
    
    def _calculate_total_return(self) -> float:
        """Calculate total return percentage."""
        initial = self.account_state.initial_balance
        final = self.account_state.total_balance
        return ((final - initial) / initial) * 100
    
    def _calculate_annualized_return(self, equity_df: pd.DataFrame) -> float:
        """Calculate annualized return."""
        total_return = self._calculate_total_return() / 100
        days = (equity_df['timestamp'].max() - equity_df['timestamp'].min()).days
        years = days / 365.25
        return ((1 + total_return) ** (1 / years) - 1) * 100
    
    def _calculate_sharpe_ratio(self, equity_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio."""
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns = equity_df['returns'].dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Assume risk-free rate of 0%
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 4)  # Annualized for 15m
        return sharpe
    
    def _calculate_sortino_ratio(self, equity_df: pd.DataFrame) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns = equity_df['returns'].dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return float('inf')
        
        sortino = (returns.mean() / downside_deviation) * np.sqrt(252 * 24 * 4)
        return sortino
    
    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage."""
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        return equity_df['drawdown'].min() * 100
    
    def _calculate_max_drawdown_duration(self, equity_df: pd.DataFrame) -> timedelta:
        """Calculate maximum drawdown duration."""
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['in_drawdown'] = equity_df['equity'] < equity_df['cummax']
        
        max_duration = timedelta(0)
        current_duration = timedelta(0)
        
        for i in range(1, len(equity_df)):
            if equity_df['in_drawdown'].iloc[i]:
                current_duration += (equity_df['timestamp'].iloc[i] - equity_df['timestamp'].iloc[i-1])
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = timedelta(0)
        
        return max_duration
    
    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate percentage."""
        if len(trades_df) == 0:
            return 0.0
        winning = len(trades_df[trades_df['pnl'] > 0])
        return (winning / len(trades_df)) * 100
    
    def _calculate_avg_win(self, trades_df: pd.DataFrame) -> float:
        """Calculate average winning trade."""
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        return wins.mean() if len(wins) > 0 else 0.0
    
    def _calculate_avg_loss(self, trades_df: pd.DataFrame) -> float:
        """Calculate average losing trade."""
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        return losses.mean() if len(losses) > 0 else 0.0
    
    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_avg_trade(self, trades_df: pd.DataFrame) -> float:
        """Calculate average trade P&L."""
        if len(trades_df) == 0:
            return 0.0
        return trades_df['pnl'].mean()
    
    def _calculate_avg_exposure_time(self, trades_df: pd.DataFrame) -> timedelta:
        """Calculate average position exposure time."""
        if len(trades_df) == 0:
            return timedelta(0)
        return trades_df['duration_seconds'].mean() * timedelta(seconds=1)
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annualized_return = self._calculate_annualized_return(pd.DataFrame())
        max_drawdown = abs(self._calculate_max_drawdown(pd.DataFrame()))
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (total P&L / max drawdown)."""
        total_pnl = self.account_state.realized_pnl
        max_drawdown = abs(self._calculate_max_drawdown(pd.DataFrame()))
        
        if max_drawdown == 0:
            return float('inf') if total_pnl > 0 else 0.0
        
        return total_pnl / max_drawdown
```

---

## 9. Configuration

### 9.1 Backtest Configuration File

```yaml
# config/backtest_config.yaml

# Backtest Settings
backtest:
  # Date range for backtest
  start_date: "2025-01-01T00:00:00"
  end_date: "2025-12-31T23:59:59"
  
  # Trading symbols
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
  
  # Timeframe
  timeframe: "15m"
  
  # Initial account state
  initial_balance: 10000.0
  initial_currency: "USDT"

# Order Simulation
order_simulation:
  # Slippage model: percentage, volatility_based, order_book_based
  slippage_model: "percentage"
  slippage_pct: 0.05  # 0.05%
  
  # Trading fees
  maker_fee: 0.02  # 0.02%
  taker_fee: 0.06  # 0.06%
  
  # Partial fills
  enable_partial_fills: false
  max_fill_pct_volume: 10.0  # Max 10% of candle volume

# Data Source
data:
  # Source type: csv, sqlite, api
  source: "csv"
  path: "data/historical/2025/"
  
  # Data validation
  validate_data: true
  fill_gaps: false

# Output Settings
output:
  directory: "backtest_results/2025/"
  
  # What to save
  save_trade_log: true
  save_equity_curve: true
  save_order_log: true
  generate_report: true
  
  # Report format: html, json, csv
  report_format: "html"

# Trading Bot Configuration (same as live)
# Import from main config or specify here
trading_bot_config:
  check_interval: 5.0
  enable_hedging: true
  enable_signal_logging: true
```

### 9.2 Configuration Manager

```python
class BacktestConfigManager:
    """Manages backtest configuration."""
    
    def __init__(self, config_path: str = "config/backtest_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> BacktestConfig:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return BacktestConfig(
            start_date=datetime.fromisoformat(config_data['backtest']['start_date']),
            end_date=datetime.fromisoformat(config_data['backtest']['end_date']),
            symbols=config_data['backtest']['symbols'],
            timeframe=config_data['backtest']['timeframe'],
            initial_balance=config_data['backtest']['initial_balance'],
            initial_currency=config_data['backtest']['initial_currency'],
            slippage_model=SlippageModel(config_data['order_simulation']['slippage_model']),
            slippage_pct=config_data['order_simulation']['slippage_pct'],
            maker_fee=config_data['order_simulation']['maker_fee'],
            taker_fee=config_data['order_simulation']['taker_fee'],
            data_source=config_data['data']['source'],
            data_path=config_data['data']['path'],
            output_dir=config_data['output']['directory'],
            save_trade_log=config_data['output']['save_trade_log'],
            save_equity_curve=config_data['output']['save_equity_curve'],
            generate_report=config_data['output']['generate_report']
        )
    
    def get_config(self) -> BacktestConfig:
        """Get the backtest configuration."""
        return self.config
```

---

## 10. Output & Reporting

### 10.1 Trade Log Format

```python
class TradeLogger:
    """Logs trades during backtesting."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.trades: List[dict] = []
    
    def log_trade(self, trade: TradeRecord):
        """Log a trade."""
        self.trades.append(trade.to_dict())
    
    def save_to_csv(self, filename: str = "trade_log.csv"):
        """Save trade log to CSV."""
        df = pd.DataFrame(self.trades)
        df.to_csv(f"{self.output_dir}/{filename}", index=False)
    
    def save_to_json(self, filename: str = "trade_log.json"):
        """Save trade log to JSON."""
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            json.dump(self.trades, f, indent=2)
```

### 10.2 Equity Curve Generation

```python
class EquityCurveGenerator:
    """Generates equity curve visualizations."""
    
    def __init__(self, equity_curve: EquityCurve, output_dir: str):
        self.equity_curve = equity_curve
        self.output_dir = output_dir
    
    def generate_plot(self, filename: str = "equity_curve.png"):
        """Generate equity curve plot."""
        df = self.equity_curve.to_dataframe()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['equity'], label='Equity', linewidth=2)
        plt.fill_between(df['timestamp'], df['equity'], alpha=0.3)
        
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300)
        plt.close()
    
    def generate_drawdown_plot(self, filename: str = "drawdown.png"):
        """Generate drawdown plot."""
        df = self.equity_curve.to_dataframe()
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['timestamp'], df['drawdown'], 0, alpha=0.3, color='red')
        plt.plot(df['timestamp'], df['drawdown'], color='red', linewidth=1)
        
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300)
        plt.close()
```

### 10.3 Performance Report Structure

```python
class ReportGenerator:
    """Generates comprehensive backtest reports."""
    
    def __init__(self, metrics: PerformanceMetrics, 
                 trade_history: TradeHistory,
                 equity_curve: EquityCurve,
                 output_dir: str):
        self.metrics = metrics
        self.trade_history = trade_history
        self.equity_curve = equity_curve
        self.output_dir = output_dir
    
    def generate_html_report(self, filename: str = "backtest_report.html"):
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - 2025</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          background: #f9f9f9; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Trading Bot Backtest Report - 2025</h1>
            
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value {'positive' if self.metrics.total_return > 0 else 'negative'}">
                    {self.metrics.total_return:.2f}%
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metrics.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metrics.win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{self.metrics.max_drawdown:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            
            <h2>Return Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td class="{'positive' if self.metrics.total_return > 0 else 'negative'}">
                        {self.metrics.total_return:.2f}%
                    </td>
                </tr>
                <tr>
                    <td>Annualized Return</td>
                    <td class="{'positive' if self.metrics.annualized_return > 0 else 'negative'}">
                        {self.metrics.annualized_return:.2f}%
                    </td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{self.metrics.sharpe_ratio:.2f}</td>
                </tr>
                <tr>
                    <td>Sortino Ratio</td>
                    <td>{self.metrics.sortino_ratio:.2f}</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{self.metrics.calmar_ratio:.2f}</td>
                </tr>
            </table>
            
            <h2>Risk Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td class="negative">{self.metrics.max_drawdown:.2f}%</td>
                </tr>
                <tr>
                    <td>Max Drawdown Duration</td>
                    <td>{self.metrics.max_drawdown_duration.days} days</td>
                </tr>
            </table>
            
            <h2>Trade Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{self.metrics.total_trades}</td>
                </tr>
                <tr>
                    <td>Winning Trades</td>
                    <td class="positive">{self.metrics.winning_trades}</td>
                </tr>
                <tr>
                    <td>Losing Trades</td>
                    <td class="negative">{self.metrics.losing_trades}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{self.metrics.win_rate:.1f}%</td>
                </tr>
                <tr>
                    <td>Average Win</td>
                    <td class="positive">${self.metrics.avg_win:.2f}</td>
                </tr>
                <tr>
                    <td>Average Loss</td>
                    <td class="negative">${self.metrics.avg_loss:.2f}</td>
                </tr>
                <tr>
                    <td>Average Trade</td>
                    <td class="{'positive' if self.metrics.avg_trade > 0 else 'negative'}">
                        ${self.metrics.avg_trade:.2f}
                    </td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{self.metrics.profit_factor:.2f}</td>
                </tr>
                <tr>
                    <td>Average Exposure Time</td>
                    <td>{self.metrics.avg_exposure_time.total_seconds() / 3600:.1f} hours</td>
                </tr>
            </table>
            
            <h2>P&L Breakdown</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total P&L</td>
                    <td class="{'positive' if self.metrics.total_pnl > 0 else 'negative'}">
                        ${self.metrics.total_pnl:.2f}
                    </td>
                </tr>
                <tr>
                    <td>Gross Profit</td>
                    <td class="positive">${self.metrics.avg_win * self.metrics.winning_trades:.2f}</td>
                </tr>
                <tr>
                    <td>Gross Loss</td>
                    <td class="negative">${abs(self.metrics.avg_loss * self.metrics.losing_trades):.2f}</td>
                </tr>
            </table>
            
            <h2>Charts</h2>
            <img src="equity_curve.png" alt="Equity Curve" style="max-width: 100%;">
            <img src="drawdown.png" alt="Drawdown" style="max-width: 100%;">
            
        </body>
        </html>
        """
        
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            f.write(html)
    
    def generate_json_report(self, filename: str = "backtest_report.json"):
        """Generate JSON report."""
        report = {
            'summary': {
                'total_return': self.metrics.total_return,
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown
            },
            'return_metrics': {
                'total_return': self.metrics.total_return,
                'annualized_return': self.metrics.annualized_return,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'sortino_ratio': self.metrics.sortino_ratio,
                'calmar_ratio': self.metrics.calmar_ratio
            },
            'risk_metrics': {
                'max_drawdown': self.metrics.max_drawdown,
                'max_drawdown_duration_days': self.metrics.max_drawdown_duration.days
            },
            'trade_metrics': {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'win_rate': self.metrics.win_rate,
                'avg_win': self.metrics.avg_win,
                'avg_loss': self.metrics.avg_loss,
                'avg_trade': self.metrics.avg_trade,
                'profit_factor': self.metrics.profit_factor,
                'avg_exposure_time_hours': self.metrics.avg_exposure_time.total_seconds() / 3600
            },
            'pnl_breakdown': {
                'total_pnl': self.metrics.total_pnl,
                'gross_profit': self.metrics.avg_win * self.metrics.winning_trades,
                'gross_loss': abs(self.metrics.avg_loss * self.metrics.losing_trades)
            }
        }
        
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            json.dump(report, f, indent=2)
```

---

## 11. Implementation Roadmap

### Phase 1: Core Infrastructure
1. Create backtest configuration structure
2. Implement HistoricalDataLoader
3. Implement TimeController
4. Implement AccountState

### Phase 2: Mock Components
1. Implement BacktestExchangeClient
2. Implement BacktestStreamManager
3. Implement BacktestDataCache
4. Implement BacktestOrderSimulator

### Phase 3: Backtest Engine
1. Implement BacktestEngine
2. Implement main backtest loop
3. Integrate with existing trading logic
4. Test with sample data

### Phase 4: State Tracking & Metrics
1. Implement TradeHistory
2. Implement EquityCurve
3. Implement PerformanceCalculator
4. Implement PerformanceMetrics

### Phase 5: Reporting
1. Implement TradeLogger
2. Implement EquityCurveGenerator
3. Implement ReportGenerator
4. Create HTML/JSON report templates

### Phase 6: Data Preparation
1. Fetch 2025 historical data for all symbols
2. Validate and preprocess data
3. Store in CSV/SQLite format
4. Create data validation scripts

### Phase 7: Testing & Validation
1. Unit tests for all components
2. Integration tests
3. Compare backtest results with live trading (if available)
4. Performance optimization

---

## 12. Key Design Decisions

### 12.1 Why Mock Components Instead of Modifying Trading Logic?

**Decision**: Use dependency injection to mock data sources and order execution without modifying any trading logic.

**Rationale**:
- Ensures backtest behavior exactly matches live behavior
- Reduces risk of introducing bugs during backtesting
- Makes it easy to switch between live and backtest modes
- Trading logic remains the single source of truth

### 12.2 Why Iterate Through Candles Instead of Ticks?

**Decision**: Iterate through 15-minute candles, not tick-by-tick.

**Rationale**:
- Trading strategy operates on 15-minute timeframe
- Significantly faster execution
- Sufficient accuracy for strategy validation
- Historical tick data is often unavailable or expensive

### 12.3 Why Use CSV/SQLite for Data Storage?

**Decision**: Support both CSV and SQLite for historical data storage.

**Rationale**:
- CSV: Simple, human-readable, easy to share
- SQLite: Faster queries, better for large datasets
- Flexibility to choose based on use case
- Easy to convert between formats

### 12.4 Why Simulate Limit Order Fills Based on Candle High/Low?

**Decision**: Use candle high/low to determine limit order fills.

**Rationale**:
- Realistic approximation of order fills
- No tick-level data required
- Conservative approach (fills only if price crosses)
- Matches how most backtesting frameworks work

### 12.5 Why Track Both Realized and Unrealized P&L?

**Decision**: Track both realized and unrealized P&L separately.

**Rationale**:
- Realized P&L: Actual profit/loss from closed trades
- Unrealized P&L: Current profit/loss on open positions
- Important for understanding risk exposure
- Matches how exchanges report P&L

---

## 13. Future Enhancements

### 13.1 Advanced Order Simulation
- Implement order book depth simulation
- Add market impact modeling
- Support iceberg orders
- Implement time-weighted average price (TWAP) fills

### 13.2 Advanced Risk Metrics
- Value at Risk (VaR)
- Expected Shortfall
- Beta calculation
- Correlation analysis

### 13.3 Optimization Features
- Parameter optimization (grid search, genetic algorithms)
- Walk-forward analysis
- Monte Carlo simulation
- Bootstrap analysis

### 13.4 Visualization Enhancements
- Interactive charts with Plotly
- Trade-by-trade visualization
- Heatmaps for performance by time/day
- Position size over time chart

### 13.5 Multi-Asset Backtesting
- Portfolio-level backtesting
- Correlation-aware position sizing
- Multi-strategy backtesting
- Asset allocation optimization

---

## 14. Conclusion

This design document provides a comprehensive blueprint for building a backtesting system that exactly mirrors the live trading bot's behavior. The key strengths of this design are:

1. **Zero Logic Modification**: All trading logic remains unchanged
2. **Component Injection**: Mock components implement the same interfaces
3. **Realistic Simulation**: Order fills, slippage, and fees modeled accurately
4. **Complete State Tracking**: All positions, orders, and P&L tracked
5. **Comprehensive Reporting**: Detailed performance metrics and visualizations

The backtester will provide valuable insights into the strategy's performance on 2025 data, helping to validate the approach before deploying with real capital.

---

## Appendix A: File Structure

```
src/
├── backtest/
│   ├── __init__.py
│   ├── backtest_engine.py          # Main backtest engine
│   ├── backtest_config.py          # Configuration classes
│   ├── time_controller.py          # Time control mechanism
│   ├── account_state.py            # Account state tracking
│   ├── trade_history.py            # Trade history management
│   ├── equity_curve.py             # Equity curve tracking
│   ├── performance_calculator.py  # Performance metrics
│   ├── report_generator.py         # Report generation
│   └── mock/
│       ├── __init__.py
│       ├── backtest_exchange_client.py
│       ├── backtest_stream_manager.py
│       ├── backtest_data_cache.py
│       └── backtest_order_simulator.py
├── data/
│   └── historical/
│       └── 2025/
│           ├── BTC_USDT_15m.csv
│           ├── ETH_USDT_15m.csv
│           └── SOL_USDT_15m.csv
├── config/
│   └── backtest_config.yaml
└── backtest_results/
    └── 2025/
        ├── trade_log.csv
        ├── trade_log.json
        ├── equity_curve.csv
        ├── equity_curve.png
        ├── drawdown.png
        ├── backtest_report.html
        └── backtest_report.json
```

## Appendix B: Usage Example

```python
import asyncio
from src.backtest import BacktestEngine, BacktestConfigManager

async def main():
    # Load configuration
    config_manager = BacktestConfigManager("config/backtest_config.yaml")
    config = config_manager.get_config()
    
    # Create backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest
    results = await engine.run_backtest()
    
    # Print summary
    print(f"Total Return: {results.metrics.total_return:.2f}%")
    print(f"Total Trades: {results.metrics.total_trades}")
    print(f"Win Rate: {results.metrics.win_rate:.1f}%")
    print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.metrics.max_drawdown:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
```

---

**Document End**
