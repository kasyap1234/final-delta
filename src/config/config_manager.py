"""
Configuration Manager for Delta Exchange India Trading Bot.

This module provides centralized configuration management with support for:
- YAML and JSON configuration files
- Environment variable overrides
- Pydantic-based validation
- Default values for optional parameters
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum


class OrderType(str, Enum):
    """Supported order types."""
    LIMIT = "limit"
    MARKET = "market"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExchangeSettings(BaseModel):
    """Exchange connection settings."""
    exchange_id: str = Field(default="delta", description="Exchange identifier")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    api_secret: Optional[str] = Field(default=None, description="API secret for authentication")
    sandbox: bool = Field(default=True, description="Use sandbox environment for testing")
    testnet: bool = Field(default=True, description="Use testnet for testing")

    @validator('exchange_id')
    def validate_exchange_id(cls, v):
        if v not in ["delta", "delta_exchange"]:
            raise ValueError("exchange_id must be 'delta' or 'delta_exchange'")
        return v


class TradingSettings(BaseModel):
    """Trading strategy settings."""
    timeframe: str = Field(default="15m", description="Candlestick timeframe")
    symbols: List[str] = Field(
        default=["BTC/USD", "ETH/USD", "SOL/USD"],
        description="List of trading pairs"
    )
    max_positions: int = Field(default=5, ge=1, description="Maximum concurrent positions")

    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        if v not in valid_timeframes:
            raise ValueError(f"timeframe must be one of {valid_timeframes}")
        return v

    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("symbols list cannot be empty")
        return v


class StrategySettings(BaseModel):
    """Technical indicator and strategy parameters."""
    ema_fast: int = Field(default=9, ge=1, description="Fast EMA period")
    ema_medium: int = Field(default=21, ge=1, description="Medium EMA period")
    ema_slow: int = Field(default=50, ge=1, description="Slow EMA period")
    ema_trend: int = Field(default=200, ge=1, description="Trend EMA period")
    rsi_period: int = Field(default=14, ge=1, description="RSI calculation period")
    rsi_long_threshold: float = Field(
        default=70.0, ge=0, le=100,
        description="RSI threshold for long signals (oversold level)"
    )
    rsi_short_threshold: float = Field(
        default=30.0, ge=0, le=100,
        description="RSI threshold for short signals (overbought level)"
    )
    atr_period: int = Field(default=14, ge=1, description="ATR calculation period")
    atr_multiplier: float = Field(
        default=2.0, ge=0.5, le=5.0,
        description="ATR multiplier for stop-loss calculation"
    )
    pivot_lookback: int = Field(
        default=10, ge=1,
        description="Lookback period for pivot/resistance detection"
    )
    
    # Regime filter parameters (matching backtest)
    min_adx_for_entry: float = Field(
        default=15.0, ge=0,
        description="Minimum ADX value for entry signals"
    )
    min_ema_spread_for_entry: float = Field(
        default=0.005, ge=0,
        description="Minimum EMA spread (as decimal) for entry signals (0.005 = 0.5%)"
    )
    
    # Signal strength position sizing parameters (matching backtest)
    enable_signal_strength_sizing: bool = Field(
        default=True,
        description="Enable position sizing based on signal strength"
    )
    strong_signal_threshold: float = Field(
        default=0.8, ge=0, le=1.0,
        description="Signal strength threshold for full position size"
    )
    weak_signal_threshold: float = Field(
        default=0.3, ge=0, le=1.0,
        description="Signal strength threshold for minimum position size"
    )
    
    # Dynamic parameter adjustment (matching backtest)
    enable_dynamic_atr: bool = Field(
        default=True,
        description="Enable dynamic ATR multiplier based on volatility"
    )
    enable_adaptive_rr: bool = Field(
        default=True,
        description="Enable adaptive risk:reward ratio based on market conditions"
    )
    high_volatility_atr_threshold: float = Field(
        default=0.03, ge=0,
        description="ATR% threshold for high volatility (3% = 0.03)"
    )
    ranging_ema_threshold: float = Field(
        default=0.01, ge=0,
        description="EMA range threshold for ranging market detection (1% = 0.01)"
    )

    @validator('rsi_long_threshold', 'rsi_short_threshold')
    def validate_rsi_thresholds(cls, v, values):
        return v

    @validator('ema_fast', 'ema_medium', 'ema_slow', 'ema_trend')
    def validate_ema_order(cls, v, values, **kwargs):
        # Get field name from validator info
        field_info = kwargs.get('field')
        if field_info is None:
            # For Pydantic v2 compatibility
            field_name = None
            for name, val in values.items():
                if val == v:
                    field_name = name
                    break
        else:
            field_name = field_info.name
        
        # Build current EMA values
        ema_values = {
            'ema_fast': values.get('ema_fast') or 9,
            'ema_medium': values.get('ema_medium') or 21,
            'ema_slow': values.get('ema_slow') or 50,
            'ema_trend': values.get('ema_trend') or 200
        }
        if field_name:
            ema_values[field_name] = v
        
        # Check ordering: fast < medium < slow < trend
        if (ema_values['ema_fast'] >= ema_values['ema_medium'] or
            ema_values['ema_medium'] >= ema_values['ema_slow'] or
            ema_values['ema_slow'] >= ema_values['ema_trend']):
            raise ValueError(
                "EMA periods must be ordered: fast < medium < slow < trend"
            )
        return v


class RiskManagementSettings(BaseModel):
    """Risk management and position sizing settings."""
    account_balance: float = Field(
        default=10000.0, gt=0,
        description="Total trading capital"
    )
    risk_per_trade_percent: float = Field(
        default=1.0, gt=0, le=100,
        description="Risk per trade as percentage of account"
    )
    max_risk_per_trade_percent: float = Field(
        default=2.0, gt=0, le=100,
        description="Maximum risk per trade percentage"
    )
    take_profit_r_ratio: float = Field(
        default=2.0, gt=0,
        description="Risk:Reward ratio for take profit"
    )

    @validator('max_risk_per_trade_percent')
    def validate_max_risk(cls, v, values):
        risk_per_trade = values.get('risk_per_trade_percent', 1.0)
        if v < risk_per_trade:
            raise ValueError(
                "max_risk_per_trade_percent must be >= risk_per_trade_percent"
            )
        return v


class HedgeSettings(BaseModel):
    """Hedge configuration settings."""
    hedge_trigger_percent: float = Field(
        default=50.0, ge=0, le=100,
        description="Trigger hedge at this percentage of SL distance"
    )
    hedge_size_ratio: float = Field(
        default=0.5, gt=0, le=1.0,
        description="Hedge size as ratio of original position"
    )
    hedge_chunks: int = Field(
        default=3, ge=1,
        description="Number of chunks to split hedge into"
    )
    correlation_lookback_days: int = Field(
        default=30, ge=1,
        description="Days of historical data for correlation calculation"
    )
    correlation_symbols: List[str] = Field(
        default=["BTC/USD", "ETH/USD", "SOL/USD"],
        description="Symbols to consider for correlation-based hedging"
    )
    max_hedges_per_position: int = Field(
        default=5, ge=0,
        description="Maximum number of hedge orders per position"
    )


class OrderSettings(BaseModel):
    """Order execution settings."""
    order_type: OrderType = Field(
        default=OrderType.LIMIT,
        description="Default order type"
    )
    post_only: bool = Field(
        default=True,
        description="Use post-only orders (maker fees)"
    )
    price_offset_percent: float = Field(
        default=0.01, ge=0, le=1.0,
        description="Price offset from market price in percent"
    )
    retry_attempts: int = Field(
        default=3, ge=0,
        description="Number of retry attempts for failed orders"
    )
    retry_delay_seconds: int = Field(
        default=1, ge=0,
        description="Delay between retry attempts in seconds"
    )


class DatabaseSettings(BaseModel):
    """Database and logging settings."""
    db_path: str = Field(
        default="data/trading_bot.db",
        description="Path to SQLite database file"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    log_file: str = Field(
        default="logs/trading_bot.log",
        description="Path to log file"
    )


class TradingBotConfig(BaseModel):
    """Complete trading bot configuration."""
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk_management: RiskManagementSettings = Field(default_factory=RiskManagementSettings)
    hedge: HedgeSettings = Field(default_factory=HedgeSettings)
    order: OrderSettings = Field(default_factory=OrderSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)


class ConfigManager:
    """
    Configuration manager for the trading bot.
    
    Handles loading configuration from YAML/JSON files with support for:
    - Environment variable overrides
    - Validation using Pydantic models
    - Default values for optional parameters
    - Saving configuration back to file
    
    Environment Variables:
        DELTA_API_KEY: Override API key
        DELTA_API_SECRET: Override API secret
        DELTA_SANDBOX: Override sandbox mode (true/false)
        DELTA_TESTNET: Override testnet mode (true/false)
        DELTA_ACCOUNT_BALANCE: Override account balance
        DELTA_LOG_LEVEL: Override log level
    """
    
    # Environment variable mappings
    ENV_MAPPINGS = {
        'DELTA_API_KEY': ('exchange', 'api_key'),
        'DELTA_API_SECRET': ('exchange', 'api_secret'),
        'DELTA_SANDBOX': ('exchange', 'sandbox'),
        'DELTA_TESTNET': ('exchange', 'testnet'),
        'DELTA_ACCOUNT_BALANCE': ('risk_management', 'account_balance'),
        'DELTA_LOG_LEVEL': ('database', 'log_level'),
        'DELTA_DB_PATH': ('database', 'db_path'),
        'DELTA_LOG_FILE': ('database', 'log_file'),
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self._config_path: Optional[Path] = Path(config_path) if config_path else None
        self._config: Optional[TradingBotConfig] = None
        self._raw_config: Dict[str, Any] = {}
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> TradingBotConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file. If not provided, uses the path
                        specified during initialization.
        
        Returns:
            TradingBotConfig: Validated configuration object
        
        Raises:
            FileNotFoundError: If configuration file does not exist
            ValueError: If configuration format is invalid
            ValidationError: If configuration fails validation
        """
        # Use provided path or stored path
        path = Path(config_path) if config_path else self._config_path
        
        if path is None:
            # No config file, use defaults with environment overrides
            self._config = self._apply_env_overrides(TradingBotConfig())
            return self._config
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Load based on file extension
        suffix = path.suffix.lower()
        
        try:
            with open(path, 'r') as f:
                if suffix in ['.yaml', '.yml']:
                    self._raw_config = yaml.safe_load(f)
                elif suffix == '.json':
                    self._raw_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {suffix}")
        except Exception as e:
            raise ValueError(f"Error reading config file: {e}")
        
        # Parse and validate
        self._config = TradingBotConfig.parse_obj(self._raw_config)
        
        # Apply environment overrides
        self._config = self._apply_env_overrides(self._config)
        
        return self._config
    
    def _apply_env_overrides(self, config: TradingBotConfig) -> TradingBotConfig:
        """Apply environment variable overrides to configuration."""
        config_dict = config.dict()
        
        for env_var, (section, key) in self.ENV_MAPPINGS.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if key in ['sandbox', 'testnet']:
                    value = value.lower() in ['true', '1', 'yes']
                elif key in ['account_balance']:
                    value = float(value)
                
                # Apply to config dict
                if section in config_dict:
                    config_dict[section][key] = value
        
        return TradingBotConfig.parse_obj(config_dict)
    
    def get_config(self) -> TradingBotConfig:
        """Get the current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None):
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. If not provided, uses the path
                        specified during initialization.
        """
        if self._config is None:
            raise RuntimeError("No configuration to save")
        
        path = Path(config_path) if config_path else self._config_path
        if path is None:
            raise ValueError("No config path specified")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        suffix = path.suffix.lower()
        
        with open(path, 'w') as f:
            if suffix in ['.yaml', '.yml']:
                yaml.dump(self._config.dict(), f, default_flow_style=False)
            elif suffix == '.json':
                json.dump(self._config.dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {suffix}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> TradingBotConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        TradingBotConfig: Validated configuration object
    """
    manager = ConfigManager(config_path)
    return manager.load_config()
