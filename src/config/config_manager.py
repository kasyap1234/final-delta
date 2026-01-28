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
        default=60.0, ge=0, le=100,
        description="RSI threshold for long signals"
    )
    rsi_short_threshold: float = Field(
        default=40.0, ge=0, le=100,
        description="RSI threshold for short signals"
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
        if config_path:
            self._config_path = Path(config_path)
        
        if not self._config_path:
            raise ValueError("No configuration path specified")
        
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )
        
        # Load raw configuration from file
        self._raw_config = self._load_file(self._config_path)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate and create configuration object
        try:
            self._config = TradingBotConfig(**self._raw_config)
        except ValidationError as e:
            raise ValidationError(
                f"Configuration validation failed: {e}"
            ) from e
        
        return self._config
    
    def _load_file(self, path: Path) -> Dict[str, Any]:
        """
        Load configuration from file based on extension.
        
        Args:
            path: Path to configuration file
        
        Returns:
            Dict containing configuration data
        
        Raises:
            ValueError: If file format is not supported
        """
        suffix = path.suffix.lower()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif suffix == '.json':
                    return json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported configuration format: {suffix}. "
                        "Use .yaml, .yml, or .json"
                    )
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables take precedence over file configuration.
        """
        for env_var, (section, key) in self.ENV_MAPPINGS.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value, section, key)
                
                # Ensure section exists
                if section not in self._raw_config:
                    self._raw_config[section] = {}
                
                # Apply override
                self._raw_config[section][key] = converted_value
    
    def _convert_env_value(
        self, value: str, section: str, key: str
    ) -> Union[str, bool, int, float]:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable
            section: Configuration section name
            key: Configuration key name
        
        Returns:
            Converted value with appropriate type
        """
        # Boolean fields
        bool_fields = [
            ('exchange', 'sandbox'),
            ('exchange', 'testnet'),
            ('order', 'post_only'),
        ]
        
        if (section, key) in bool_fields:
            return value.lower() in ('true', '1', 'yes', 'on')
        
        # Integer fields
        int_fields = [
            ('trading', 'max_positions'),
            ('strategy', 'ema_fast'),
            ('strategy', 'ema_medium'),
            ('strategy', 'ema_slow'),
            ('strategy', 'ema_trend'),
            ('strategy', 'rsi_period'),
            ('strategy', 'atr_period'),
            ('strategy', 'pivot_lookback'),
            ('hedge', 'hedge_chunks'),
            ('hedge', 'correlation_lookback_days'),
            ('hedge', 'max_hedges_per_position'),
            ('order', 'retry_attempts'),
            ('order', 'retry_delay_seconds'),
        ]
        
        if (section, key) in int_fields:
            try:
                return int(value)
            except ValueError:
                raise ValueError(
                    f"Environment variable {section.upper()}_{key.upper()} "
                    f"must be an integer, got: {value}"
                )
        
        # Float fields
        float_fields = [
            ('strategy', 'rsi_long_threshold'),
            ('strategy', 'rsi_short_threshold'),
            ('strategy', 'atr_multiplier'),
            ('risk_management', 'account_balance'),
            ('risk_management', 'risk_per_trade_percent'),
            ('risk_management', 'max_risk_per_trade_percent'),
            ('risk_management', 'take_profit_r_ratio'),
            ('hedge', 'hedge_trigger_percent'),
            ('hedge', 'hedge_size_ratio'),
            ('order', 'price_offset_percent'),
        ]
        
        if (section, key) in float_fields:
            try:
                return float(value)
            except ValueError:
                raise ValueError(
                    f"Environment variable {section.upper()}_{key.upper()} "
                    f"must be a number, got: {value}"
                )
        
        # String fields (default)
        return value
    
    def save_config(
        self, config_path: Optional[Union[str, Path]] = None, format: str = 'yaml'
    ) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. If not provided, uses the
                        path specified during initialization.
            format: Output format ('yaml' or 'json')
        
        Raises:
            ValueError: If no configuration is loaded or format is invalid
        """
        if not self._config:
            raise ValueError("No configuration loaded to save")
        
        save_path = Path(config_path) if config_path else self._config_path
        if not save_path:
            raise ValueError("No save path specified")
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict
        config_dict = self._config.dict()
        
        # Write to file
        with open(save_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
    
    def get_config(self) -> TradingBotConfig:
        """
        Get the current configuration.
        
        Returns:
            TradingBotConfig: Current configuration object
        
        Raises:
            ValueError: If no configuration is loaded
        """
        if not self._config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        return self._config
    
    def get_trading_params(self) -> TradingSettings:
        """Get trading-specific parameters."""
        return self.get_config().trading
    
    def get_risk_params(self) -> RiskManagementSettings:
        """Get risk management parameters."""
        return self.get_config().risk_management
    
    def get_hedge_params(self) -> HedgeSettings:
        """Get hedge configuration parameters."""
        return self.get_config().hedge
    
    def get_exchange_params(self) -> ExchangeSettings:
        """Get exchange connection parameters."""
        return self.get_config().exchange
    
    def get_strategy_params(self) -> StrategySettings:
        """Get strategy parameters."""
        return self.get_config().strategy
    
    def get_order_params(self) -> OrderSettings:
        """Get order execution parameters."""
        return self.get_config().order
    
    def get_database_params(self) -> DatabaseSettings:
        """Get database and logging parameters."""
        return self.get_config().database
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid
        
        Raises:
            ValidationError: If configuration is invalid
        """
        if not self._config:
            raise ValueError("No configuration loaded")
        
        # Re-validate by creating a new instance
        try:
            TradingBotConfig(**self._config.dict())
            return True
        except ValidationError:
            raise
    
    def reload(self) -> TradingBotConfig:
        """
        Reload configuration from file.
        
        Returns:
            TradingBotConfig: Reloaded configuration object
        """
        return self.load_config(self._config_path)
    
    @property
    def is_loaded(self) -> bool:
        """Check if configuration is loaded."""
        return self._config is not None
    
    @classmethod
    def create_default_config(cls, config_path: Union[str, Path]) -> TradingBotConfig:
        """
        Create a default configuration file.
        
        Args:
            config_path: Path where to save the default configuration
        
        Returns:
            TradingBotConfig: Default configuration object
        """
        config_path = Path(config_path)
        manager = cls()
        manager._config = TradingBotConfig()
        manager._config_path = config_path
        
        # Determine format from extension
        format = 'yaml' if config_path.suffix in ['.yaml', '.yml'] else 'json'
        manager.save_config(format=format)
        
        return manager._config


# Convenience function for quick access
def load_config(config_path: Union[str, Path]) -> TradingBotConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        TradingBotConfig: Validated configuration object
    """
    manager = ConfigManager(config_path)
    return manager.load_config()
