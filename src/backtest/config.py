"""
Backtest configuration module.

This module provides configuration classes for the backtesting system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import yaml
import logging

logger = logging.getLogger(__name__)


class SlippageModel(str, Enum):
    """Slippage model types."""
    PERCENTAGE = "percentage"
    FIXED = "fixed"
    NONE = "none"


class FeeScheduleType(str, Enum):
    """Fee schedule preset types."""
    DELTA = "delta"
    BINANCE = "binance"
    BYBIT = "bybit"
    DYDX = "dydx"
    CUSTOM = "custom"


@dataclass
class VolumeTierConfig:
    """Configuration for a volume tier."""
    min_volume_30d: float
    maker_fee_rate: float
    taker_fee_rate: float
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'min_volume_30d': self.min_volume_30d,
            'maker_fee_rate': self.maker_fee_rate,
            'taker_fee_rate': self.taker_fee_rate,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VolumeTierConfig':
        return cls(
            min_volume_30d=data.get('min_volume_30d', 0.0),
            maker_fee_rate=data.get('maker_fee_rate', 0.0002),
            taker_fee_rate=data.get('taker_fee_rate', 0.0006),
            name=data.get('name', 'Default')
        )


@dataclass
class FundingRateConfig:
    """Configuration for funding rates."""
    enabled: bool = True
    default_rate_annual: float = 0.10
    rate_update_interval_hours: int = 8
    max_rate_annual: float = 1.0
    min_rate_annual: float = -1.0
    rate_source: str = "synthetic"  # synthetic, historical, fixed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'default_rate_annual': self.default_rate_annual,
            'rate_update_interval_hours': self.rate_update_interval_hours,
            'max_rate_annual': self.max_rate_annual,
            'min_rate_annual': self.min_rate_annual,
            'rate_source': self.rate_source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FundingRateConfig':
        return cls(
            enabled=data.get('enabled', True),
            default_rate_annual=data.get('default_rate_annual', 0.10),
            rate_update_interval_hours=data.get('rate_update_interval_hours', 8),
            max_rate_annual=data.get('max_rate_annual', 1.0),
            min_rate_annual=data.get('min_rate_annual', -1.0),
            rate_source=data.get('rate_source', 'synthetic')
        )


@dataclass
class FeeConfig:
    """Configuration for trading fees."""
    fee_schedule: FeeScheduleType = FeeScheduleType.DELTA
    maker_fee_rate: float = 0.0002  # 0.02%
    taker_fee_rate: float = 0.0006  # 0.06%
    use_volume_tiers: bool = True
    volume_tiers: List[VolumeTierConfig] = field(default_factory=list)
    funding_config: FundingRateConfig = field(default_factory=FundingRateConfig)
    withdrawal_fee_fixed: float = 0.0
    deposit_fee_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fee_schedule': self.fee_schedule.value,
            'maker_fee_rate': self.maker_fee_rate,
            'taker_fee_rate': self.taker_fee_rate,
            'use_volume_tiers': self.use_volume_tiers,
            'volume_tiers': [tier.to_dict() for tier in self.volume_tiers],
            'funding_config': self.funding_config.to_dict(),
            'withdrawal_fee_fixed': self.withdrawal_fee_fixed,
            'deposit_fee_percent': self.deposit_fee_percent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeeConfig':
        # Parse volume tiers
        tiers_data = data.get('volume_tiers', [])
        volume_tiers = [VolumeTierConfig.from_dict(t) for t in tiers_data]
        
        # Parse funding config
        funding_data = data.get('funding_config', {})
        funding_config = FundingRateConfig.from_dict(funding_data)
        
        # Parse fee schedule type
        schedule_str = data.get('fee_schedule', 'delta')
        try:
            fee_schedule = FeeScheduleType(schedule_str)
        except ValueError:
            fee_schedule = FeeScheduleType.DELTA
        
        return cls(
            fee_schedule=fee_schedule,
            maker_fee_rate=data.get('maker_fee_rate', 0.0002),
            taker_fee_rate=data.get('taker_fee_rate', 0.0006),
            use_volume_tiers=data.get('use_volume_tiers', True),
            volume_tiers=volume_tiers,
            funding_config=funding_config,
            withdrawal_fee_fixed=data.get('withdrawal_fee_fixed', 0.0),
            deposit_fee_percent=data.get('deposit_fee_percent', 0.0)
        )


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Date range
    start_date: datetime = datetime(2025, 1, 1)
    end_date: datetime = datetime(2025, 12, 31, 23, 59, 59)
    
    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD", "SOL/USD"])
    
    # Timeframe
    timeframe: str = "15m"
    
    # Initial account state
    initial_balance: float = 10000.0
    initial_currency: str = "USD"
    
    # Order simulation
    slippage_model: SlippageModel = SlippageModel.PERCENTAGE
    slippage_pct: float = 0.01  # 0.01% slippage
    maker_fee_pct: float = 0.02  # 0.02% maker fee (legacy, use fee_config)
    taker_fee_pct: float = 0.06  # 0.06% taker fee (legacy, use fee_config)
    latency_ms: int = 100  # Simulated order latency
    
    # Fee configuration (new)
    fee_config: FeeConfig = field(default_factory=FeeConfig)
    
    # Data source
    data_source: str = "api"  # csv, sqlite, or api
    data_dir: str = "data/backtest"
    data_year: Optional[int] = None  # Year for year-specific data files
    symbol_files: Dict[str, str] = field(default_factory=dict)  # Symbol to filename mapping
    
    # Output
    output_dir: str = "backtest_results"
    save_trade_log: bool = True
    save_equity_curve: bool = True
    generate_report: bool = True
    
    # Trading bot config (same as live)
    trading_bot_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """Create BacktestConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            BacktestConfig instance
        """
        backtest_config = config_dict.get('backtest', {})
        simulation_config = config_dict.get('simulation', {})
        data_config = config_dict.get('data', {})
        fee_config_data = config_dict.get('fees', {})
        
        # Parse dates
        start_date = datetime.fromisoformat(backtest_config.get('start_date', '2025-01-01T00:00:00Z'))
        end_date = datetime.fromisoformat(backtest_config.get('end_date', '2025-12-31T23:59:59Z'))
        
        # Parse slippage model
        slippage_model_str = simulation_config.get('slippage_model', 'percentage')
        slippage_model = SlippageModel(slippage_model_str)
        
        # Parse fee configuration
        fee_config = FeeConfig.from_dict(fee_config_data)
        
        return cls(
            start_date=start_date,
            end_date=end_date,
            symbols=backtest_config.get('symbols', ["BTC/USD", "ETH/USD", "SOL/USD"]),
            timeframe=backtest_config.get('timeframe', '15m'),
            initial_balance=backtest_config.get('initial_balance', 10000.0),
            initial_currency=backtest_config.get('initial_currency', 'USD'),
            slippage_model=slippage_model,
            slippage_pct=simulation_config.get('slippage_percent', 0.01),
            maker_fee_pct=simulation_config.get('maker_fee_percent', 0.02),
            taker_fee_pct=simulation_config.get('taker_fee_percent', 0.06),
            latency_ms=simulation_config.get('latency_ms', 100),
            fee_config=fee_config,
            data_source=data_config.get('source', 'api'),
            data_dir=data_config.get('data_dir', 'data/backtest'),
            data_year=data_config.get('year'),
            symbol_files=data_config.get('symbol_files', {}),
            output_dir=backtest_config.get('output_dir', 'backtest_results'),
            save_trade_log=backtest_config.get('save_trade_log', True),
            save_equity_curve=backtest_config.get('save_equity_curve', True),
            generate_report=backtest_config.get('generate_report', True),
            trading_bot_config=config_dict.get('trading_bot')
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BacktestConfig':
        """Load BacktestConfig from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            BacktestConfig instance
        """
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            logger.info(f"Loaded backtest configuration from {yaml_path}")
            return cls.from_dict(config_dict)
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {yaml_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BacktestConfig to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'backtest': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'initial_balance': self.initial_balance,
                'initial_currency': self.initial_currency,
                'output_dir': self.output_dir,
                'save_trade_log': self.save_trade_log,
                'save_equity_curve': self.save_equity_curve,
                'generate_report': self.generate_report
            },
            'simulation': {
                'slippage_model': self.slippage_model.value,
                'slippage_percent': self.slippage_pct,
                'maker_fee_percent': self.maker_fee_pct,
                'taker_fee_percent': self.taker_fee_pct,
                'latency_ms': self.latency_ms
            },
            'fees': self.fee_config.to_dict(),
            'data': {
                'source': self.data_source,
                'data_dir': self.data_dir
            },
            'trading_bot': self.trading_bot_config
        }
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")
        
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")
        
        if self.maker_fee_pct < 0 or self.taker_fee_pct < 0:
            raise ValueError("fees cannot be negative")
        
        if self.timeframe not in ['1m', '5m', '15m', '1h', '4h', '1d']:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
        
        if self.data_source not in ['csv', 'sqlite', 'api']:
            raise ValueError(f"Unsupported data source: {self.data_source}")
        
        # Validate fee configuration
        if self.fee_config.maker_fee_rate < 0 or self.fee_config.taker_fee_rate < 0:
            raise ValueError("fee rates cannot be negative")
        
        if self.fee_config.funding_config.rate_update_interval_hours <= 0:
            raise ValueError("funding rate update interval must be positive")
        
        return True
    
    def create_fee_calculator(self) -> 'FeeCalculator':
        """
        Create a FeeCalculator instance from this configuration.
        
        Returns:
            FeeCalculator instance configured according to this config
        """
        from src.backtest.fees import (
            FeeCalculator, FeeSchedule, FeeSchedulePresets,
            VolumeTier, FundingRateConfig as FeeFundingConfig
        )
        
        # Use preset or custom fee schedule
        if self.fee_config.fee_schedule != FeeScheduleType.CUSTOM:
            try:
                fee_schedule = FeeSchedulePresets.get_preset(self.fee_config.fee_schedule.value)
            except ValueError:
                fee_schedule = FeeSchedulePresets.delta_exchange()
        else:
            # Build custom fee schedule
            volume_tiers = []
            for tier_config in self.fee_config.volume_tiers:
                volume_tiers.append(VolumeTier(
                    min_volume_30d=tier_config.min_volume_30d,
                    maker_fee_rate=tier_config.maker_fee_rate,
                    taker_fee_rate=tier_config.taker_fee_rate,
                    name=tier_config.name
                ))
            
            funding_config = FeeFundingConfig(
                enabled=self.fee_config.funding_config.enabled,
                default_rate_annual=self.fee_config.funding_config.default_rate_annual,
                rate_update_interval_hours=self.fee_config.funding_config.rate_update_interval_hours,
                max_rate_annual=self.fee_config.funding_config.max_rate_annual,
                min_rate_annual=self.fee_config.funding_config.min_rate_annual,
                rate_source=self.fee_config.funding_config.rate_source
            )
            
            fee_schedule = FeeSchedule(
                exchange_name=self.fee_config.fee_schedule.value,
                default_maker_fee=self.fee_config.maker_fee_rate,
                default_taker_fee=self.fee_config.taker_fee_rate,
                volume_tiers=volume_tiers,
                funding_config=funding_config,
                withdrawal_fee_fixed=self.fee_config.withdrawal_fee_fixed,
                deposit_fee_percent=self.fee_config.deposit_fee_percent,
                use_volume_tiers=self.fee_config.use_volume_tiers
            )
        
        return FeeCalculator(
            fee_schedule=fee_schedule,
            quote_currency=self.initial_currency
        )
