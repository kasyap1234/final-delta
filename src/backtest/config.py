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
    maker_fee_pct: float = 0.02  # 0.02% maker fee
    taker_fee_pct: float = 0.06  # 0.06% taker fee
    latency_ms: int = 100  # Simulated order latency
    
    # Data source
    data_source: str = "api"  # csv, sqlite, or api
    data_dir: str = "data/backtest"
    
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
        
        # Parse dates
        start_date = datetime.fromisoformat(backtest_config.get('start_date', '2025-01-01T00:00:00Z'))
        end_date = datetime.fromisoformat(backtest_config.get('end_date', '2025-12-31T23:59:59Z'))
        
        # Parse slippage model
        slippage_model_str = simulation_config.get('slippage_model', 'percentage')
        slippage_model = SlippageModel(slippage_model_str)
        
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
            data_source=data_config.get('source', 'api'),
            data_dir=data_config.get('data_dir', 'data/backtest'),
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
        
        return True
