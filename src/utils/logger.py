"""
Logging system for the cryptocurrency trading bot.

This module provides a comprehensive logging system with structured logging,
multiple handlers, database integration, and category-based logging.

Example Usage:
    from utils import get_logger, setup_logging
    from config import load_config

    # Setup logging
    config = load_config('config/config.yaml')
    setup_logging(config)

    # Get logger
    logger = get_logger('trading_bot.signals')

    # Log with context
    logger.info("Buy signal generated", extra={
        'category': 'TRADING',
        'correlation_id': 'abc-123',
        'data': {
            'symbol': 'BTC/USD',
            'price': 50000,
            'strength': 0.85
        }
    })

    # Log trade event
    logger.log_trade({
        'trade_id': 'trade_001',
        'symbol': 'BTC/USD',
        'side': 'buy',
        'price': 50000,
        'quantity': 0.1
    })
"""

import logging
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import formatters
from .log_formatter import (
    JsonFormatter,
    ColoredFormatter,
    DetailedFormatter,
    CompactFormatter,
    CategoryFilter,
    LevelFilter
)

# Import handlers
from .log_handlers import (
    DatabaseLogHandler,
    AsyncDatabaseLogHandler,
    TimedRotatingFileHandler,
    SizeRotatingFileHandler,
    ColoredConsoleHandler,
    ErrorFileHandler,
    CategoryFileHandler,
    MemoryHandler
)


class LogCategory(Enum):
    """Log categories for organizing log output."""
    TRADING = "TRADING"
    RISK = "RISK"
    HEDGE = "HEDGE"
    ORDERS = "ORDERS"
    SYSTEM = "SYSTEM"
    WEBSOCKET = "WEBSOCKET"
    SIGNALS = "SIGNALS"
    GENERAL = "GENERAL"


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds correlation ID and category support.
    
    Provides convenient methods for logging with context and
    category-specific logging methods.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize logger adapter.
        
        Args:
            logger: Base logger instance
            extra: Default extra fields
        """
        super().__init__(logger, extra or {})
        self._correlation_id = None
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message and kwargs.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
            
        Returns:
            Tuple of (message, kwargs)
        """
        # Ensure extra dict exists
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add correlation ID if set
        if self._correlation_id and 'correlation_id' not in kwargs['extra']:
            kwargs['extra']['correlation_id'] = self._correlation_id
        
        # Merge default extra fields
        for key, value in self.extra.items():
            if key not in kwargs['extra']:
                kwargs['extra'][key] = value
        
        return msg, kwargs
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """
        Set correlation ID for this logger.
        
        Args:
            correlation_id: Correlation ID string
        """
        self._correlation_id = correlation_id
    
    def clear_correlation_id(self) -> None:
        """Clear the current correlation ID."""
        self._correlation_id = None
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """
        Context manager for correlation ID scope.
        
        Args:
            correlation_id: Correlation ID (generates UUID if None)
            
        Example:
            with logger.correlation_context():
                logger.info("Processing request")
                # All logs in this block have the same correlation ID
        """
        old_id = self._correlation_id
        self._correlation_id = correlation_id or str(uuid.uuid4())
        try:
            yield self._correlation_id
        finally:
            self._correlation_id = old_id
    
    # Category-specific logging methods
    def log_trade(self, trade_data: Dict[str, Any], msg: str = "", level: int = logging.INFO) -> None:
        """
        Log a trade event.
        
        Args:
            trade_data: Trade data dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = f"Trade: {trade_data.get('side', 'unknown')} {trade_data.get('symbol', 'unknown')}"
        
        self.log(level, msg, extra={
            'category': LogCategory.TRADING.value,
            'trade_data': trade_data
        })
    
    def log_signal(self, signal_data: Dict[str, Any], msg: str = "", level: int = logging.INFO) -> None:
        """
        Log a signal event.
        
        Args:
            signal_data: Signal data dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = f"Signal: {signal_data.get('signal_type', 'unknown')} {signal_data.get('symbol', 'unknown')}"
        
        self.log(level, msg, extra={
            'category': LogCategory.SIGNALS.value,
            'signal_data': signal_data
        })
    
    def log_risk_event(self, event_data: Dict[str, Any], msg: str = "", level: int = logging.WARNING) -> None:
        """
        Log a risk management event.
        
        Args:
            event_data: Risk event data dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = f"Risk event: {event_data.get('event_type', 'unknown')}"
        
        self.log(level, msg, extra={
            'category': LogCategory.RISK.value,
            'risk_data': event_data
        })
    
    def log_hedge_event(self, event_data: Dict[str, Any], msg: str = "", level: int = logging.INFO) -> None:
        """
        Log a hedge operation event.
        
        Args:
            event_data: Hedge event data dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = f"Hedge: {event_data.get('primary_symbol', 'unknown')} / {event_data.get('hedge_symbol', 'unknown')}"
        
        self.log(level, msg, extra={
            'category': LogCategory.HEDGE.value,
            'hedge_data': event_data
        })
    
    def log_order_event(self, event_data: Dict[str, Any], msg: str = "", level: int = logging.INFO) -> None:
        """
        Log an order execution event.
        
        Args:
            event_data: Order event data dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = f"Order: {event_data.get('side', 'unknown')} {event_data.get('order_type', 'unknown')} {event_data.get('symbol', 'unknown')}"
        
        self.log(level, msg, extra={
            'category': LogCategory.ORDERS.value,
            'order_data': event_data
        })
    
    def log_system_event(self, event_data: Dict[str, Any], msg: str = "", level: int = logging.INFO) -> None:
        """
        Log a system event.
        
        Args:
            event_data: System event data dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = f"System: {event_data.get('event_type', 'unknown')}"
        
        self.log(level, msg, extra={
            'category': LogCategory.SYSTEM.value,
            'system_data': event_data
        })
    
    def log_websocket_event(self, event_data: Dict[str, Any], msg: str = "", level: int = logging.DEBUG) -> None:
        """
        Log a WebSocket event.
        
        Args:
            event_data: WebSocket event data dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = f"WebSocket: {event_data.get('event_type', 'unknown')}"
        
        self.log(level, msg, extra={
            'category': LogCategory.WEBSOCKET.value,
            'websocket_data': event_data
        })
    
    def log_performance_metrics(self, metrics: Dict[str, Any], msg: str = "", level: int = logging.INFO) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
            msg: Optional message
            level: Log level
        """
        if not msg:
            msg = "Performance metrics"
        
        self.log(level, msg, extra={
            'category': LogCategory.SYSTEM.value,
            'performance_metrics': metrics
        })


class LoggerManager:
    """
    Manager for the logging system.
    
    Handles initialization, configuration, and lifecycle of loggers.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for LoggerManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize logger manager."""
        if self._initialized:
            return
        
        self._initialized = True
        self._loggers: Dict[str, LoggerAdapter] = {}
        self._config: Dict[str, Any] = {}
        self._handlers: List[logging.Handler] = []
        self._setup_done = False
    
    def setup_logging(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Setup the logging system with configuration.
        
        Args:
            config: Logging configuration dictionary
        """
        if self._setup_done:
            return
        
        self._config = config or {}
        
        # Get logging config
        log_config = self._config.get('logging', {})
        
        # Set root logger level
        root_level = self._get_log_level(log_config.get('level', 'INFO'))
        logging.getLogger().setLevel(root_level)
        
        # Clear existing handlers
        logging.getLogger().handlers = []
        
        # Setup handlers based on configuration
        if log_config.get('console', True):
            self._setup_console_handler(log_config.get('console_config', {}))
        
        if log_config.get('file', False):
            self._setup_file_handler(log_config.get('file_config', {}))
        
        if log_config.get('error_file', False):
            self._setup_error_handler(log_config.get('error_file_config', {}))
        
        if log_config.get('database', False):
            self._setup_database_handler(log_config.get('database_config', {}))
        
        if log_config.get('category_files', False):
            self._setup_category_handler(log_config.get('category_config', {}))
        
        self._setup_done = True
        
        # Log startup message
        logger = self.get_logger('system')
        logger.info("Logging system initialized", extra={
            'category': LogCategory.SYSTEM.value,
            'config': {
                'level': log_config.get('level', 'INFO'),
                'console': log_config.get('console', True),
                'file': log_config.get('file', False),
                'database': log_config.get('database', False)
            }
        })
    
    def _get_log_level(self, level: Union[str, int]) -> int:
        """
        Convert log level string to constant.
        
        Args:
            level: Log level string or int
            
        Returns:
            Log level constant
        """
        if isinstance(level, int):
            return level
        
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(level.upper(), logging.INFO)
    
    def _setup_console_handler(self, config: Dict[str, Any]) -> None:
        """
        Setup console handler.
        
        Args:
            config: Console handler configuration
        """
        handler = ColoredConsoleHandler(sys.stdout)
        
        level = self._get_log_level(config.get('level', 'DEBUG'))
        handler.setLevel(level)
        
        # Use colored formatter
        use_colors = config.get('colors', True)
        formatter = ColoredFormatter(use_colors=use_colors)
        handler.setFormatter(formatter)
        
        # Add category filter if specified
        categories = config.get('categories')
        if categories:
            handler.addFilter(CategoryFilter(include_categories=categories))
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def _setup_file_handler(self, config: Dict[str, Any]) -> None:
        """
        Setup file handler with rotation.
        
        Args:
            config: File handler configuration
        """
        log_dir = config.get('directory', 'logs')
        filename = config.get('filename', 'trading_bot.log')
        filepath = Path(log_dir) / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        rotation_type = config.get('rotation', 'time')
        
        if rotation_type == 'time':
            handler = TimedRotatingFileHandler(
                filename=str(filepath),
                when=config.get('when', 'midnight'),
                interval=config.get('interval', 1),
                backupCount=config.get('backup_count', 30)
            )
        else:  # size
            handler = SizeRotatingFileHandler(
                filename=str(filepath),
                maxBytes=config.get('max_bytes', 10*1024*1024),
                backupCount=config.get('backup_count', 10)
            )
        
        level = self._get_log_level(config.get('level', 'INFO'))
        handler.setLevel(level)
        
        # Use JSON formatter for file logs
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def _setup_error_handler(self, config: Dict[str, Any]) -> None:
        """
        Setup error file handler.
        
        Args:
            config: Error handler configuration
        """
        log_dir = config.get('directory', 'logs')
        filename = config.get('filename', 'errors.log')
        filepath = Path(log_dir) / filename
        
        handler = ErrorFileHandler(str(filepath))
        
        # Use detailed formatter for errors
        formatter = DetailedFormatter()
        handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def _setup_database_handler(self, config: Dict[str, Any]) -> None:
        """
        Setup database handler.
        
        Args:
            config: Database handler configuration
        """
        db_path = config.get('db_path', 'data/trading_bot.db')
        
        if config.get('async', True):
            handler = AsyncDatabaseLogHandler(
                db_path=db_path,
                max_queue_size=config.get('max_queue_size', 10000),
                flush_interval=config.get('flush_interval', 5.0)
            )
        else:
            handler = DatabaseLogHandler(db_path=db_path)
        
        level = self._get_log_level(config.get('level', 'INFO'))
        handler.setLevel(level)
        
        # Add category filter if specified
        categories = config.get('categories')
        if categories:
            handler.addFilter(CategoryFilter(include_categories=categories))
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def _setup_category_handler(self, config: Dict[str, Any]) -> None:
        """
        Setup category-based file handler.
        
        Args:
            config: Category handler configuration
        """
        log_dir = config.get('directory', 'logs')
        handler = CategoryFileHandler(base_dir=log_dir)
        
        # Add categories
        categories = config.get('categories', {
            'TRADING': 'trading.log',
            'RISK': 'risk.log',
            'HEDGE': 'hedge.log',
            'ORDERS': 'orders.log',
            'SYSTEM': 'system.log',
            'WEBSOCKET': 'websocket.log'
        })
        
        for category, filename in categories.items():
            handler.add_category(
                category=category,
                filename=filename,
                level=self._get_log_level(config.get('level', 'DEBUG'))
            )
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
    
    def get_logger(self, name: str) -> LoggerAdapter:
        """
        Get a logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            LoggerAdapter instance
        """
        if name not in self._loggers:
            base_logger = logging.getLogger(name)
            self._loggers[name] = LoggerAdapter(base_logger)
        
        return self._loggers[name]
    
    def shutdown(self) -> None:
        """Shutdown the logging system."""
        # Close all handlers
        for handler in self._handlers:
            handler.close()
        
        # Clear handlers list
        self._handlers = []
        
        # Shutdown logging
        logging.shutdown()
        
        self._setup_done = False


# Global logger manager instance
_logger_manager = LoggerManager()


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup the logging system.
    
    Args:
        config: Logging configuration dictionary
        
    Example:
        setup_logging({
            'logging': {
                'level': 'INFO',
                'console': True,
                'file': True,
                'file_config': {
                    'directory': 'logs',
                    'filename': 'trading_bot.log'
                },
                'database': True,
                'database_config': {
                    'db_path': 'data/trading_bot.db'
                }
            }
        })
    """
    _logger_manager.setup_logging(config)


def get_logger(name: str) -> LoggerAdapter:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        LoggerAdapter instance
        
    Example:
        logger = get_logger('trading_bot.signals')
        logger.info("Signal generated")
    """
    return _logger_manager.get_logger(name)


def shutdown_logging() -> None:
    """Shutdown the logging system."""
    _logger_manager.shutdown()


@contextmanager
def log_context(correlation_id: Optional[str] = None, **extra):
    """
    Context manager for logging with correlation ID and extra fields.
    
    Args:
        correlation_id: Correlation ID (generates UUID if None)
        **extra: Extra fields to add to all logs in context
        
    Example:
        with log_context(correlation_id='req-123', user_id='user-456'):
            logger.info("Processing request")
            # All logs have correlation_id and user_id
    """
    cid = correlation_id or str(uuid.uuid4())
    
    # Store old extra fields
    old_extra = {}
    
    # Create a temporary adapter with the context
    logger = get_logger('context')
    
    # Set correlation ID
    logger.set_correlation_id(cid)
    
    # Add extra fields
    for key, value in extra.items():
        if key not in logger.extra:
            old_extra[key] = None
        else:
            old_extra[key] = logger.extra[key]
        logger.extra[key] = value
    
    try:
        yield logger
    finally:
        # Restore old state
        logger.clear_correlation_id()
        for key, value in old_extra.items():
            if value is None:
                del logger.extra[key]
            else:
                logger.extra[key] = value


def set_global_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set a global correlation ID for all loggers.
    
    Args:
        correlation_id: Correlation ID (generates UUID if None)
        
    Returns:
        The correlation ID
    """
    cid = correlation_id or str(uuid.uuid4())
    
    for logger in _logger_manager._loggers.values():
        logger.set_correlation_id(cid)
    
    return cid


def clear_global_correlation_id() -> None:
    """Clear the global correlation ID from all loggers."""
    for logger in _logger_manager._loggers.values():
        logger.clear_correlation_id()
