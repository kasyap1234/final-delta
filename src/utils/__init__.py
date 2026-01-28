"""
Utilities package for the cryptocurrency trading bot.

This package provides utility modules including:
- Logging system with structured logging and database integration
- Log formatters (JSON, colored, detailed)
- Custom log handlers (database, rotating files, console)

Example Usage:
    from utils import get_logger, setup_logging, log_context
    from config import load_config

    # Setup logging
    config = load_config('config/config.yaml')
    setup_logging(config)

    # Get logger
    logger = get_logger('trading_bot.signals')

    # Log with context
    logger.info("Buy signal generated", extra={
        'category': 'TRADING',
        'data': {'symbol': 'BTC/USD', 'price': 50000}
    })

    # Use correlation context
    with log_context(correlation_id='req-123'):
        logger.info("Processing request")

    # Category-specific logging
    logger.log_trade({
        'trade_id': 'trade_001',
        'symbol': 'BTC/USD',
        'side': 'buy',
        'price': 50000,
        'quantity': 0.1
    })
"""

# Import main logger functions
from .logger import (
    # Main functions
    setup_logging,
    get_logger,
    shutdown_logging,
    log_context,
    set_global_correlation_id,
    clear_global_correlation_id,
    
    # Classes
    LoggerAdapter,
    LoggerManager,
    LogCategory,
)

# Import formatters
from .log_formatter import (
    JsonFormatter,
    ColoredFormatter,
    DetailedFormatter,
    CompactFormatter,
    CategoryFilter,
    LevelFilter,
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
    MemoryHandler,
)

__all__ = [
    # Main logger functions
    'setup_logging',
    'get_logger',
    'shutdown_logging',
    'log_context',
    'set_global_correlation_id',
    'clear_global_correlation_id',
    
    # Logger classes
    'LoggerAdapter',
    'LoggerManager',
    'LogCategory',
    
    # Formatters
    'JsonFormatter',
    'ColoredFormatter',
    'DetailedFormatter',
    'CompactFormatter',
    'CategoryFilter',
    'LevelFilter',
    
    # Handlers
    'DatabaseLogHandler',
    'AsyncDatabaseLogHandler',
    'TimedRotatingFileHandler',
    'SizeRotatingFileHandler',
    'ColoredConsoleHandler',
    'ErrorFileHandler',
    'CategoryFileHandler',
    'MemoryHandler',
]
