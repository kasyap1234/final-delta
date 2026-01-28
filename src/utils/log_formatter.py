"""
Log formatters for the cryptocurrency trading bot.

This module provides custom log formatters for structured logging including:
- JSON format for machine parsing
- Colored console output
- Detailed text format
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON objects with standardized fields for
    machine parsing and log aggregation systems.
    
    Example output:
    {
        "timestamp": "2024-01-27T10:30:00.123456Z",
        "level": "INFO",
        "logger": "trading_bot.signals",
        "correlation_id": "abc-123-def",
        "message": "Buy signal generated",
        "category": "TRADING",
        "data": {...}
    }
    """
    
    def __init__(
        self,
        include_extra: bool = True,
        indent: Optional[int] = None,
        default_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize JSON formatter.
        
        Args:
            include_extra: Include extra fields from log record
            indent: JSON indentation (None for compact, int for pretty print)
            default_fields: Default fields to include in every log entry
        """
        super().__init__()
        self.include_extra = include_extra
        self.indent = indent
        self.default_fields = default_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_data['correlation_id'] = record.correlation_id
        
        # Add category if present
        if hasattr(record, 'category') and record.category:
            log_data['category'] = record.category
        
        # Add source location
        log_data['source'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_data = self._get_extra_fields(record)
            if extra_data:
                log_data['data'] = extra_data
        
        # Add default fields
        log_data.update(self.default_fields)
        
        return json.dumps(log_data, indent=self.indent, default=str)
    
    def _get_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Extract extra fields from log record.
        
        Args:
            record: Log record
            
        Returns:
            Dictionary of extra fields
        """
        # Standard log record attributes to exclude
        standard_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
            'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'correlation_id', 'category'
        }
        
        extra = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                extra[key] = value
        
        return extra


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for human-readable logs.
    
    Uses ANSI color codes to highlight different log levels and
    make logs easier to read in terminal output.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
        show_category: bool = True
    ):
        """
        Initialize colored formatter.
        
        Args:
            fmt: Format string (uses default if None)
            datefmt: Date format string
            use_colors: Enable/disable colors
            show_category: Show log category in output
        """
        if fmt is None:
            fmt = self._get_default_format(show_category)
        
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        self.show_category = show_category
    
    def _get_default_format(self, show_category: bool) -> str:
        """Get default format string."""
        if show_category:
            return '%(asctime)s | %(levelname)-8s | %(category)s | %(name)s | %(message)s'
        else:
            return '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Colored formatted log string
        """
        # Ensure category attribute exists
        if not hasattr(record, 'category'):
            record.category = 'GENERAL'
        
        # Apply colors
        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, self.RESET)
            
            # Format with colors
            formatted = super().format(record)
            
            # Color the level name
            level_str = f"{level_color}{record.levelname}{self.RESET}"
            formatted = formatted.replace(record.levelname, level_str)
            
            # Color the category if present
            if hasattr(record, 'category') and record.category:
                category_color = self.DIM
                category_str = f"{category_color}{record.category}{self.RESET}"
                formatted = formatted.replace(record.category, category_str)
            
            return formatted
        else:
            return super().format(record)


class DetailedFormatter(logging.Formatter):
    """
    Detailed text formatter for comprehensive log output.
    
    Provides detailed information including timestamps, source location,
    and structured data in a human-readable format.
    """
    
    def __init__(
        self,
        include_source: bool = True,
        include_data: bool = True,
        max_data_length: int = 1000
    ):
        """
        Initialize detailed formatter.
        
        Args:
            include_source: Include source file and line number
            include_data: Include extra data fields
            max_data_length: Maximum length for data field display
        """
        super().__init__()
        self.include_source = include_source
        self.include_data = include_data
        self.max_data_length = max_data_length
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with detailed information.
        
        Args:
            record: Log record to format
            
        Returns:
            Detailed formatted log string
        """
        lines = []
        
        # Header line
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        logger_name = record.name
        
        header = f"[{timestamp}] {level:8} | {logger_name}"
        
        # Add category if present
        if hasattr(record, 'category') and record.category:
            header += f" | [{record.category}]"
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id') and record.correlation_id:
            header += f" | corr_id={record.correlation_id}"
        
        lines.append(header)
        
        # Source location
        if self.include_source:
            source = f"  Source: {record.pathname}:{record.lineno} ({record.funcName})"
            lines.append(source)
        
        # Message
        message = record.getMessage()
        lines.append(f"  Message: {message}")
        
        # Extra data
        if self.include_data:
            extra_data = self._get_extra_fields(record)
            if extra_data:
                data_str = json.dumps(extra_data, indent=2, default=str)
                if len(data_str) > self.max_data_length:
                    data_str = data_str[:self.max_data_length] + "... [truncated]"
                lines.append(f"  Data: {data_str}")
        
        # Exception info
        if record.exc_info:
            lines.append("  Exception:")
            exc_text = self.formatException(record.exc_info)
            for line in exc_text.split('\n'):
                lines.append(f"    {line}")
        
        return '\n'.join(lines)
    
    def _get_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Extract extra fields from log record.
        
        Args:
            record: Log record
            
        Returns:
            Dictionary of extra fields
        """
        standard_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
            'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'correlation_id', 'category', 'asctime'
        }
        
        extra = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                extra[key] = value
        
        return extra


class CompactFormatter(logging.Formatter):
    """
    Compact formatter for concise log output.
    
    Suitable for high-volume logging where brevity is preferred.
    """
    
    def __init__(self, show_category: bool = True):
        """
        Initialize compact formatter.
        
        Args:
            show_category: Show log category in output
        """
        if show_category:
            fmt = '%(asctime)s %(levelname)s [%(category)s] %(message)s'
        else:
            fmt = '%(asctime)s %(levelname)s %(message)s'
        
        super().__init__(fmt, datefmt='%H:%M:%S')
        self.show_category = show_category
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record in compact form.
        
        Args:
            record: Log record to format
            
        Returns:
            Compact formatted log string
        """
        # Ensure category attribute exists
        if not hasattr(record, 'category'):
            record.category = 'GENERAL'
        
        return super().format(record)


class CategoryFilter(logging.Filter):
    """
    Filter logs by category.
    
    Allows filtering logs to include only specific categories
    or exclude certain categories.
    """
    
    def __init__(
        self,
        include_categories: Optional[list] = None,
        exclude_categories: Optional[list] = None
    ):
        """
        Initialize category filter.
        
        Args:
            include_categories: List of categories to include (None = all)
            exclude_categories: List of categories to exclude
        """
        super().__init__()
        self.include_categories = set(include_categories) if include_categories else None
        self.exclude_categories = set(exclude_categories) if exclude_categories else set()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record by category.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged, False otherwise
        """
        category = getattr(record, 'category', 'GENERAL')
        
        # Check exclude list
        if category in self.exclude_categories:
            return False
        
        # Check include list
        if self.include_categories is not None:
            return category in self.include_categories
        
        return True


class LevelFilter(logging.Filter):
    """
    Filter logs by level with min/max range.
    
    Allows filtering logs within a specific level range.
    """
    
    def __init__(
        self,
        min_level: int = logging.DEBUG,
        max_level: int = logging.CRITICAL
    ):
        """
        Initialize level filter.
        
        Args:
            min_level: Minimum log level to include
            max_level: Maximum log level to include
        """
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record by level.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged, False otherwise
        """
        return self.min_level <= record.levelno <= self.max_level
