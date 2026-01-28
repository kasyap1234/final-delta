"""
Custom log handlers for the cryptocurrency trading bot.

This module provides custom logging handlers including:
- Database handler for writing logs to SQLite
- Rotating file handler with time-based rotation
- Colored console handler
- Error file handler for separate error logs
"""

import logging
import logging.handlers
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional

from .log_formatter import JsonFormatter


class DatabaseLogHandler(logging.Handler):
    """
    Log handler that writes logs to SQLite database.
    
    Stores structured log data in the system_logs table for
    efficient querying and analysis.
    
    Example:
        handler = DatabaseLogHandler('data/trading_bot.db')
        logger.addHandler(handler)
    """
    
    def __init__(self, db_path: str = 'data/trading_bot.db'):
        """
        Initialize database log handler.
        
        Args:
            db_path: Path to SQLite database file
        """
        super().__init__()
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure the database directory exists."""
        directory = os.path.dirname(self.db_path)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit log record to database.
        
        Args:
            record: Log record to emit
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Extract data fields
            data = self._extract_data(record)
            
            # Generate log ID
            log_id = f"log_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{record.thread}"
            
            cursor.execute("""
                INSERT INTO system_logs (
                    log_id, level, component, message, details, traceback, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id,
                record.levelname,
                record.name,
                record.getMessage(),
                data,
                self._format_exception(record),
                datetime.utcnow().isoformat()
            ))
            
            conn.commit()
            
        except Exception:
            self.handleError(record)
    
    def _extract_data(self, record: logging.LogRecord) -> Optional[str]:
        """
        Extract extra data fields from log record.
        
        Args:
            record: Log record
            
        Returns:
            JSON string of extra data or None
        """
        standard_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
            'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'correlation_id', 'category'
        }
        
        data = {}
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id') and record.correlation_id:
            data['correlation_id'] = record.correlation_id
        
        # Add category if present
        if hasattr(record, 'category') and record.category:
            data['category'] = record.category
        
        # Add source location
        data['source'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                data[key] = value
        
        if data:
            import json
            return json.dumps(data, default=str)
        return None
    
    def _format_exception(self, record: logging.LogRecord) -> Optional[str]:
        """
        Format exception information.
        
        Args:
            record: Log record
            
        Returns:
            Formatted exception string or None
        """
        if record.exc_info:
            return self.formatException(record.exc_info)
        return None
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
        super().close()


class AsyncDatabaseLogHandler(logging.Handler):
    """
    Asynchronous database log handler using a background thread.
    
    Provides non-blocking log writes by queuing records and
    processing them in a background thread.
    
    Example:
        handler = AsyncDatabaseLogHandler('data/trading_bot.db')
        logger.addHandler(handler)
    """
    
    def __init__(
        self,
        db_path: str = 'data/trading_bot.db',
        max_queue_size: int = 10000,
        flush_interval: float = 5.0
    ):
        """
        Initialize async database log handler.
        
        Args:
            db_path: Path to SQLite database file
            max_queue_size: Maximum queue size before blocking
            flush_interval: Interval in seconds to flush queue
        """
        super().__init__()
        self.db_path = db_path
        self.queue: Queue = Queue(maxsize=max_queue_size)
        self.flush_interval = flush_interval
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        self._start_worker()
    
    def _start_worker(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._worker_thread = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread.start()
    
    def _worker(self) -> None:
        """Background worker thread that processes log queue."""
        # Ensure directory exists
        directory = os.path.dirname(self.db_path)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        
        batch = []
        last_flush = datetime.utcnow()
        
        while not self._stop_event.is_set():
            try:
                # Try to get record with timeout
                record = self.queue.get(timeout=0.1)
                
                if record is None:  # Shutdown signal
                    break
                
                batch.append(record)
                
                # Flush if batch is full or interval passed
                now = datetime.utcnow()
                if (len(batch) >= 100 or 
                    (now - last_flush).total_seconds() >= self.flush_interval):
                    self._flush_batch(conn, batch)
                    batch = []
                    last_flush = now
                    
            except Exception:
                continue
        
        # Flush remaining records
        if batch:
            self._flush_batch(conn, batch)
        
        conn.close()
    
    def _flush_batch(self, conn: sqlite3.Connection, records: list) -> None:
        """
        Flush batch of records to database.
        
        Args:
            conn: Database connection
            records: List of log records to flush
        """
        try:
            cursor = conn.cursor()
            
            for record in records:
                data = self._extract_data(record)
                log_id = f"log_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{record.thread}"
                
                cursor.execute("""
                    INSERT INTO system_logs (
                        log_id, level, component, message, details, traceback, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id,
                    record.levelname,
                    record.name,
                    record.getMessage(),
                    data,
                    self._format_exception(record),
                    datetime.utcnow().isoformat()
                ))
            
            conn.commit()
            
        except Exception as e:
            # Log error to stderr
            import sys
            print(f"Error flushing logs: {e}", file=sys.stderr)
    
    def _extract_data(self, record: logging.LogRecord) -> Optional[str]:
        """Extract extra data fields from log record."""
        standard_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
            'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'correlation_id', 'category'
        }
        
        data = {}
        
        if hasattr(record, 'correlation_id') and record.correlation_id:
            data['correlation_id'] = record.correlation_id
        
        if hasattr(record, 'category') and record.category:
            data['category'] = record.category
        
        data['source'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName
        }
        
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                data[key] = value
        
        if data:
            import json
            return json.dumps(data, default=str)
        return None
    
    def _format_exception(self, record: logging.LogRecord) -> Optional[str]:
        """Format exception information."""
        if record.exc_info:
            return self.formatException(record.exc_info)
        return None
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Queue log record for async processing.
        
        Args:
            record: Log record to emit
        """
        try:
            self.queue.put_nowait(record)
        except Exception:
            self.handleError(record)
    
    def close(self) -> None:
        """Close handler and stop worker thread."""
        self._stop_event.set()
        self.queue.put(None)  # Signal shutdown
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10.0)
        
        super().close()


class TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Timed rotating file handler with custom naming.
    
    Extends standard TimedRotatingFileHandler to provide
    more flexible log rotation options.
    
    Example:
        handler = TimedRotatingFileHandler(
            'logs/trading_bot.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
    """
    
    def __init__(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        backupCount: int = 30,
        encoding: Optional[str] = 'utf-8',
        delay: bool = False,
        utc: bool = True,
        atTime: Optional[datetime] = None
    ):
        """
        Initialize timed rotating file handler.
        
        Args:
            filename: Log file path
            when: Rotation time unit ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight')
            interval: Rotation interval
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening
            utc: Use UTC time
            atTime: Specific time for rotation
        """
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=atTime
        )


class SizeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Size-based rotating file handler.
    
    Rotates log files when they reach a specified size.
    
    Example:
        handler = SizeRotatingFileHandler(
            'logs/trading_bot.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 10*1024*1024,  # 10MB default
        backupCount: int = 10,
        encoding: Optional[str] = 'utf-8',
        delay: bool = False
    ):
        """
        Initialize size rotating file handler.
        
        Args:
            filename: Log file path
            mode: File mode
            maxBytes: Maximum file size before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening
        """
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            filename=filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )


class ColoredConsoleHandler(logging.StreamHandler):
    """
    Console handler with colored output.
    
    Automatically applies colored formatting to log output
    when writing to a TTY.
    
    Example:
        handler = ColoredConsoleHandler()
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    """
    
    def __init__(self, stream=None):
        """
        Initialize colored console handler.
        
        Args:
            stream: Output stream (defaults to sys.stderr)
        """
        super().__init__(stream)
        self._is_tty = hasattr(self.stream, 'isatty') and self.stream.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        if self.formatter:
            return self.formatter.format(record)
        return super().format(record)


class ErrorFileHandler(logging.FileHandler):
    """
    File handler specifically for error-level logs.
    
    Only logs messages at ERROR level and above.
    Useful for maintaining a separate error log file.
    
    Example:
        handler = ErrorFileHandler('logs/errors.log')
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        encoding: Optional[str] = 'utf-8',
        delay: bool = False
    ):
        """
        Initialize error file handler.
        
        Args:
            filename: Log file path
            mode: File mode
            encoding: File encoding
            delay: Delay file opening
        """
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, mode, encoding, delay)
        self.setLevel(logging.ERROR)


class CategoryFileHandler(logging.Handler):
    """
    Handler that routes logs to different files based on category.
    
    Maintains separate log files for different log categories.
    
    Example:
        handler = CategoryFileHandler('logs/', default_level=logging.INFO)
        handler.add_category('TRADING', 'logs/trading.log', logging.DEBUG)
        logger.addHandler(handler)
    """
    
    def __init__(
        self,
        base_dir: str = 'logs/',
        default_filename: str = 'general.log',
        default_level: int = logging.DEBUG
    ):
        """
        Initialize category file handler.
        
        Args:
            base_dir: Base directory for log files
            default_filename: Default log file for uncategorized logs
            default_level: Default logging level
        """
        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.handlers: Dict[str, logging.FileHandler] = {}
        self.category_levels: Dict[str, int] = {}
        
        # Create default handler
        default_path = self.base_dir / default_filename
        self.default_handler = logging.FileHandler(default_path)
        self.default_handler.setLevel(default_level)
        self.default_handler.setFormatter(JsonFormatter())
    
    def add_category(
        self,
        category: str,
        filename: str,
        level: int = logging.DEBUG
    ) -> None:
        """
        Add a category-specific log file.
        
        Args:
            category: Log category name
            filename: Log file name (relative to base_dir)
            level: Logging level for this category
        """
        filepath = self.base_dir / filename
        handler = logging.FileHandler(filepath)
        handler.setLevel(level)
        handler.setFormatter(JsonFormatter())
        
        self.handlers[category] = handler
        self.category_levels[category] = level
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit log record to appropriate file based on category.
        
        Args:
            record: Log record to emit
        """
        category = getattr(record, 'category', 'GENERAL')
        
        # Get handler for category
        handler = self.handlers.get(category, self.default_handler)
        
        # Check level
        level = self.category_levels.get(category, self.default_handler.level)
        
        if record.levelno >= level:
            handler.emit(record)
    
    def close(self) -> None:
        """Close all handlers."""
        for handler in self.handlers.values():
            handler.close()
        self.default_handler.close()
        super().close()


class MemoryHandler(logging.handlers.MemoryHandler):
    """
    Memory buffer handler that flushes on specific conditions.
    
    Buffers log records in memory and flushes when:
    - Buffer capacity is reached
    - A record at or above flushLevel is seen
    - Explicit flush is called
    
    Example:
        target = DatabaseLogHandler('data/trading_bot.db')
        handler = MemoryHandler(capacity=100, flushLevel=logging.ERROR, target=target)
        logger.addHandler(handler)
    """
    
    def __init__(
        self,
        capacity: int = 100,
        flushLevel: int = logging.ERROR,
        target: Optional[logging.Handler] = None,
        flushOnClose: bool = True
    ):
        """
        Initialize memory handler.
        
        Args:
            capacity: Buffer capacity
            flushLevel: Level that triggers automatic flush
            target: Target handler to flush to
            flushOnClose: Flush buffer when handler is closed
        """
        super().__init__(capacity, flushLevel, target, flushOnClose)
