"""
Time controller module for backtesting.

This module provides time control functionality for backtesting,
allowing the backtester to control time progression through historical data.
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TimeController:
    """
    Controls time progression during backtesting.
    
    The backtester iterates through historical candles chronologically,
    advancing time to each candle's timestamp and executing the trading
    logic at that point in time.
    """
    
    # Timeframe to minutes mapping
    TIMEFRAME_MINUTES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080,
    }
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "15m"
    ):
        """
        Initialize the time controller.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Candle timeframe (e.g., '15m', '1h', '1d')
            
        Raises:
            ValueError: If timeframe is not supported or dates are invalid
        """
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        if timeframe not in self.TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        self._start_date = start_date
        self._end_date = end_date
        self._timeframe = timeframe
        self._candle_interval = self.TIMEFRAME_MINUTES[timeframe]
        self._current_time = start_date
        self._candle_count = 0
        
        logger.info(
            f"TimeController initialized: {start_date} to {end_date}, "
            f"timeframe={timeframe}, interval={self._candle_interval}m"
        )
    
    def advance_to_next_candle(self) -> datetime:
        """
        Advance time to the next candle.
        
        Returns:
            The new current time
        """
        self._current_time += timedelta(minutes=self._candle_interval)
        self._candle_count += 1
        
        # Don't exceed end date
        if self._current_time > self._end_date:
            self._current_time = self._end_date
        
        return self._current_time
    
    def advance_to_time(self, target_time: datetime) -> datetime:
        """
        Advance time to a specific target time.
        
        Args:
            target_time: Target time to advance to
            
        Returns:
            The new current time
        """
        # Normalize both times to timezone-naive for comparison
        target_naive = target_time.replace(tzinfo=None) if target_time.tzinfo else target_time
        current_naive = self._current_time.replace(tzinfo=None) if self._current_time.tzinfo else self._current_time
        end_naive = self._end_date.replace(tzinfo=None) if self._end_date.tzinfo else self._end_date
        
        if target_naive < current_naive:
            logger.warning(
                f"Target time {target_time} is before current time {self._current_time}"
            )
            return self._current_time
        
        if target_naive > end_naive:
            target_time = self._end_date
        
        self._current_time = target_time
        return self._current_time
    
    def get_current_time(self) -> datetime:
        """
        Get the current simulation time.
        
        Returns:
            Current simulation time
        """
        return self._current_time
    
    def get_start_time(self) -> datetime:
        """
        Get the start time of the backtest.
        
        Returns:
            Start time
        """
        return self._start_date
    
    def get_end_time(self) -> datetime:
        """
        Get the end time of the backtest.
        
        Returns:
            End time
        """
        return self._end_date
    
    def is_complete(self) -> bool:
        """
        Check if backtest is complete.
        
        Returns:
            True if current time >= end time
        """
        return self._current_time >= self._end_date
    
    def get_candle_count(self) -> int:
        """
        Get the number of candles processed so far.
        
        Returns:
            Number of candles processed
        """
        return self._candle_count
    
    def get_timeframe(self) -> str:
        """
        Get the timeframe.
        
        Returns:
            Timeframe string
        """
        return self._timeframe
    
    def get_candle_interval_minutes(self) -> int:
        """
        Get the candle interval in minutes.
        
        Returns:
            Candle interval in minutes
        """
        return self._candle_interval
    
    def reset(self) -> None:
        """Reset the time controller to the start date."""
        self._current_time = self._start_date
        self._candle_count = 0
        logger.info("TimeController reset to start time")
    
    def get_progress(self) -> float:
        """
        Get the progress of the backtest as a percentage.
        
        Returns:
            Progress percentage (0.0 to 1.0)
        """
        # Normalize all datetimes to timezone-naive for comparison
        end_naive = self._end_date.replace(tzinfo=None) if self._end_date.tzinfo else self._end_date
        start_naive = self._start_date.replace(tzinfo=None) if self._start_date.tzinfo else self._start_date
        current_naive = self._current_time.replace(tzinfo=None) if self._current_time.tzinfo else self._current_time
        
        total_duration = (end_naive - start_naive).total_seconds()
        elapsed = (current_naive - start_naive).total_seconds()
        
        if total_duration == 0:
            return 1.0
        
        return min(elapsed / total_duration, 1.0)
    
    def get_remaining_candles(self) -> int:
        """
        Get the estimated number of remaining candles.
        
        Returns:
            Number of remaining candles
        """
        # Normalize all datetimes to timezone-naive for comparison
        end_naive = self._end_date.replace(tzinfo=None) if self._end_date.tzinfo else self._end_date
        current_naive = self._current_time.replace(tzinfo=None) if self._current_time.tzinfo else self._current_time
        
        remaining_seconds = (end_naive - current_naive).total_seconds()
        remaining_minutes = remaining_seconds / 60
        return int(remaining_minutes / self._candle_interval)
    
    def __repr__(self) -> str:
        """String representation of TimeController."""
        return (
            f"TimeController(current={self._current_time}, "
            f"start={self._start_date}, end={self._end_date}, "
            f"timeframe={self._timeframe}, candles={self._candle_count})"
        )
