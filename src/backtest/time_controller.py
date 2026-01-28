"""
Time controller module for backtesting.

This module provides time control functionality for backtesting,
allowing the backtester to control time progression through historical data.

Supports event-based timing with latency modeling and processing delays.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of time-based events."""
    CANDLE = "candle"
    ORDER_SUBMIT = "order_submit"
    ORDER_FILL = "order_fill"
    ORDER_CANCEL = "order_cancel"
    WEBSOCKET_UPDATE = "websocket_update"
    PROCESSING_DELAY = "processing_delay"
    CUSTOM = "custom"


@dataclass
class TimeEvent:
    """Represents a time-based event in the simulation."""
    timestamp: datetime
    event_type: EventType
    priority: int = 0  # Lower = higher priority
    data: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable[[], None]] = None
    
    def __lt__(self, other: 'TimeEvent') -> bool:
        """Compare events for priority queue ordering."""
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class TimeControllerConfig:
    """Configuration for time controller."""
    # Processing delays between candles (in seconds)
    inter_candle_delay_ms: float = 0.0
    
    # Enable event-based processing
    enable_event_processing: bool = True
    
    # Maximum events to process per step
    max_events_per_step: int = 100
    
    # Track simulated vs actual time
    track_time_drift: bool = True
    
    # Latency factor for event delays
    latency_factor: float = 1.0  # 1.0 = real-time, <1.0 = faster, >1.0 = slower
    
    # WebSocket update interval (in milliseconds)
    websocket_update_interval_ms: float = 100.0
    
    # Enable sub-candle processing
    enable_sub_candle_events: bool = False
    sub_candle_steps: int = 4  # Number of steps between candles


class TimeController:
    """
    Controls time progression during backtesting with event-based timing.
    
    The backtester iterates through historical candles chronologically,
    advancing time to each candle's timestamp and executing the trading
    logic at that point in time.
    
    Supports event-based processing with:
    - Latency modeling for realistic timing
    - Processing delays between candles
    - Sub-candle events for high-frequency simulation
    - Simulated vs actual time tracking
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
        timeframe: str = "15m",
        config: Optional[TimeControllerConfig] = None
    ):
        """
        Initialize the time controller.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Candle timeframe (e.g., '15m', '1h', '1d')
            config: Time controller configuration
            
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
        
        # Configuration
        self.config = config or TimeControllerConfig()
        
        # Event queue for event-based processing
        self._event_queue: List[TimeEvent] = []
        self._event_callbacks: Dict[EventType, List[Callable[[TimeEvent], None]]] = {
            et: [] for et in EventType
        }
        
        # Time tracking
        self._actual_start_time: Optional[datetime] = None
        self._simulated_time_elapsed: timedelta = timedelta()
        self._actual_time_elapsed: timedelta = timedelta()
        
        # Processing state
        self._last_candle_time: Optional[datetime] = None
        self._sub_candle_step = 0
        
        logger.info(
            f"TimeController initialized: {start_date} to {end_date}, "
            f"timeframe={timeframe}, interval={self._candle_interval}m, "
            f"event_processing={self.config.enable_event_processing}"
        )
    
    def advance_to_next_candle(self) -> datetime:
        """
        Advance time to the next candle.
        
        Returns:
            The new current time
        """
        # Add inter-candle processing delay
        if self.config.inter_candle_delay_ms > 0:
            delay = timedelta(milliseconds=self.config.inter_candle_delay_ms)
            self._current_time += delay
            self._simulated_time_elapsed += delay
        
        self._current_time += timedelta(minutes=self._candle_interval)
        self._candle_count += 1
        self._last_candle_time = self._current_time
        self._sub_candle_step = 0
        
        # Don't exceed end date
        if self._current_time > self._end_date:
            self._current_time = self._end_date
        
        # Schedule candle event if event processing enabled
        if self.config.enable_event_processing:
            self._schedule_event(
                TimeEvent(
                    timestamp=self._current_time,
                    event_type=EventType.CANDLE,
                    priority=1,
                    data={'candle_number': self._candle_count}
                )
            )
        
        return self._current_time
    
    def advance_with_latency(self, latency_ms: float) -> datetime:
        """
        Advance time by a specific latency amount.
        
        Args:
            latency_ms: Latency in milliseconds
            
        Returns:
            The new current time
        """
        adjusted_latency = latency_ms * self.config.latency_factor
        delay = timedelta(milliseconds=adjusted_latency)
        self._current_time += delay
        self._simulated_time_elapsed += delay
        
        # Process any events that should occur during this time
        if self.config.enable_event_processing:
            self._process_events_up_to(self._current_time)
        
        return self._current_time
    
    def _schedule_event(self, event: TimeEvent) -> None:
        """Add an event to the priority queue."""
        heapq.heappush(self._event_queue, event)
    
    def schedule_delayed_event(
        self,
        delay_ms: float,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[], None]] = None,
        priority: int = 5
    ) -> TimeEvent:
        """
        Schedule an event to occur after a delay.
        
        Args:
            delay_ms: Delay in milliseconds
            event_type: Type of event
            data: Event data
            callback: Optional callback to execute
            priority: Event priority (lower = higher priority)
            
        Returns:
            The scheduled event
        """
        adjusted_delay = delay_ms * self.config.latency_factor
        event_time = self._current_time + timedelta(milliseconds=adjusted_delay)
        
        event = TimeEvent(
            timestamp=event_time,
            event_type=event_type,
            priority=priority,
            data=data or {},
            callback=callback
        )
        
        self._schedule_event(event)
        return event
    
    def _process_events_up_to(self, target_time: datetime) -> List[TimeEvent]:
        """
        Process all events scheduled up to target_time.
        
        Args:
            target_time: Process events up to this time
            
        Returns:
            List of processed events
        """
        processed = []
        event_count = 0
        
        while (self._event_queue and 
               self._event_queue[0].timestamp <= target_time and
               event_count < self.config.max_events_per_step):
            
            event = heapq.heappop(self._event_queue)
            
            # Execute callback if provided
            if event.callback:
                try:
                    event.callback()
                except Exception as e:
                    logger.error(f"Error executing event callback: {e}")
            
            # Trigger registered callbacks
            for callback in self._event_callbacks.get(event.event_type, []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
            
            processed.append(event)
            event_count += 1
        
        return processed
    
    def on_event(
        self,
        event_type: EventType,
        callback: Callable[[TimeEvent], None]
    ) -> None:
        """
        Register a callback for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        self._event_callbacks[event_type].append(callback)
    
    def get_next_event_time(self) -> Optional[datetime]:
        """Get the timestamp of the next scheduled event."""
        if self._event_queue:
            return self._event_queue[0].timestamp
        return None
    
    def get_pending_events(self) -> List[TimeEvent]:
        """Get all pending events sorted by time."""
        return sorted(self._event_queue)
    
    def process_sub_candle_step(self) -> Optional[datetime]:
        """
        Process a sub-candle step for high-frequency simulation.
        
        Returns:
            Current time after step, or None if sub-candle processing disabled
        """
        if not self.config.enable_sub_candle_events:
            return None
        
        if self._sub_candle_step >= self.config.sub_candle_steps:
            return None
        
        step_duration_ms = (self._candle_interval * 60 * 1000) / self.config.sub_candle_steps
        self._sub_candle_step += 1
        
        step_time = self._last_candle_time + timedelta(milliseconds=step_duration_ms * self._sub_candle_step)
        
        # Process events up to this sub-step
        if self.config.enable_event_processing:
            self._process_events_up_to(step_time)
        
        return step_time
    
    def start_timing(self) -> None:
        """Start tracking actual execution time."""
        self._actual_start_time = datetime.now()
    
    def get_time_drift(self) -> Optional[timedelta]:
        """
        Get the difference between simulated and actual time.
        
        Returns:
            Time drift (positive = simulation faster than real-time)
        """
        if not self.config.track_time_drift or not self._actual_start_time:
            return None
        
        actual_elapsed = datetime.now() - self._actual_start_time
        return self._simulated_time_elapsed - actual_elapsed
    
    def get_timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        stats = {
            'simulated_time_elapsed_seconds': self._simulated_time_elapsed.total_seconds(),
            'current_time': self._current_time.isoformat(),
            'candle_count': self._candle_count,
            'pending_events': len(self._event_queue),
        }
        
        if self.config.track_time_drift and self._actual_start_time:
            actual_elapsed = datetime.now() - self._actual_start_time
            stats['actual_time_elapsed_seconds'] = actual_elapsed.total_seconds()
            stats['time_drift_seconds'] = (
                self._simulated_time_elapsed - actual_elapsed
            ).total_seconds()
            stats['speed_ratio'] = (
                self._simulated_time_elapsed.total_seconds() / actual_elapsed.total_seconds()
                if actual_elapsed.total_seconds() > 0 else 0
            )
        
        return stats
    
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
        self._last_candle_time = None
        self._sub_candle_step = 0
        self._simulated_time_elapsed = timedelta()
        self._actual_time_elapsed = timedelta()
        self._actual_start_time = None
        
        # Clear event queue
        self._event_queue.clear()
        
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
