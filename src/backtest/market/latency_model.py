"""
Latency model module for realistic network and exchange delay simulation.

This module provides latency modeling for backtesting, simulating:
- Network latency for order submission, cancellation, and fills
- Exchange processing delays
- Variable latency based on market conditions
- WebSocket message delays
- Multiple latency distributions (normal, log-normal, pareto)
"""

import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class LatencyDistribution(str, Enum):
    """Latency distribution types."""
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    PARETO = "pareto"
    UNIFORM = "uniform"
    FIXED = "fixed"


class LatencyType(str, Enum):
    """Types of latency events."""
    ORDER_SUBMIT = "order_submit"
    ORDER_CANCEL = "order_cancel"
    ORDER_FILL = "order_fill"
    WEBSOCKET_MESSAGE = "websocket_message"
    EXCHANGE_PROCESSING = "exchange_processing"


@dataclass
class LatencyConfig:
    """Configuration for latency model."""
    # Base latencies in milliseconds
    base_order_submit_ms: float = 50.0
    base_order_cancel_ms: float = 30.0
    base_order_fill_ms: float = 100.0
    base_websocket_ms: float = 20.0
    base_exchange_processing_ms: float = 10.0
    
    # Distribution settings
    distribution: LatencyDistribution = LatencyDistribution.LOG_NORMAL
    
    # Distribution parameters
    normal_std_ms: float = 10.0  # Standard deviation for normal distribution
    log_normal_sigma: float = 0.5  # Sigma for log-normal distribution
    pareto_alpha: float = 2.0  # Shape parameter for pareto distribution
    uniform_range_ms: float = 20.0  # Range for uniform distribution
    
    # Market condition factors
    volatility_factor: float = 1.0  # Multiplier during high volatility
    volume_factor: float = 0.8  # Multiplier during high volume
    congestion_threshold: float = 0.7  # Queue utilization threshold for congestion
    
    # Exchange load simulation
    enable_exchange_load: bool = True
    max_exchange_queue_size: int = 1000
    processing_rate_per_sec: int = 500
    
    # Network jitter
    enable_jitter: bool = True
    jitter_std_ms: float = 5.0
    
    # Regional latency (for different exchange locations)
    regional_latency_ms: float = 0.0


@dataclass
class LatencyEvent:
    """Represents a latency event with timing information."""
    event_type: LatencyType
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_completed(self) -> bool:
        """Check if the event has completed."""
        return self.actual_time is not None
    
    @property
    def actual_latency_ms(self) -> float:
        """Get the actual latency in milliseconds."""
        if self.actual_time and self.scheduled_time:
            return (self.actual_time - self.scheduled_time).total_seconds() * 1000
        return 0.0


@dataclass
class ExchangeQueueState:
    """State of the exchange processing queue."""
    queue_size: int = 0
    processing_rate: int = 500
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def utilization(self) -> float:
        """Calculate queue utilization (0-1)."""
        if self.processing_rate <= 0:
            return 0.0
        return min(1.0, self.queue_size / self.processing_rate)
    
    @property
    def is_congested(self) -> bool:
        """Check if the queue is congested."""
        return self.utilization > 0.7


class LatencyModel:
    """
    Models realistic network and exchange latency for backtesting.
    
    This class simulates various types of delays that occur in real trading:
    - Network latency between client and exchange
    - Exchange processing delays
    - Variable latency based on market conditions
    - WebSocket message propagation delays
    
    Supports multiple latency distributions:
    - Normal: Symmetric distribution around mean
    - Log-normal: Skewed distribution with long tail (realistic for network latency)
    - Pareto: Heavy-tailed distribution for extreme latency events
    - Uniform: Even distribution within range
    - Fixed: Constant latency
    
    Example:
        ```python
        config = LatencyConfig(
            base_order_submit_ms=50.0,
            distribution=LatencyDistribution.LOG_NORMAL
        )
        model = LatencyModel(config)
        
        # Calculate latency for order submission
        latency = model.calculate_latency(LatencyType.ORDER_SUBMIT)
        
        # Schedule an event
        event = model.schedule_event(LatencyType.ORDER_FILL, datetime.now())
        ```
    """
    
    def __init__(self, config: Optional[LatencyConfig] = None):
        """
        Initialize the latency model.
        
        Args:
            config: Latency configuration. Uses defaults if not provided.
        """
        self.config = config or LatencyConfig()
        self._exchange_queue = ExchangeQueueState(
            processing_rate=self.config.processing_rate_per_sec
        )
        self._pending_events: List[LatencyEvent] = []
        self._event_callbacks: Dict[LatencyType, List[Callable[[LatencyEvent], None]]] = {
            lt: [] for lt in LatencyType
        }
        
        # Statistics tracking
        self._stats: Dict[str, Dict[str, float]] = {
            lt.value: {
                'count': 0,
                'total_latency_ms': 0.0,
                'min_latency_ms': float('inf'),
                'max_latency_ms': 0.0,
            }
            for lt in LatencyType
        }
        
        # Market condition state
        self._current_volatility: float = 0.02
        self._current_volume: float = 1.0
        
        logger.info(f"LatencyModel initialized with {self.config.distribution.value} distribution")
    
    def calculate_latency(
        self,
        latency_type: LatencyType,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate latency for a specific event type.
        
        Args:
            latency_type: Type of latency event
            market_conditions: Optional dict with 'volatility' and 'volume' keys
            
        Returns:
            Latency in milliseconds
        """
        # Get base latency for event type
        base_latency = self._get_base_latency(latency_type)
        
        # Apply distribution
        distributed_latency = self._apply_distribution(base_latency)
        
        # Apply market condition factors
        if market_conditions:
            distributed_latency = self._apply_market_conditions(
                distributed_latency, market_conditions
            )
        
        # Apply exchange queue delay
        if self.config.enable_exchange_load:
            distributed_latency += self._calculate_queue_delay()
        
        # Apply network jitter
        if self.config.enable_jitter:
            jitter = random.gauss(0, self.config.jitter_std_ms)
            distributed_latency += jitter
        
        # Apply regional latency
        distributed_latency += self.config.regional_latency_ms
        
        # Ensure non-negative
        return max(0.0, distributed_latency)
    
    def _get_base_latency(self, latency_type: LatencyType) -> float:
        """Get base latency for event type."""
        base_latencies = {
            LatencyType.ORDER_SUBMIT: self.config.base_order_submit_ms,
            LatencyType.ORDER_CANCEL: self.config.base_order_cancel_ms,
            LatencyType.ORDER_FILL: self.config.base_order_fill_ms,
            LatencyType.WEBSOCKET_MESSAGE: self.config.base_websocket_ms,
            LatencyType.EXCHANGE_PROCESSING: self.config.base_exchange_processing_ms,
        }
        return base_latencies.get(latency_type, 50.0)
    
    def _apply_distribution(self, base_latency: float) -> float:
        """Apply the configured distribution to base latency."""
        dist = self.config.distribution
        
        if dist == LatencyDistribution.FIXED:
            return base_latency
        
        elif dist == LatencyDistribution.NORMAL:
            return random.gauss(base_latency, self.config.normal_std_ms)
        
        elif dist == LatencyDistribution.LOG_NORMAL:
            # Log-normal: mean = log(base), sigma = config.sigma
            mu = math.log(max(1.0, base_latency))
            return random.lognormvariate(mu, self.config.log_normal_sigma)
        
        elif dist == LatencyDistribution.PARETO:
            # Pareto distribution for heavy tails
            # Scale to have mean close to base_latency
            scale = base_latency * (self.config.pareto_alpha - 1) / self.config.pareto_alpha
            if self.config.pareto_alpha > 1:
                return random.paretovariate(self.config.pareto_alpha) * scale
            return base_latency
        
        elif dist == LatencyDistribution.UNIFORM:
            half_range = self.config.uniform_range_ms / 2
            return random.uniform(
                max(0.0, base_latency - half_range),
                base_latency + half_range
            )
        
        return base_latency
    
    def _apply_market_conditions(
        self,
        latency: float,
        conditions: Dict[str, float]
    ) -> float:
        """Apply market condition multipliers to latency."""
        adjusted_latency = latency
        
        # Volatility factor
        volatility = conditions.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility threshold
            adjusted_latency *= self.config.volatility_factor
        
        # Volume factor
        volume = conditions.get('volume', 1.0)
        if volume > 2.0:  # High volume threshold
            adjusted_latency *= self.config.volume_factor
        
        return adjusted_latency
    
    def _calculate_queue_delay(self) -> float:
        """Calculate additional delay from exchange queue congestion."""
        utilization = self._exchange_queue.utilization
        
        if utilization < self.config.congestion_threshold:
            return 0.0
        
        # Exponential delay increase as queue fills
        congestion_factor = (utilization - self.config.congestion_threshold) / (1 - self.config.congestion_threshold)
        return congestion_factor * congestion_factor * 100.0  # Max 100ms additional delay
    
    def schedule_event(
        self,
        event_type: LatencyType,
        current_time: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LatencyEvent:
        """
        Schedule a latency event.
        
        Args:
            event_type: Type of latency event
            current_time: Current simulation time
            metadata: Optional metadata for the event
            
        Returns:
            LatencyEvent with scheduled completion time
        """
        latency_ms = self.calculate_latency(event_type)
        latency_delta = timedelta(milliseconds=latency_ms)
        
        event = LatencyEvent(
            event_type=event_type,
            scheduled_time=current_time + latency_delta,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        
        self._pending_events.append(event)
        
        # Update statistics
        self._update_stats(event_type, latency_ms)
        
        return event
    
    def process_events(self, current_time: datetime) -> List[LatencyEvent]:
        """
        Process events that should complete by current_time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of completed events
        """
        completed = []
        remaining = []
        
        for event in self._pending_events:
            if event.scheduled_time <= current_time:
                event.actual_time = current_time
                completed.append(event)
                self._trigger_callbacks(event)
            else:
                remaining.append(event)
        
        self._pending_events = remaining
        return completed
    
    def update_exchange_queue(self, queue_size: int, current_time: datetime) -> None:
        """
        Update the exchange queue state.
        
        Args:
            queue_size: Current number of orders in queue
            current_time: Current time for tracking
        """
        self._exchange_queue.queue_size = queue_size
        self._exchange_queue.last_update = current_time
    
    def simulate_order_arrival(self) -> None:
        """Simulate a new order arriving at the exchange queue."""
        self._exchange_queue.queue_size = min(
            self._exchange_queue.queue_size + 1,
            self.config.max_exchange_queue_size
        )
    
    def simulate_order_processing(self, count: int = 1) -> None:
        """Simulate orders being processed from the queue."""
        self._exchange_queue.queue_size = max(
            0,
            self._exchange_queue.queue_size - count
        )
    
    def set_market_conditions(
        self,
        volatility: Optional[float] = None,
        volume: Optional[float] = None
    ) -> None:
        """
        Update current market conditions for latency calculations.
        
        Args:
            volatility: Current market volatility (0-1)
            volume: Current volume relative to average (1.0 = average)
        """
        if volatility is not None:
            self._current_volatility = volatility
        if volume is not None:
            self._current_volume = volume
    
    def on_event(
        self,
        event_type: LatencyType,
        callback: Callable[[LatencyEvent], None]
    ) -> None:
        """
        Register a callback for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event completes
        """
        self._event_callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event: LatencyEvent) -> None:
        """Trigger callbacks for a completed event."""
        for callback in self._event_callbacks.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in latency callback: {e}")
    
    def _update_stats(self, event_type: LatencyType, latency_ms: float) -> None:
        """Update latency statistics."""
        stats = self._stats[event_type.value]
        stats['count'] += 1
        stats['total_latency_ms'] += latency_ms
        stats['min_latency_ms'] = min(stats['min_latency_ms'], latency_ms)
        stats['max_latency_ms'] = max(stats['max_latency_ms'], latency_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get latency statistics.
        
        Returns:
            Dictionary with latency statistics by event type
        """
        result = {}
        for event_type, stats in self._stats.items():
            count = stats['count']
            result[event_type] = {
                'count': count,
                'avg_latency_ms': stats['total_latency_ms'] / count if count > 0 else 0.0,
                'min_latency_ms': stats['min_latency_ms'] if count > 0 else 0.0,
                'max_latency_ms': stats['max_latency_ms'] if count > 0 else 0.0,
            }
        
        result['pending_events'] = len(self._pending_events)
        result['exchange_queue'] = {
            'size': self._exchange_queue.queue_size,
            'utilization': self._exchange_queue.utilization,
            'is_congested': self._exchange_queue.is_congested,
        }
        
        return result
    
    def reset(self) -> None:
        """Reset the latency model state."""
        self._pending_events.clear()
        self._exchange_queue = ExchangeQueueState(
            processing_rate=self.config.processing_rate_per_sec
        )
        
        for stats in self._stats.values():
            stats['count'] = 0
            stats['total_latency_ms'] = 0.0
            stats['min_latency_ms'] = float('inf')
            stats['max_latency_ms'] = 0.0
        
        self._current_volatility = 0.02
        self._current_volume = 1.0
        
        logger.info("LatencyModel reset")
    
    def get_pending_events(self) -> List[LatencyEvent]:
        """Get all pending latency events."""
        return list(self._pending_events)
    
    def get_next_event_time(self) -> Optional[datetime]:
        """Get the scheduled time of the next pending event."""
        if not self._pending_events:
            return None
        return min(e.scheduled_time for e in self._pending_events)


class VariableLatencyModel(LatencyModel):
    """
    Extended latency model with time-varying latency based on market hours.
    
    Simulates different latency characteristics during:
    - Peak trading hours
    - Off-peak hours
    - Market open/close
    - High volatility periods
    """
    
    def __init__(
        self,
        config: Optional[LatencyConfig] = None,
        peak_hours: Optional[List[int]] = None
    ):
        """
        Initialize variable latency model.
        
        Args:
            config: Latency configuration
            peak_hours: List of hours (0-23) considered peak trading time
        """
        super().__init__(config)
        self.peak_hours = peak_hours or [9, 10, 11, 14, 15, 16]
        self._peak_multiplier = 1.2  # Higher latency during peak
        self._off_peak_multiplier = 0.8  # Lower latency off-peak
    
    def calculate_latency(
        self,
        latency_type: LatencyType,
        market_conditions: Optional[Dict[str, float]] = None,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate latency with time-of-day adjustments.
        
        Args:
            latency_type: Type of latency event
            market_conditions: Optional market conditions
            current_time: Optional current time for time-of-day adjustment
            
        Returns:
            Latency in milliseconds
        """
        base_latency = super().calculate_latency(latency_type, market_conditions)
        
        if current_time:
            hour = current_time.hour
            if hour in self.peak_hours:
                base_latency *= self._peak_multiplier
            else:
                base_latency *= self._off_peak_multiplier
        
        return base_latency
