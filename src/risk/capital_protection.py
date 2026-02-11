"""
Capital Protection Module - Phase 4: Benchmark-Aware Deployment Logic

Implements:
1. Benchmark-aware deployment state machine (ACTIVE/PASSIVE/DEFENSIVE)
2. Capital protection controls (daily loss stop, consecutive-loss throttle, regime kill switch)
3. Benchmark-relative guardrails

This module ensures the strategy protects capital during adverse conditions
while maintaining parity between backtest and live trading.
"""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DeploymentState(Enum):
    """Deployment state for the strategy.

    ACTIVE: Normal trading with all gates active
    PASSIVE: Stricter entries only (higher edge floor, stricter ambiguity veto)
    DEFENSIVE: No new entries; manage exits only
    """

    ACTIVE = "active"
    PASSIVE = "passive"
    DEFENSIVE = "defensive"


# ─── V3 Operating Modes ────────────────────────────────────────────────────────

class V3OperatingMode(Enum):
    """V3 operating modes for the strategy.

    RISK_ON_BULL_CAPTURE: Capture persistent bull trends with long-biased active trading
    RISK_ON_SELECTIVE: Trade only high-conviction setups with symmetric but selective entries
    PASSIVE_DEFER: Avoid low-edge overtrading while preserving reactivation path
    RISK_OFF_NO_TRADE: Protect capital in adverse conditions with no new entries
    """

    RISK_ON_BULL_CAPTURE = "risk_on_bull_capture"
    RISK_ON_SELECTIVE = "risk_on_selective"
    PASSIVE_DEFER = "passive_defer"
    RISK_OFF_NO_TRADE = "risk_off_no_trade"


@dataclass
class DeploymentStateTransition:
    """Record of a deployment state transition."""

    from_state: DeploymentState
    to_state: DeploymentState
    timestamp: datetime
    reason: str
    drawdown: float
    underperformance_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "drawdown": self.drawdown,
            "underperformance_pct": self.underperformance_pct,
        }


@dataclass
class CapitalProtectionConfig:
    """Configuration for capital protection controls."""

    # Drawdown circuit breaker thresholds (from Phase 4 design)
    drawdown_d0_threshold: float = 0.04  # 4% - normal
    drawdown_d1_threshold: float = 0.06  # 6% - reduced risk
    drawdown_d2_threshold: float = 0.08  # 8% - trend/breakout only
    drawdown_d3_threshold: float = 0.10  # 10% - top-edge only
    drawdown_d4_threshold: float = 0.12  # 12% - blocked

    # Risk multipliers for each drawdown stage
    drawdown_d0_multiplier: float = 1.00
    drawdown_d1_multiplier: float = 0.75
    drawdown_d2_multiplier: float = 0.50
    drawdown_d3_multiplier: float = 0.25
    drawdown_d4_multiplier: float = 0.00

    # Daily loss stop
    daily_loss_cap_pct: float = 0.03  # 3% daily loss cap

    # Consecutive loss throttle
    max_consecutive_losses: int = 3
    consecutive_loss_cooldown_candles: int = 20

    # Regime-level kill switch
    regime_kill_switch_window: int = 20  # Number of trades to evaluate
    regime_kill_switch_min_expectancy: float = -0.05  # Minimum expectancy
    regime_kill_switch_min_win_rate_diff: float = 0.05  # 5% below break-even
    regime_kill_switch_consecutive_failures: int = 2  # Consecutive windows to trigger

    # Benchmark-relative guardrails
    benchmark_up_year_threshold: float = 0.30  # 30% benchmark return
    benchmark_min_capture_pct: float = 0.60  # 60% capture
    benchmark_max_underperformance_pct: float = 0.20  # 20% max underperformance
    benchmark_drawdown_excess_limit: float = 0.05  # 5% excess drawdown

    # Live degradation guardrail
    live_degradation_window_days: int = 60
    live_degradation_max_underperformance: float = 0.12  # 12%
    live_degradation_min_drawdown: float = 0.08  # 8%

    # ─── V3: Capture-First State Machine Parameters ───────────────────────────
    # Risk-on bull-capture activation thresholds
    v3_regime_confidence_min: float = 0.72  # Minimum regime confidence (0.68-0.80)
    v3_benchmark_trend_strength_min: float = 0.68  # Minimum benchmark trend strength (0.60-0.80)
    v3_drawdown_max_for_risk_on: float = 0.08  # Maximum drawdown for risk-on mode (8%)

    # V3 loss-streak controls
    v3_loss_streak_3_risk_mult: float = 0.70  # Risk multiplier after 3 consecutive losses
    v3_loss_streak_3_cooldown: int = 12  # Cooldown candles after 3 consecutive losses
    v3_loss_streak_5_risk_mult: float = 0.40  # Risk multiplier after 5 consecutive losses
    v3_loss_streak_5_cooldown: int = 24  # Cooldown candles after 5 consecutive losses
    v3_loss_streak_6_force_passive: bool = True  # Force PASSIVE_DEFER after 6 consecutive losses
    v3_loss_streak_6_cooldown: int = 32  # Cooldown candles after 6 consecutive losses

    # V3 re-entry hysteresis
    v3_recovery_windows_required: int = 2  # Number of consecutive positive windows required
    v3_hysteresis_buffer_pct: float = 0.01  # 1% hysteresis buffer for drawdown recovery

    # Hysteresis / cooldown for state transitions
    state_transition_cooldown_candles: int = 10
    state_transition_hysteresis_pct: float = 0.01  # 1% hysteresis

    # Passive mode stricter thresholds
    passive_edge_floor_multiplier: float = 1.5  # 1.5x edge floor
    passive_ambiguity_threshold_multiplier: float = 1.5  # 1.5x ambiguity threshold

    # Recovery window for regime kill switch reset
    regime_recovery_window: int = 10  # Number of trades to evaluate recovery

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drawdown_d0_threshold": self.drawdown_d0_threshold,
            "drawdown_d1_threshold": self.drawdown_d1_threshold,
            "drawdown_d2_threshold": self.drawdown_d2_threshold,
            "drawdown_d3_threshold": self.drawdown_d3_threshold,
            "drawdown_d4_threshold": self.drawdown_d4_threshold,
            "drawdown_d0_multiplier": self.drawdown_d0_multiplier,
            "drawdown_d1_multiplier": self.drawdown_d1_multiplier,
            "drawdown_d2_multiplier": self.drawdown_d2_multiplier,
            "drawdown_d3_multiplier": self.drawdown_d3_multiplier,
            "drawdown_d4_multiplier": self.drawdown_d4_multiplier,
            "daily_loss_cap_pct": self.daily_loss_cap_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "consecutive_loss_cooldown_candles": self.consecutive_loss_cooldown_candles,
            "regime_kill_switch_window": self.regime_kill_switch_window,
            "regime_kill_switch_min_expectancy": self.regime_kill_switch_min_expectancy,
            "regime_kill_switch_min_win_rate_diff": self.regime_kill_switch_min_win_rate_diff,
            "regime_kill_switch_consecutive_failures": self.regime_kill_switch_consecutive_failures,
            "benchmark_up_year_threshold": self.benchmark_up_year_threshold,
            "benchmark_min_capture_pct": self.benchmark_min_capture_pct,
            "benchmark_max_underperformance_pct": self.benchmark_max_underperformance_pct,
            "benchmark_drawdown_excess_limit": self.benchmark_drawdown_excess_limit,
            "live_degradation_window_days": self.live_degradation_window_days,
            "live_degradation_max_underperformance": self.live_degradation_max_underperformance,
            "live_degradation_min_drawdown": self.live_degradation_min_drawdown,
            "state_transition_cooldown_candles": self.state_transition_cooldown_candles,
            "state_transition_hysteresis_pct": self.state_transition_hysteresis_pct,
            "passive_edge_floor_multiplier": self.passive_edge_floor_multiplier,
            "passive_ambiguity_threshold_multiplier": self.passive_ambiguity_threshold_multiplier,
            "regime_recovery_window": self.regime_recovery_window,
        }


@dataclass
class RegimePerformanceMetrics:
    """Performance metrics for a specific regime."""

    regime: str
    trade_count: int
    win_count: int
    loss_count: int
    total_pnl_r: float  # Total PnL in R units
    expectancy_r: float  # Expectancy in R units
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    is_killed: bool = False
    kill_reason: Optional[str] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_pnl_r": self.total_pnl_r,
            "expectancy_r": self.expectancy_r,
            "win_rate": self.win_rate,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
            "is_killed": self.is_killed,
            "kill_reason": self.kill_reason,
            "consecutive_failures": self.consecutive_failures,
        }


class CapitalProtectionManager:
    """
    Manages capital protection and deployment state transitions.

    This class implements:
    1. Benchmark-aware deployment state machine
    2. Drawdown circuit breaker
    3. Daily loss stop
    4. Consecutive loss throttle
    5. Regime-level kill switch
    6. Benchmark-relative guardrails
    """

    def __init__(self, config: Optional[CapitalProtectionConfig] = None):
        """
        Initialize the capital protection manager.

        Args:
            config: Configuration for capital protection controls
        """
        self.config = config or CapitalProtectionConfig()

        # Current deployment state
        self._current_state = DeploymentState.ACTIVE
        self._state_history: List[DeploymentStateTransition] = []

        # State transition cooldown
        self._candles_since_last_transition = 0
        self._last_transition_time: Optional[datetime] = None

        # Drawdown tracking
        self._peak_balance: float = 0.0
        self._current_balance: float = 0.0
        self._initial_balance: float = 0.0

        # Daily loss tracking
        self._daily_pnl: float = 0.0
        self._current_day: Optional[str] = None

        # Consecutive loss tracking
        self._consecutive_losses: int = 0
        self._consecutive_loss_cooldown: int = 0

        # Regime performance tracking
        self._regime_metrics: Dict[str, RegimePerformanceMetrics] = {}
        self._regime_trade_buffer: Dict[str, List[Dict[str, Any]]] = {}

        # Benchmark tracking
        self._benchmark_prices: Dict[str, List[Tuple[datetime, float]]] = {}
        self._strategy_returns: List[Tuple[datetime, float]] = []

        # Trade history for performance calculation
        self._trade_history: List[Dict[str, Any]] = []

        logger.info("CapitalProtectionManager initialized with deployment state machine")

    # ─── Deployment State Management ───────────────────────────────────────────

    @property
    def current_state(self) -> DeploymentState:
        """Get the current deployment state."""
        return self._current_state

    def get_state(self) -> DeploymentState:
        """Get the current deployment state (alias for current_state)."""
        return self._current_state

    def _transition_state(
        self,
        new_state: DeploymentState,
        reason: str,
        drawdown: float,
        underperformance_pct: float = 0.0,
    ) -> None:
        """
        Transition to a new deployment state.

        Args:
            new_state: New deployment state
            reason: Reason for the transition
            drawdown: Current drawdown
            underperformance_pct: Current underperformance vs benchmark
        """
        if new_state == self._current_state:
            return

        transition = DeploymentStateTransition(
            from_state=self._current_state,
            to_state=new_state,
            timestamp=datetime.now(),
            reason=reason,
            drawdown=drawdown,
            underperformance_pct=underperformance_pct,
        )

        self._state_history.append(transition)
        self._current_state = new_state
        self._candles_since_last_transition = 0
        self._last_transition_time = datetime.now()

        logger.info(
            f"Deployment state transition: {transition.from_state.value} -> {transition.to_state.value} "
            f"(reason: {reason}, drawdown: {drawdown:.2%}, underperformance: {underperformance_pct:.2%})"
        )

    def evaluate_deployment_state(
        self,
        current_drawdown: float,
        active_edge: float,
        benchmark_trend_strength: float = 0.0,
        underperformance_pct: float = 0.0,
        regime_value: str = "unknown",
    ) -> DeploymentState:
        """
        Evaluate and update deployment state based on current conditions.

        State machine logic:
        - ACTIVE: rolling active edge above floor, risk breakers not active
        - PASSIVE: benchmark directional strength high, active edge below floor, underperformance rising
        - DEFENSIVE: active edge below floor and benchmark directional edge weak

        V2 Fix A2: Add hysteresis to prevent unnecessary transition to defensive state
        when drawdown is low and trend quality is strong. This addresses over-defensive
        behavior in healthy bull trends.

        Args:
            current_drawdown: Current drawdown as decimal (e.g., 0.05 for 5%)
            active_edge: Current active edge score
            benchmark_trend_strength: Benchmark trend strength (0-1)
            underperformance_pct: Underperformance vs benchmark as decimal
            regime_value: Current regime value for trend quality assessment

        Returns:
            Current deployment state
        """
        # Check cooldown before allowing state transition
        if self._candles_since_last_transition < self.config.state_transition_cooldown_candles:
            return self._current_state

        # Get thresholds from config
        edge_floor = 0.08  # Default edge floor
        benchmark_trend_threshold = 0.6  # Threshold for strong benchmark trend

        # Apply hysteresis to drawdown thresholds
        hysteresis = self.config.state_transition_hysteresis_pct

        # V2 Fix A2: Assess trend quality to prevent over-defensive behavior
        # In strong uptrend with low drawdown, stay ACTIVE even if edge is moderate
        trend_quality_strong = (
            regime_value == "trending_up" and
            benchmark_trend_strength >= 0.7 and
            current_drawdown < self.config.drawdown_d1_threshold  # Below 6%
        )

        # Determine target state based on conditions
        target_state = self._current_state

        if self._current_state == DeploymentState.ACTIVE:
            # Check if we should move to PASSIVE or DEFENSIVE
            if active_edge < edge_floor:
                # V2 Fix A2: In strong uptrend with low drawdown, stay ACTIVE
                # even if edge is slightly below floor (use relaxed threshold)
                if trend_quality_strong and active_edge >= edge_floor * 0.8:
                    # Stay ACTIVE with relaxed edge threshold
                    target_state = DeploymentState.ACTIVE
                elif benchmark_trend_strength > benchmark_trend_threshold:
                    target_state = DeploymentState.PASSIVE
                else:
                    target_state = DeploymentState.DEFENSIVE
            elif current_drawdown > self.config.drawdown_d1_threshold + hysteresis:
                target_state = DeploymentState.PASSIVE
            elif current_drawdown > self.config.drawdown_d3_threshold + hysteresis:
                target_state = DeploymentState.DEFENSIVE

        elif self._current_state == DeploymentState.PASSIVE:
            # Check if we should move to ACTIVE or DEFENSIVE
            # V2 Fix A2: In strong uptrend with low drawdown, return to ACTIVE more easily
            if trend_quality_strong and current_drawdown < self.config.drawdown_d0_threshold:
                target_state = DeploymentState.ACTIVE
            elif active_edge >= edge_floor and current_drawdown < self.config.drawdown_d0_threshold - hysteresis:
                target_state = DeploymentState.ACTIVE
            elif current_drawdown > self.config.drawdown_d3_threshold + hysteresis:
                target_state = DeploymentState.DEFENSIVE

        elif self._current_state == DeploymentState.DEFENSIVE:
            # Check if we should move to PASSIVE or ACTIVE
            # V2 Fix A2: In strong uptrend with low drawdown, return to ACTIVE more easily
            if trend_quality_strong and current_drawdown < self.config.drawdown_d1_threshold:
                target_state = DeploymentState.ACTIVE
            elif active_edge >= edge_floor and current_drawdown < self.config.drawdown_d2_threshold - hysteresis:
                target_state = DeploymentState.PASSIVE
            elif active_edge >= edge_floor and current_drawdown < self.config.drawdown_d0_threshold - hysteresis:
                target_state = DeploymentState.ACTIVE

        # Apply transition if state changed
        if target_state != self._current_state:
            reason = self._get_transition_reason(target_state, active_edge, benchmark_trend_strength, current_drawdown, trend_quality_strong)
            self._transition_state(target_state, reason, current_drawdown, underperformance_pct)

        return self._current_state

    def _get_transition_reason(
        self,
        target_state: DeploymentState,
        active_edge: float,
        benchmark_trend_strength: float,
        current_drawdown: float,
        trend_quality_strong: bool = False,
    ) -> str:
        """Generate a human-readable reason for state transition."""
        if target_state == DeploymentState.ACTIVE:
            if trend_quality_strong:
                return f"Strong uptrend with low drawdown {current_drawdown:.2%}, returning to ACTIVE"
            return f"Active edge {active_edge:.3f} above floor, drawdown {current_drawdown:.2%} below threshold"
        elif target_state == DeploymentState.PASSIVE:
            if active_edge < 0.08:
                return f"Active edge {active_edge:.3f} below floor, benchmark trend {benchmark_trend_strength:.2f} strong"
            else:
                return f"Drawdown {current_drawdown:.2%} above threshold, reducing risk"
        else:  # DEFENSIVE
            return f"Active edge {active_edge:.3f} below floor, drawdown {current_drawdown:.2%} high - blocking entries"

    def increment_candle_count(self) -> None:
        """Increment the candle count for cooldown tracking."""
        self._candles_since_last_transition += 1
        if self._consecutive_loss_cooldown > 0:
            self._consecutive_loss_cooldown -= 1

    # ─── V3 Operating Mode Evaluation ───────────────────────────────────────────

    def evaluate_v3_operating_mode(
        self,
        current_drawdown: float,
        active_edge: float,
        benchmark_trend_strength: float,
        regime_value: str = "unknown",
        regime_confidence: float = 0.0,
    ) -> V3OperatingMode:
        """
        Evaluate V3 operating mode based on current conditions.

        Implements the V3 capture-first state machine:
        - RISK_ON_BULL_CAPTURE: Strong uptrend with high confidence and low drawdown
        - RISK_ON_SELECTIVE: Trade only high-conviction setups
        - PASSIVE_DEFER: Avoid low-edge overtrading while preserving reactivation path
        - RISK_OFF_NO_TRADE: Protect capital in adverse conditions

        Args:
            current_drawdown: Current drawdown as decimal
            active_edge: Current active edge score
            benchmark_trend_strength: Benchmark trend strength (0-1)
            regime_value: Current regime value
            regime_confidence: Confidence in regime detection (0-1)

        Returns:
            V3OperatingMode based on conditions
        """
        # Check hard breakers first - force RISK_OFF_NO_TRADE
        if self.is_drawdown_circuit_breaker_active():
            return V3OperatingMode.RISK_OFF_NO_TRADE

        if self.is_daily_loss_cap_exceeded():
            return V3OperatingMode.RISK_OFF_NO_TRADE

        if self.is_regime_killed(regime_value):
            return V3OperatingMode.RISK_OFF_NO_TRADE

        # Check consecutive loss throttle - force PASSIVE_DEFER
        if self.is_consecutive_loss_throttle_active():
            return V3OperatingMode.PASSIVE_DEFER

        # Check for RISK_ON_BULL_CAPTURE conditions
        # All conditions must be met:
        # 1. Regime is trending_up
        # 2. Regime confidence >= minimum
        # 3. Benchmark trend strength >= minimum
        # 4. Drawdown <= maximum for risk-on
        # 5. No hard circuit breaker active (already checked above)
        if (
            regime_value == "trending_up"
            and regime_confidence >= self.config.v3_regime_confidence_min
            and benchmark_trend_strength >= self.config.v3_benchmark_trend_strength_min
            and current_drawdown <= self.config.v3_drawdown_max_for_risk_on
        ):
            return V3OperatingMode.RISK_ON_BULL_CAPTURE

        # Check for RISK_ON_SELECTIVE conditions
        # Trade only high-conviction setups when:
        # 1. Regime is trending_down, ranging, or volatile
        # 2. Active edge is above minimum threshold
        # 3. Drawdown is manageable
        if (
            regime_value in ("trending_down", "ranging", "volatile")
            and active_edge >= 0.10  # Minimum edge for selective mode
            and current_drawdown <= 0.10  # Manageable drawdown
        ):
            return V3OperatingMode.RISK_ON_SELECTIVE

        # Default to PASSIVE_DEFER for ambiguous/low-edge conditions
        return V3OperatingMode.PASSIVE_DEFER

    # ─── Drawdown Circuit Breaker ─────────────────────────────────────────────

    def update_balance(self, balance: float) -> None:
        """
        Update the current balance and track drawdown.

        Args:
            balance: Current account balance
        """
        if self._initial_balance == 0.0:
            self._initial_balance = balance
            self._peak_balance = balance

        self._current_balance = balance
        self._peak_balance = max(self._peak_balance, balance)

    def get_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak.

        Returns:
            Drawdown as decimal (e.g., 0.05 for 5%)
        """
        if self._peak_balance <= 0:
            return 0.0
        return (self._peak_balance - self._current_balance) / self._peak_balance

    def get_drawdown_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current drawdown.

        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        drawdown = self.get_current_drawdown()

        if drawdown < self.config.drawdown_d0_threshold:
            return self.config.drawdown_d0_multiplier
        elif drawdown < self.config.drawdown_d1_threshold:
            return self.config.drawdown_d1_multiplier
        elif drawdown < self.config.drawdown_d2_threshold:
            return self.config.drawdown_d2_multiplier
        elif drawdown < self.config.drawdown_d3_threshold:
            return self.config.drawdown_d3_multiplier
        elif drawdown < self.config.drawdown_d4_threshold:
            return self.config.drawdown_d4_multiplier
        else:
            return 0.0  # Complete block

    def is_drawdown_circuit_breaker_active(self) -> bool:
        """
        Check if drawdown circuit breaker is blocking entries.

        Returns:
            True if entries are blocked due to drawdown
        """
        return self.get_drawdown_size_multiplier() == 0.0

    # ─── Daily Loss Stop ─────────────────────────────────────────────────────

    def update_daily_pnl(self, pnl: float, current_day: str) -> None:
        """
        Update daily PnL and check if daily loss cap is exceeded.

        Args:
            pnl: PnL for the current trade
            current_day: Current day identifier (e.g., "2024-01-15")

        Returns:
            True if daily loss cap is exceeded
        """
        # Reset daily PnL on day boundary
        if self._current_day != current_day:
            self._daily_pnl = 0.0
            self._current_day = current_day

        self._daily_pnl += pnl

        # Check if daily loss cap is exceeded
        if self._daily_pnl < -self.config.daily_loss_cap_pct * self._initial_balance:
            logger.warning(
                f"Daily loss cap exceeded: {self._daily_pnl:.2f} < "
                f"-{self.config.daily_loss_cap_pct:.2%} of initial balance"
            )
            return True

        return False

    def is_daily_loss_cap_exceeded(self) -> bool:
        """
        Check if daily loss cap is exceeded.

        Returns:
            True if daily loss cap is exceeded
        """
        if self._initial_balance == 0:
            return False
        return self._daily_pnl < -self.config.daily_loss_cap_pct * self._initial_balance

    # ─── Consecutive Loss Throttle ────────────────────────────────────────────

    def update_consecutive_losses(self, is_loss: bool) -> None:
        """
        Update consecutive loss counter.

        Args:
            is_loss: True if the last trade was a loss
        """
        if is_loss:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def is_consecutive_loss_throttle_active(self) -> bool:
        """
        Check if consecutive loss throttle is blocking entries.

        Returns:
            True if entries are blocked due to consecutive losses
        """
        if self._consecutive_loss_cooldown > 0:
            return True

        return self._consecutive_losses >= self.config.max_consecutive_losses

    def trigger_consecutive_loss_cooldown(self) -> None:
        """Trigger cooldown after consecutive loss throttle is hit."""
        self._consecutive_loss_cooldown = self.config.consecutive_loss_cooldown_candles
        self._consecutive_losses = 0
        logger.info(f"Consecutive loss cooldown triggered for {self._consecutive_loss_cooldown} candles")

    # ─── Regime-Level Kill Switch ─────────────────────────────────────────────

    def record_trade_result(
        self,
        regime: str,
        pnl_r: float,
        is_win: bool,
    ) -> None:
        """
        Record a trade result for regime performance tracking.

        Args:
            regime: Regime identifier
            pnl_r: PnL in R units
            is_win: True if the trade was a win
        """
        # Initialize regime metrics if not exists
        if regime not in self._regime_metrics:
            self._regime_metrics[regime] = RegimePerformanceMetrics(
                regime=regime,
                trade_count=0,
                win_count=0,
                loss_count=0,
                total_pnl_r=0.0,
                expectancy_r=0.0,
                win_rate=0.0,
                avg_win_r=0.0,
                avg_loss_r=0.0,
            )
            self._regime_trade_buffer[regime] = []

        # Add to buffer
        self._regime_trade_buffer[regime].append({
            "pnl_r": pnl_r,
            "is_win": is_win,
            "timestamp": datetime.now(),
        })

        # Update metrics
        metrics = self._regime_metrics[regime]
        metrics.trade_count += 1
        metrics.total_pnl_r += pnl_r

        if is_win:
            metrics.win_count += 1
        else:
            metrics.loss_count += 1

        # Recalculate derived metrics
        if metrics.trade_count > 0:
            metrics.win_rate = metrics.win_count / metrics.trade_count
            metrics.expectancy_r = metrics.total_pnl_r / metrics.trade_count

        # Check if we should evaluate regime kill switch
        if len(self._regime_trade_buffer[regime]) >= self.config.regime_kill_switch_window:
            self._evaluate_regime_kill_switch(regime)

    def _evaluate_regime_kill_switch(self, regime: str) -> None:
        """
        Evaluate if regime should be killed based on recent performance.

        Kill switch triggers when:
        - Rolling regime expectancy below -0.05R
        - Win rate below required break-even by at least 5 percentage points
        - Observed across two consecutive monitoring windows

        Args:
            regime: Regime identifier
        """
        if regime not in self._regime_trade_buffer:
            return

        buffer = self._regime_trade_buffer[regime]
        if len(buffer) < self.config.regime_kill_switch_window:
            return

        # Calculate window metrics
        window_trades = buffer[-self.config.regime_kill_switch_window:]
        window_pnl_r = sum(t["pnl_r"] for t in window_trades)
        window_wins = sum(1 for t in window_trades if t["is_win"])
        window_expectancy = window_pnl_r / len(window_trades)
        window_win_rate = window_wins / len(window_trades)

        # Calculate break-even win rate for this regime
        # Simplified: assume 1R stop loss, average winner ~2R
        avg_win_r = sum(t["pnl_r"] for t in window_trades if t["is_win"]) / window_wins if window_wins > 0 else 2.0
        break_even_win_rate = 1.0 / (1.0 + avg_win_r)

        # Check kill switch conditions
        expectancy_below_threshold = window_expectancy < self.config.regime_kill_switch_min_expectancy
        win_rate_below_threshold = window_win_rate < (break_even_win_rate - self.config.regime_kill_switch_min_win_rate_diff)

        metrics = self._regime_metrics[regime]

        if expectancy_below_threshold and win_rate_below_threshold:
            metrics.consecutive_failures += 1

            if metrics.consecutive_failures >= self.config.regime_kill_switch_consecutive_failures:
                metrics.is_killed = True
                metrics.kill_reason = (
                    f"Regime {regime} killed: expectancy {window_expectancy:.3f}R < "
                    f"{self.config.regime_kill_switch_min_expectancy:.3f}R, "
                    f"win rate {window_win_rate:.2%} < break-even {break_even_win_rate:.2%} - "
                    f"{self.config.regime_kill_switch_min_win_rate_diff:.2%}"
                )
                logger.warning(metrics.kill_reason)
        else:
            # Reset consecutive failures if conditions improve
            if metrics.consecutive_failures > 0:
                metrics.consecutive_failures = 0

        # Clear buffer for next window
        self._regime_trade_buffer[regime] = []

    def is_regime_killed(self, regime: str) -> bool:
        """
        Check if a regime is currently killed.

        Args:
            regime: Regime identifier

        Returns:
            True if regime is killed
        """
        if regime not in self._regime_metrics:
            return False
        return self._regime_metrics[regime].is_killed

    def attempt_regime_recovery(self, regime: str) -> bool:
        """
        Attempt to recover a killed regime.

        Recovery requires positive rolling expectancy and gate pass for recovery window.

        Args:
            regime: Regime identifier

        Returns:
            True if regime was successfully recovered
        """
        if regime not in self._regime_metrics:
            return False

        metrics = self._regime_metrics[regime]
        if not metrics.is_killed:
            return True  # Already active

        # Check if we have enough recovery window data
        if len(self._regime_trade_buffer[regime]) < self.config.regime_recovery_window:
            return False

        # Calculate recovery window metrics
        buffer = self._regime_trade_buffer[regime]
        window_trades = buffer[-self.config.regime_recovery_window:]
        window_pnl_r = sum(t["pnl_r"] for t in window_trades)
        window_expectancy = window_pnl_r / len(window_trades)

        # Recovery condition: positive expectancy
        if window_expectancy > 0.0:
            metrics.is_killed = False
            metrics.kill_reason = None
            metrics.consecutive_failures = 0
            logger.info(f"Regime {regime} recovered: expectancy {window_expectancy:.3f}R > 0")
            return True

        return False

    # ─── Entry Gating Based on Deployment State ───────────────────────────────

    def should_allow_entry(
        self,
        regime: str,
        edge_score: float,
        direction_margin: float,
        edge_floor: float = 0.08,
        ambiguity_threshold: float = 0.05,
    ) -> Tuple[bool, str]:
        """
        Check if entry should be allowed based on deployment state.

        Args:
            regime: Current regime
            edge_score: Edge score for the trade
            direction_margin: Direction margin (long_score - short_score)
            edge_floor: Base edge floor for the regime
            ambiguity_threshold: Base ambiguity threshold for the regime

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check if regime is killed
        if self.is_regime_killed(regime):
            return False, f"Regime {regime} is killed: {self._regime_metrics[regime].kill_reason}"

        # Check deployment state
        state = self._current_state

        if state == DeploymentState.DEFENSIVE:
            return False, "DEFENSIVE mode: no new entries allowed"

        elif state == DeploymentState.PASSIVE:
            # Apply stricter thresholds
            passive_edge_floor = edge_floor * self.config.passive_edge_floor_multiplier
            passive_ambiguity_threshold = ambiguity_threshold * self.config.passive_ambiguity_threshold_multiplier

            if edge_score < passive_edge_floor:
                return False, (
                    f"PASSIVE mode: edge score {edge_score:.3f} < "
                    f"stricter floor {passive_edge_floor:.3f}"
                )

            if direction_margin < passive_ambiguity_threshold:
                return False, (
                    f"PASSIVE mode: direction margin {direction_margin:.3f} < "
                    f"stricter threshold {passive_ambiguity_threshold:.3f}"
                )

        elif state == DeploymentState.ACTIVE:
            # Normal gates apply
            if edge_score < edge_floor:
                return False, f"Edge score {edge_score:.3f} < floor {edge_floor:.3f}"

            if direction_margin < ambiguity_threshold:
                return False, f"Direction margin {direction_margin:.3f} < threshold {ambiguity_threshold:.3f}"

        # Check drawdown circuit breaker
        if self.is_drawdown_circuit_breaker_active():
            return False, f"Drawdown circuit breaker active: drawdown {self.get_current_drawdown():.2%}"

        # Check daily loss cap
        if self.is_daily_loss_cap_exceeded():
            return False, f"Daily loss cap exceeded: {self._daily_pnl:.2f}"

        # Check consecutive loss throttle
        if self.is_consecutive_loss_throttle_active():
            return False, f"Consecutive loss throttle active: {self._consecutive_losses} consecutive losses"

        return True, f"Entry allowed in {state.value} mode"

    # ─── Benchmark-Relative Guardrails ────────────────────────────────────────

    def update_benchmark_price(self, symbol: str, price: float, timestamp: datetime) -> None:
        """
        Update benchmark price for tracking.

        Args:
            symbol: Benchmark symbol (e.g., "BTC", "ETH", "SOL")
            price: Current price
            timestamp: Timestamp
        """
        if symbol not in self._benchmark_prices:
            self._benchmark_prices[symbol] = []

        self._benchmark_prices[symbol].append((timestamp, price))

        # Keep only recent data (last 365 days)
        cutoff = timestamp - timedelta(days=365)
        self._benchmark_prices[symbol] = [
            (t, p) for t, p in self._benchmark_prices[symbol] if t > cutoff
        ]

    def update_strategy_return(self, return_pct: float, timestamp: datetime) -> None:
        """
        Update strategy return for tracking.

        Args:
            return_pct: Strategy return as decimal
            timestamp: Timestamp
        """
        self._strategy_returns.append((timestamp, return_pct))

        # Keep only recent data (last 365 days)
        cutoff = timestamp - timedelta(days=365)
        self._strategy_returns = [
            (t, r) for t, r in self._strategy_returns if t > cutoff
        ]

    def check_benchmark_guardrails(
        self,
        benchmark_return: float,
        strategy_return: float,
        benchmark_drawdown: float,
        strategy_drawdown: float,
    ) -> Tuple[bool, str]:
        """
        Check benchmark-relative guardrails.

        Guardrails:
        1. Up-year capture: strategy must capture at least 60% of benchmark return
           or underperform by no more than 20 percentage points
        2. Drawdown: strategy max drawdown must not exceed benchmark drawdown by more than 5 percentage points

        Args:
            benchmark_return: Benchmark return as decimal
            strategy_return: Strategy return as decimal
            benchmark_drawdown: Benchmark max drawdown as decimal
            strategy_drawdown: Strategy max drawdown as decimal

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        # Up-year capture guardrail
        if benchmark_return > self.config.benchmark_up_year_threshold:
            capture_ratio = strategy_return / benchmark_return if benchmark_return > 0 else 0.0
            underperformance = benchmark_return - strategy_return

            if capture_ratio < self.config.benchmark_min_capture_pct and underperformance > self.config.benchmark_max_underperformance_pct:
                return (
                    False,
                    f"Up-year capture guardrail failed: capture {capture_ratio:.2%} < "
                    f"{self.config.benchmark_min_capture_pct:.2%}, underperformance {underperformance:.2%} > "
                    f"{self.config.benchmark_max_underperformance_pct:.2%}"
                )

        # Drawdown guardrail
        drawdown_excess = strategy_drawdown - benchmark_drawdown
        if drawdown_excess > self.config.benchmark_drawdown_excess_limit:
            return (
                False,
                f"Drawdown guardrail failed: strategy drawdown {strategy_drawdown:.2%} exceeds "
                f"benchmark drawdown {benchmark_drawdown:.2%} by {drawdown_excess:.2%} > "
                f"{self.config.benchmark_drawdown_excess_limit:.2%}"
            )

        return True, "Benchmark guardrails passed"

    def check_live_degradation_guardrail(
        self,
        rolling_underperformance: float,
        current_drawdown: float,
    ) -> Tuple[bool, str]:
        """
        Check live degradation guardrail.

        If rolling 60-day underperformance exceeds 12 percentage points and
        active drawdown is above 8%, force passive defer or no-trade mode.

        Args:
            rolling_underperformance: Rolling underperformance vs benchmark as decimal
            current_drawdown: Current drawdown as decimal

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        if (
            rolling_underperformance > self.config.live_degradation_max_underperformance
            and current_drawdown > self.config.live_degradation_min_drawdown
        ):
            return (
                False,
                f"Live degradation guardrail triggered: underperformance {rolling_underperformance:.2%} > "
                f"{self.config.live_degradation_max_underperformance:.2%}, drawdown {current_drawdown:.2%} > "
                f"{self.config.live_degradation_min_drawdown:.2%}"
            )

        return True, "Live degradation guardrail passed"

    # ─── Utility Methods ─────────────────────────────────────────────────────

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get the history of state transitions."""
        return [t.to_dict() for t in self._state_history]

    def get_regime_metrics(self, regime: str) -> Optional[RegimePerformanceMetrics]:
        """Get performance metrics for a regime."""
        return self._regime_metrics.get(regime)

    def get_all_regime_metrics(self) -> Dict[str, RegimePerformanceMetrics]:
        """Get performance metrics for all regimes."""
        return self._regime_metrics.copy()

    def reset(self) -> None:
        """Reset the capital protection manager to initial state."""
        self._current_state = DeploymentState.ACTIVE
        self._state_history.clear()
        self._candles_since_last_transition = 0
        self._last_transition_time = None
        self._peak_balance = 0.0
        self._current_balance = 0.0
        self._daily_pnl = 0.0
        self._current_day = None
        self._consecutive_losses = 0
        self._consecutive_loss_cooldown = 0
        self._regime_metrics.clear()
        self._regime_trade_buffer.clear()
        self._benchmark_prices.clear()
        self._strategy_returns.clear()
        self._trade_history.clear()

        logger.info("CapitalProtectionManager reset to initial state")
