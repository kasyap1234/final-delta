"""
Comparison report module for differential testing.

This module provides data structures and metrics for comparing
live trading and backtest results.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
from pathlib import Path


class DivergenceSeverity(Enum):
    """Severity levels for divergence detection."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class ComparisonCategory(Enum):
    """Categories of comparison between live and backtest."""
    SIGNAL_GENERATION = "signal_generation"
    ORDER_EXECUTION = "order_execution"
    POSITION_MANAGEMENT = "position_management"
    PNL_CALCULATION = "pnl_calculation"
    RISK_MANAGEMENT = "risk_management"
    TIMING = "timing"
    STATE_TRANSITION = "state_transition"


@dataclass
class SignalComparison:
    """Comparison of signal generation between live and backtest."""
    timestamp: datetime
    symbol: str
    live_signal: Optional[str] = None
    backtest_signal: Optional[str] = None
    live_strength: float = 0.0
    backtest_strength: float = 0.0
    live_price: float = 0.0
    backtest_price: float = 0.0
    signal_match: bool = False
    strength_diff: float = 0.0
    price_diff: float = 0.0
    severity: DivergenceSeverity = DivergenceSeverity.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderComparison:
    """Comparison of order execution between live and backtest."""
    timestamp: datetime
    symbol: str
    order_id: str
    side: str
    order_type: str
    live_size: float = 0.0
    backtest_size: float = 0.0
    live_price: float = 0.0
    backtest_price: float = 0.0
    live_filled: float = 0.0
    backtest_filled: float = 0.0
    live_fill_price: float = 0.0
    backtest_fill_price: float = 0.0
    size_diff: float = 0.0
    price_diff: float = 0.0
    fill_diff: float = 0.0
    fill_price_diff: float = 0.0
    slippage_live: float = 0.0
    slippage_backtest: float = 0.0
    severity: DivergenceSeverity = DivergenceSeverity.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionComparison:
    """Comparison of position management between live and backtest."""
    timestamp: datetime
    symbol: str
    position_id: str
    side: str
    live_entry_price: float = 0.0
    backtest_entry_price: float = 0.0
    live_size: float = 0.0
    backtest_size: float = 0.0
    live_stop_loss: float = 0.0
    backtest_stop_loss: float = 0.0
    live_take_profit: float = 0.0
    backtest_take_profit: float = 0.0
    entry_price_diff: float = 0.0
    size_diff: float = 0.0
    stop_loss_diff: float = 0.0
    take_profit_diff: float = 0.0
    severity: DivergenceSeverity = DivergenceSeverity.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PnLComparison:
    """Comparison of P&L calculations between live and backtest."""
    timestamp: datetime
    symbol: str
    trade_id: str
    live_realized_pnl: float = 0.0
    backtest_realized_pnl: float = 0.0
    live_unrealized_pnl: float = 0.0
    backtest_unrealized_pnl: float = 0.0
    live_fees: float = 0.0
    backtest_fees: float = 0.0
    realized_pnl_diff: float = 0.0
    unrealized_pnl_diff: float = 0.0
    fees_diff: float = 0.0
    total_pnl_diff: float = 0.0
    severity: DivergenceSeverity = DivergenceSeverity.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskComparison:
    """Comparison of risk management decisions between live and backtest."""
    timestamp: datetime
    symbol: str
    check_type: str
    live_allowed: bool = False
    backtest_allowed: bool = False
    live_risk_amount: float = 0.0
    backtest_risk_amount: float = 0.0
    live_position_size: float = 0.0
    backtest_position_size: float = 0.0
    decision_match: bool = False
    risk_amount_diff: float = 0.0
    position_size_diff: float = 0.0
    live_reason: str = ""
    backtest_reason: str = ""
    severity: DivergenceSeverity = DivergenceSeverity.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingComparison:
    """Comparison of timing between live and backtest."""
    timestamp: datetime
    event_type: str
    live_timestamp: datetime = field(default_factory=datetime.now)
    backtest_timestamp: datetime = field(default_factory=datetime.now)
    time_diff_seconds: float = 0.0
    severity: DivergenceSeverity = DivergenceSeverity.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransitionComparison:
    """Comparison of state transitions between live and backtest."""
    timestamp: datetime
    transition_type: str
    live_from_state: str = ""
    live_to_state: str = ""
    backtest_from_state: str = ""
    backtest_to_state: str = ""
    transition_match: bool = False
    severity: DivergenceSeverity = DivergenceSeverity.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergenceMetric:
    """Metric for tracking divergence in a specific category."""
    category: ComparisonCategory
    total_comparisons: int = 0
    divergences: int = 0
    minor_divergences: int = 0
    moderate_divergences: int = 0
    critical_divergences: int = 0
    max_divergence_value: float = 0.0
    avg_divergence_value: float = 0.0
    divergence_rate: float = 0.0

    def update(self, severity: DivergenceSeverity, value: float = 0.0):
        """Update metric with a new divergence."""
        self.total_comparisons += 1
        
        if severity != DivergenceSeverity.NONE:
            self.divergences += 1
            
            if severity == DivergenceSeverity.MINOR:
                self.minor_divergences += 1
            elif severity == DivergenceSeverity.MODERATE:
                self.moderate_divergences += 1
            elif severity == DivergenceSeverity.CRITICAL:
                self.critical_divergences += 1
            
            self.max_divergence_value = max(self.max_divergence_value, abs(value))
            
            # Update average
            current_sum = self.avg_divergence_value * (self.divergences - 1)
            self.avg_divergence_value = (current_sum + abs(value)) / self.divergences
        
        self.divergence_rate = self.divergences / self.total_comparisons if self.total_comparisons > 0 else 0.0


@dataclass
class ComparisonReport:
    """Complete comparison report between live and backtest."""
    report_id: str
    generated_at: datetime
    start_time: datetime
    end_time: datetime
    symbols: List[str]
    
    # Raw comparison data
    signal_comparisons: List[SignalComparison] = field(default_factory=list)
    order_comparisons: List[OrderComparison] = field(default_factory=list)
    position_comparisons: List[PositionComparison] = field(default_factory=list)
    pnl_comparisons: List[PnLComparison] = field(default_factory=list)
    risk_comparisons: List[RiskComparison] = field(default_factory=list)
    timing_comparisons: List[TimingComparison] = field(default_factory=list)
    state_transitions: List[StateTransitionComparison] = field(default_factory=list)
    
    # Summary metrics
    metrics: Dict[ComparisonCategory, DivergenceMetric] = field(default_factory=dict)
    
    # Overall assessment
    overall_severity: DivergenceSeverity = DivergenceSeverity.NONE
    passed_validation: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metrics for all categories."""
        for category in ComparisonCategory:
            if category not in self.metrics:
                self.metrics[category] = DivergenceMetric(category=category)
    
    def add_signal_comparison(self, comparison: SignalComparison):
        """Add a signal comparison and update metrics."""
        self.signal_comparisons.append(comparison)
        self.metrics[ComparisonCategory.SIGNAL_GENERATION].update(
            comparison.severity,
            comparison.strength_diff
        )
    
    def add_order_comparison(self, comparison: OrderComparison):
        """Add an order comparison and update metrics."""
        self.order_comparisons.append(comparison)
        self.metrics[ComparisonCategory.ORDER_EXECUTION].update(
            comparison.severity,
            comparison.fill_price_diff
        )
    
    def add_position_comparison(self, comparison: PositionComparison):
        """Add a position comparison and update metrics."""
        self.position_comparisons.append(comparison)
        self.metrics[ComparisonCategory.POSITION_MANAGEMENT].update(
            comparison.severity,
            comparison.entry_price_diff
        )
    
    def add_pnl_comparison(self, comparison: PnLComparison):
        """Add a P&L comparison and update metrics."""
        self.pnl_comparisons.append(comparison)
        self.metrics[ComparisonCategory.PNL_CALCULATION].update(
            comparison.severity,
            comparison.total_pnl_diff
        )
    
    def add_risk_comparison(self, comparison: RiskComparison):
        """Add a risk comparison and update metrics."""
        self.risk_comparisons.append(comparison)
        self.metrics[ComparisonCategory.RISK_MANAGEMENT].update(
            comparison.severity,
            comparison.risk_amount_diff
        )
    
    def add_timing_comparison(self, comparison: TimingComparison):
        """Add a timing comparison and update metrics."""
        self.timing_comparisons.append(comparison)
        self.metrics[ComparisonCategory.TIMING].update(
            comparison.severity,
            comparison.time_diff_seconds
        )
    
    def add_state_transition(self, transition: StateTransitionComparison):
        """Add a state transition comparison and update metrics."""
        self.state_transitions.append(transition)
        self.metrics[ComparisonCategory.STATE_TRANSITION].update(
            transition.severity,
            1.0 if not transition.transition_match else 0.0
        )
    
    def calculate_overall_severity(self) -> DivergenceSeverity:
        """Calculate overall severity based on all metrics."""
        critical_count = sum(
            m.critical_divergences for m in self.metrics.values()
        )
        moderate_count = sum(
            m.moderate_divergences for m in self.metrics.values()
        )
        minor_count = sum(
            m.minor_divergences for m in self.metrics.values()
        )
        
        if critical_count > 0:
            return DivergenceSeverity.CRITICAL
        elif moderate_count > 0:
            return DivergenceSeverity.MODERATE
        elif minor_count > 0:
            return DivergenceSeverity.MINOR
        else:
            return DivergenceSeverity.NONE
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the report."""
        total_comparisons = sum(
            m.total_comparisons for m in self.metrics.values()
        )
        total_divergences = sum(
            m.divergences for m in self.metrics.values()
        )
        
        self.summary = {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'test_period': {
                'start': self.start_time.isoformat(),
                'end': self.end_time.isoformat()
            },
            'symbols': self.symbols,
            'total_comparisons': total_comparisons,
            'total_divergences': total_divergences,
            'divergence_rate': total_divergences / total_comparisons if total_comparisons > 0 else 0.0,
            'overall_severity': self.overall_severity.value,
            'passed_validation': self.passed_validation,
            'category_metrics': {
                cat.value: {
                    'total_comparisons': m.total_comparisons,
                    'divergences': m.divergences,
                    'divergence_rate': m.divergence_rate,
                    'minor': m.minor_divergences,
                    'moderate': m.moderate_divergences,
                    'critical': m.critical_divergences,
                    'max_divergence': m.max_divergence_value,
                    'avg_divergence': m.avg_divergence_value
                }
                for cat, m in self.metrics.items()
            },
            'critical_issues': self._get_critical_issues(),
            'recommendations': self._generate_recommendations()
        }
        
        return self.summary
    
    def _get_critical_issues(self) -> List[Dict[str, Any]]:
        """Get list of critical divergence issues."""
        issues = []
        
        for comp in self.signal_comparisons:
            if comp.severity == DivergenceSeverity.CRITICAL:
                issues.append({
                    'category': 'signal_generation',
                    'timestamp': comp.timestamp.isoformat(),
                    'symbol': comp.symbol,
                    'issue': f"Signal mismatch: live={comp.live_signal}, backtest={comp.backtest_signal}",
                    'severity': 'critical'
                })
        
        for comp in self.order_comparisons:
            if comp.severity == DivergenceSeverity.CRITICAL:
                issues.append({
                    'category': 'order_execution',
                    'timestamp': comp.timestamp.isoformat(),
                    'symbol': comp.symbol,
                    'issue': f"Order fill mismatch: live={comp.live_fill_price}, backtest={comp.backtest_fill_price}",
                    'severity': 'critical'
                })
        
        for comp in self.pnl_comparisons:
            if comp.severity == DivergenceSeverity.CRITICAL:
                issues.append({
                    'category': 'pnl_calculation',
                    'timestamp': comp.timestamp.isoformat(),
                    'symbol': comp.symbol,
                    'issue': f"P&L mismatch: diff=${comp.total_pnl_diff:.2f}",
                    'severity': 'critical'
                })
        
        for comp in self.risk_comparisons:
            if comp.severity == DivergenceSeverity.CRITICAL:
                issues.append({
                    'category': 'risk_management',
                    'timestamp': comp.timestamp.isoformat(),
                    'symbol': comp.symbol,
                    'issue': f"Risk decision mismatch: live={comp.live_allowed}, backtest={comp.backtest_allowed}",
                    'severity': 'critical'
                })
        
        return issues
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        for category, metric in self.metrics.items():
            if metric.critical_divergences > 0:
                recommendations.append(
                    f"CRITICAL: Fix {category.value} - {metric.critical_divergences} critical divergences found"
                )
            elif metric.moderate_divergences > 0:
                recommendations.append(
                    f"MODERATE: Review {category.value} - {metric.moderate_divergences} moderate divergences found"
                )
            elif metric.minor_divergences > 10:  # Threshold for minor issues
                recommendations.append(
                    f"MINOR: Consider reviewing {category.value} - {metric.minor_divergences} minor divergences found"
                )
        
        if not recommendations:
            recommendations.append("No significant divergences found. Backtest fidelity is good.")
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'symbols': self.symbols,
            'signal_comparisons': [asdict(c) for c in self.signal_comparisons],
            'order_comparisons': [asdict(c) for c in self.order_comparisons],
            'position_comparisons': [asdict(c) for c in self.position_comparisons],
            'pnl_comparisons': [asdict(c) for c in self.pnl_comparisons],
            'risk_comparisons': [asdict(c) for c in self.risk_comparisons],
            'timing_comparisons': [asdict(c) for c in self.timing_comparisons],
            'state_transitions': [asdict(c) for c in self.state_transitions],
            'metrics': {
                cat.value: asdict(m) for cat, m in self.metrics.items()
            },
            'overall_severity': self.overall_severity.value,
            'passed_validation': self.passed_validation,
            'summary': self.summary
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export report to JSON string or file."""
        data = self.to_dict()
        
        # Convert datetime objects to strings
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        
        json_str = json.dumps(data, indent=2, default=serialize_datetime)
        
        if filepath:
            Path(filepath).write_text(json_str)
        
        return json_str
    
    def generate_html(self, filepath: Optional[str] = None) -> str:
        """Generate HTML report."""
        summary = self.generate_summary()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Fidelity Report - {self.report_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #333;
        }}
        .severity-none {{ color: #4CAF50; }}
        .severity-minor {{ color: #FFC107; }}
        .severity-moderate {{ color: #FF9800; }}
        .severity-critical {{ color: #F44336; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .issue-critical {{
            background-color: #ffebee;
            border-left: 4px solid #F44336;
            padding: 10px;
            margin: 5px 0;
        }}
        .issue-moderate {{
            background-color: #fff3e0;
            border-left: 4px solid #FF9800;
            padding: 10px;
            margin: 5px 0;
        }}
        .recommendation {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 5px 0;
        }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .fail {{ color: #F44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Fidelity Validation Report</h1>
        <p>Report ID: {self.report_id}</p>
        <p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-card">
            <h2>Overall Result</h2>
            <p class="{'pass' if self.passed_validation else 'fail'}">
                {'PASSED' if self.passed_validation else 'FAILED'}
            </p>
            <p>Severity: <span class="severity-{self.overall_severity.value}">{self.overall_severity.value.upper()}</span></p>
        </div>
        
        <div class="summary-card">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <span class="metric-label">Total Comparisons:</span>
                <span class="metric-value">{summary.get('total_comparisons', 0)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Divergences:</span>
                <span class="metric-value">{summary.get('total_divergences', 0)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Divergence Rate:</span>
                <span class="metric-value">{summary.get('divergence_rate', 0):.2%}</span>
            </div>
        </div>
        
        <h2>Category Metrics</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Comparisons</th>
                <th>Divergences</th>
                <th>Rate</th>
                <th>Minor</th>
                <th>Moderate</th>
                <th>Critical</th>
            </tr>
"""
        
        for cat, m in self.metrics.items():
            html += f"""
            <tr>
                <td>{cat.value.replace('_', ' ').title()}</td>
                <td>{m.total_comparisons}</td>
                <td>{m.divergences}</td>
                <td>{m.divergence_rate:.2%}</td>
                <td>{m.minor_divergences}</td>
                <td>{m.moderate_divergences}</td>
                <td class="severity-{DivergenceSeverity.CRITICAL.value if m.critical_divergences > 0 else 'none'}">{m.critical_divergences}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>Critical Issues</h2>
"""
        
        critical_issues = summary.get('critical_issues', [])
        if critical_issues:
            for issue in critical_issues:
                html += f"""
        <div class="issue-{issue['severity']}">
            <strong>[{issue['severity'].upper()}]</strong> {issue['category']}: {issue['issue']}
            <br><small>{issue['timestamp']}</small>
        </div>
"""
        else:
            html += "<p>No critical issues found.</p>"
        
        html += """
        <h2>Recommendations</h2>
"""
        
        recommendations = summary.get('recommendations', [])
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        
        html += """
    </div>
</body>
</html>
"""
        
        if filepath:
            Path(filepath).write_text(html)
        
        return html


class ToleranceConfig:
    """Configuration for tolerance thresholds in comparisons."""
    
    def __init__(
        self,
        signal_strength_tolerance: float = 0.05,
        signal_price_tolerance: float = 0.001,
        order_size_tolerance: float = 0.001,
        order_price_tolerance: float = 0.001,
        fill_price_tolerance: float = 0.005,
        position_size_tolerance: float = 0.001,
        entry_price_tolerance: float = 0.001,
        stop_loss_tolerance: float = 0.001,
        take_profit_tolerance: float = 0.001,
        pnl_tolerance: float = 0.01,
        fee_tolerance: float = 0.01,
        risk_amount_tolerance: float = 0.01,
        timing_tolerance_seconds: float = 1.0,
        minor_threshold: float = 0.01,
        moderate_threshold: float = 0.05,
        critical_threshold: float = 0.10
    ):
        self.signal_strength_tolerance = signal_strength_tolerance
        self.signal_price_tolerance = signal_price_tolerance
        self.order_size_tolerance = order_size_tolerance
        self.order_price_tolerance = order_price_tolerance
        self.fill_price_tolerance = fill_price_tolerance
        self.position_size_tolerance = position_size_tolerance
        self.entry_price_tolerance = entry_price_tolerance
        self.stop_loss_tolerance = stop_loss_tolerance
        self.take_profit_tolerance = take_profit_tolerance
        self.pnl_tolerance = pnl_tolerance
        self.fee_tolerance = fee_tolerance
        self.risk_amount_tolerance = risk_amount_tolerance
        self.timing_tolerance_seconds = timing_tolerance_seconds
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.critical_threshold = critical_threshold
    
    def determine_severity(self, relative_diff: float) -> DivergenceSeverity:
        """Determine severity based on relative difference."""
        if relative_diff >= self.critical_threshold:
            return DivergenceSeverity.CRITICAL
        elif relative_diff >= self.moderate_threshold:
            return DivergenceSeverity.MODERATE
        elif relative_diff >= self.minor_threshold:
            return DivergenceSeverity.MINOR
        else:
            return DivergenceSeverity.NONE
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'signal_strength_tolerance': self.signal_strength_tolerance,
            'signal_price_tolerance': self.signal_price_tolerance,
            'order_size_tolerance': self.order_size_tolerance,
            'order_price_tolerance': self.order_price_tolerance,
            'fill_price_tolerance': self.fill_price_tolerance,
            'position_size_tolerance': self.position_size_tolerance,
            'entry_price_tolerance': self.entry_price_tolerance,
            'stop_loss_tolerance': self.stop_loss_tolerance,
            'take_profit_tolerance': self.take_profit_tolerance,
            'pnl_tolerance': self.pnl_tolerance,
            'fee_tolerance': self.fee_tolerance,
            'risk_amount_tolerance': self.risk_amount_tolerance,
            'timing_tolerance_seconds': self.timing_tolerance_seconds,
            'minor_threshold': self.minor_threshold,
            'moderate_threshold': self.moderate_threshold,
            'critical_threshold': self.critical_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ToleranceConfig':
        """Create from dictionary."""
        return cls(**data)
