"""
Differential tester module for comparing live and backtest results.

This module provides comprehensive comparison logic to validate that
backtest produces identical results to live trading bot.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import uuid

from .comparison_report import (
    ComparisonReport,
    SignalComparison,
    OrderComparison,
    PositionComparison,
    PnLComparison,
    RiskComparison,
    TimingComparison,
    StateTransitionComparison,
    DivergenceSeverity,
    ComparisonCategory,
    ToleranceConfig
)

from .mock_live_runner import LiveDataCapture

logger = logging.getLogger(__name__)


class DifferentialTester:
    """
    Differential tester for comparing live and backtest results.
    
    This class performs detailed comparisons between live trading data
    and backtest results to identify any discrepancies.
    """
    
    def __init__(
        self,
        tolerance_config: Optional[ToleranceConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the differential tester.
        
        Args:
            tolerance_config: Configuration for tolerance thresholds
            output_dir: Directory to save reports
        """
        self.tolerance = tolerance_config or ToleranceConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("validation_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DifferentialTester initialized")
    
    def compare_results(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        symbols: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ComparisonReport:
        """
        Compare live trading data with backtest results.
        
        Args:
            live_data: Captured live trading data
            backtest_data: Backtest results
            symbols: List of symbols tested
            start_time: Test start time
            end_time: Test end time
            
        Returns:
            ComparisonReport with all findings
        """
        report_id = str(uuid.uuid4())[:8]
        
        report = ComparisonReport(
            report_id=report_id,
            generated_at=datetime.now(),
            start_time=start_time or datetime.now(),
            end_time=end_time or datetime.now(),
            symbols=symbols
        )
        
        logger.info(f"Starting comparison {report_id} for {len(symbols)} symbols")
        
        # Compare signals
        self._compare_signals(live_data, backtest_data, report)
        
        # Compare orders
        self._compare_orders(live_data, backtest_data, report)
        
        # Compare positions
        self._compare_positions(live_data, backtest_data, report)
        
        # Compare P&L
        self._compare_pnl(live_data, backtest_data, report)
        
        # Compare risk checks
        self._compare_risk_checks(live_data, backtest_data, report)
        
        # Compare timing
        self._compare_timing(live_data, backtest_data, report)
        
        # Compare state transitions
        self._compare_state_transitions(live_data, backtest_data, report)
        
        # Calculate overall severity
        report.overall_severity = report.calculate_overall_severity()
        
        # Determine if validation passed
        report.passed_validation = (
            report.overall_severity != DivergenceSeverity.CRITICAL and
            report.metrics[ComparisonCategory.PNL_CALCULATION].critical_divergences == 0 and
            report.metrics[ComparisonCategory.SIGNAL_GENERATION].critical_divergences == 0
        )
        
        # Generate summary
        report.generate_summary()
        
        logger.info(f"Comparison {report_id} complete. Severity: {report.overall_severity.value}")
        
        return report
    
    def _compare_signals(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        report: ComparisonReport
    ):
        """Compare signal generation between live and backtest."""
        live_signals = live_data.get('signals', [])
        backtest_signals = backtest_data.get('signals', [])
        
        logger.debug(f"Comparing {len(live_signals)} live signals with {len(backtest_signals)} backtest signals")
        
        # Index signals by timestamp and symbol for matching
        live_by_key = {}
        for sig in live_signals:
            key = (sig.get('timestamp'), sig.get('symbol'))
            live_by_key[key] = sig
        
        backtest_by_key = {}
        for sig in backtest_signals:
            key = (sig.get('timestamp'), sig.get('symbol'))
            backtest_by_key[key] = sig
        
        # Compare all unique keys
        all_keys = set(live_by_key.keys()) | set(backtest_by_key.keys())
        
        for key in all_keys:
            live_sig = live_by_key.get(key)
            backtest_sig = backtest_by_key.get(key)
            
            comparison = self._create_signal_comparison(live_sig, backtest_sig, key)
            report.add_signal_comparison(comparison)
    
    def _create_signal_comparison(
        self,
        live_sig: Optional[Dict],
        backtest_sig: Optional[Dict],
        key: Tuple
    ) -> SignalComparison:
        """Create a signal comparison."""
        timestamp_str, symbol = key
        
        # Parse timestamp
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str or datetime.now()
        
        live_signal = live_sig.get('signal_type') if live_sig else None
        backtest_signal = backtest_sig.get('signal_type') if backtest_sig else None
        
        live_strength = live_sig.get('strength', 0.0) if live_sig else 0.0
        backtest_strength = backtest_sig.get('strength', 0.0) if backtest_sig else 0.0
        
        live_price = live_sig.get('price', 0.0) if live_sig else 0.0
        backtest_price = backtest_sig.get('price', 0.0) if backtest_sig else 0.0
        
        # Calculate differences
        signal_match = live_signal == backtest_signal
        strength_diff = abs(live_strength - backtest_strength)
        price_diff = abs(live_price - backtest_price)
        
        # Determine severity
        if not signal_match:
            severity = DivergenceSeverity.CRITICAL
        elif live_price > 0 and price_diff / live_price > self.tolerance.critical_threshold:
            severity = DivergenceSeverity.MODERATE
        elif strength_diff > self.tolerance.signal_strength_tolerance:
            severity = DivergenceSeverity.MINOR
        else:
            severity = DivergenceSeverity.NONE
        
        return SignalComparison(
            timestamp=timestamp,
            symbol=symbol,
            live_signal=live_signal,
            backtest_signal=backtest_signal,
            live_strength=live_strength,
            backtest_strength=backtest_strength,
            live_price=live_price,
            backtest_price=backtest_price,
            signal_match=signal_match,
            strength_diff=strength_diff,
            price_diff=price_diff,
            severity=severity,
            details={
                'live_indicators': live_sig.get('indicators') if live_sig else None,
                'backtest_indicators': backtest_sig.get('indicators') if backtest_sig else None
            }
        )
    
    def _compare_orders(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        report: ComparisonReport
    ):
        """Compare order execution between live and backtest."""
        live_orders = live_data.get('orders', [])
        backtest_orders = backtest_data.get('orders', [])
        
        logger.debug(f"Comparing {len(live_orders)} live orders with {len(backtest_orders)} backtest orders")
        
        # Index by order ID
        live_by_id = {o.get('order_id'): o for o in live_orders if o.get('order_id')}
        backtest_by_id = {o.get('order_id'): o for o in backtest_orders if o.get('order_id')}
        
        # Also index by symbol and timestamp for matching
        live_by_key = {}
        for o in live_orders:
            key = (o.get('symbol'), o.get('timestamp'), o.get('side'))
            live_by_key[key] = o
        
        backtest_by_key = {}
        for o in backtest_orders:
            key = (o.get('symbol'), o.get('timestamp'), o.get('side'))
            backtest_by_key[key] = o
        
        # Compare matched orders
        all_keys = set(live_by_key.keys()) | set(backtest_by_key.keys())
        
        for key in all_keys:
            live_order = live_by_key.get(key)
            backtest_order = backtest_by_key.get(key)
            
            comparison = self._create_order_comparison(live_order, backtest_order, key)
            report.add_order_comparison(comparison)
    
    def _create_order_comparison(
        self,
        live_order: Optional[Dict],
        backtest_order: Optional[Dict],
        key: Tuple
    ) -> OrderComparison:
        """Create an order comparison."""
        symbol, timestamp_str, side = key
        
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str or datetime.now()
        
        order_id = live_order.get('order_id') if live_order else (backtest_order.get('order_id') if backtest_order else 'unknown')
        order_type = live_order.get('order_type') if live_order else (backtest_order.get('order_type') if backtest_order else 'market')
        
        live_size = live_order.get('size', 0.0) if live_order else 0.0
        backtest_size = backtest_order.get('size', 0.0) if backtest_order else 0.0
        
        live_price = live_order.get('price', 0.0) if live_order else 0.0
        backtest_price = backtest_order.get('price', 0.0) if backtest_order else 0.0
        
        live_filled = live_order.get('filled_size', 0.0) if live_order else 0.0
        backtest_filled = backtest_order.get('filled_size', 0.0) if backtest_order else 0.0
        
        live_fill_price = live_order.get('fill_price', 0.0) if live_order else 0.0
        backtest_fill_price = backtest_order.get('fill_price', 0.0) if backtest_order else 0.0
        
        # Calculate differences
        size_diff = abs(live_size - backtest_size)
        price_diff = abs(live_price - backtest_price)
        fill_diff = abs(live_filled - backtest_filled)
        fill_price_diff = abs(live_fill_price - backtest_fill_price)
        
        # Calculate slippage
        slippage_live = live_order.get('slippage', 0.0) if live_order else 0.0
        slippage_backtest = backtest_order.get('slippage', 0.0) if backtest_order else 0.0
        
        # Determine severity
        avg_price = (live_fill_price + backtest_fill_price) / 2 if (live_fill_price + backtest_fill_price) > 0 else 1.0
        fill_price_relative_diff = fill_price_diff / avg_price if avg_price > 0 else 0.0
        
        if fill_price_relative_diff > self.tolerance.critical_threshold:
            severity = DivergenceSeverity.CRITICAL
        elif fill_price_relative_diff > self.tolerance.moderate_threshold:
            severity = DivergenceSeverity.MODERATE
        elif fill_price_relative_diff > self.tolerance.minor_threshold:
            severity = DivergenceSeverity.MINOR
        else:
            severity = DivergenceSeverity.NONE
        
        return OrderComparison(
            timestamp=timestamp,
            symbol=symbol,
            order_id=order_id,
            side=side,
            order_type=order_type,
            live_size=live_size,
            backtest_size=backtest_size,
            live_price=live_price,
            backtest_price=backtest_price,
            live_filled=live_filled,
            backtest_filled=backtest_filled,
            live_fill_price=live_fill_price,
            backtest_fill_price=backtest_fill_price,
            size_diff=size_diff,
            price_diff=price_diff,
            fill_diff=fill_diff,
            fill_price_diff=fill_price_diff,
            slippage_live=slippage_live,
            slippage_backtest=slippage_backtest,
            severity=severity,
            details={
                'live_fees': live_order.get('fees') if live_order else None,
                'backtest_fees': backtest_order.get('fees') if backtest_order else None
            }
        )
    
    def _compare_positions(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        report: ComparisonReport
    ):
        """Compare position management between live and backtest."""
        live_positions = live_data.get('positions', [])
        backtest_positions = backtest_data.get('positions', [])
        
        logger.debug(f"Comparing {len(live_positions)} live positions with {len(backtest_positions)} backtest positions")
        
        # Index by position ID
        live_by_id = {p.get('position_id'): p for p in live_positions if p.get('position_id')}
        backtest_by_id = {p.get('position_id'): p for p in backtest_positions if p.get('position_id')}
        
        # Compare all unique position IDs
        all_ids = set(live_by_id.keys()) | set(backtest_by_id.keys())
        
        for pos_id in all_ids:
            live_pos = live_by_id.get(pos_id)
            backtest_pos = backtest_by_id.get(pos_id)
            
            comparison = self._create_position_comparison(live_pos, backtest_pos, pos_id)
            report.add_position_comparison(comparison)
    
    def _create_position_comparison(
        self,
        live_pos: Optional[Dict],
        backtest_pos: Optional[Dict],
        position_id: str
    ) -> PositionComparison:
        """Create a position comparison."""
        # Get timestamp from either position
        timestamp_str = live_pos.get('timestamp') if live_pos else (backtest_pos.get('timestamp') if backtest_pos else None)
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str or datetime.now()
        
        symbol = live_pos.get('symbol') if live_pos else (backtest_pos.get('symbol') if backtest_pos else 'unknown')
        side = live_pos.get('side') if live_pos else (backtest_pos.get('side') if backtest_pos else 'long')
        
        live_entry = live_pos.get('entry_price', 0.0) if live_pos else 0.0
        backtest_entry = backtest_pos.get('entry_price', 0.0) if backtest_pos else 0.0
        
        live_size = live_pos.get('size', 0.0) if live_pos else 0.0
        backtest_size = backtest_pos.get('size', 0.0) if backtest_pos else 0.0
        
        live_sl = live_pos.get('stop_loss', 0.0) if live_pos else 0.0
        backtest_sl = backtest_pos.get('stop_loss', 0.0) if backtest_pos else 0.0
        
        live_tp = live_pos.get('take_profit', 0.0) if live_pos else 0.0
        backtest_tp = backtest_pos.get('take_profit', 0.0) if backtest_pos else 0.0
        
        # Calculate differences
        entry_price_diff = abs(live_entry - backtest_entry)
        size_diff = abs(live_size - backtest_size)
        stop_loss_diff = abs(live_sl - backtest_sl)
        take_profit_diff = abs(live_tp - backtest_tp)
        
        # Determine severity based on entry price difference
        avg_entry = (live_entry + backtest_entry) / 2 if (live_entry + backtest_entry) > 0 else 1.0
        entry_relative_diff = entry_price_diff / avg_entry if avg_entry > 0 else 0.0
        
        if entry_relative_diff > self.tolerance.critical_threshold:
            severity = DivergenceSeverity.CRITICAL
        elif entry_relative_diff > self.tolerance.moderate_threshold:
            severity = DivergenceSeverity.MODERATE
        elif entry_relative_diff > self.tolerance.minor_threshold:
            severity = DivergenceSeverity.MINOR
        else:
            severity = DivergenceSeverity.NONE
        
        return PositionComparison(
            timestamp=timestamp,
            symbol=symbol,
            position_id=position_id,
            side=side,
            live_entry_price=live_entry,
            backtest_entry_price=backtest_entry,
            live_size=live_size,
            backtest_size=backtest_size,
            live_stop_loss=live_sl,
            backtest_stop_loss=backtest_sl,
            live_take_profit=live_tp,
            backtest_take_profit=backtest_tp,
            entry_price_diff=entry_price_diff,
            size_diff=size_diff,
            stop_loss_diff=stop_loss_diff,
            take_profit_diff=take_profit_diff,
            severity=severity,
            details={
                'live_risk_amount': live_pos.get('risk_amount') if live_pos else None,
                'backtest_risk_amount': backtest_pos.get('risk_amount') if backtest_pos else None
            }
        )
    
    def _compare_pnl(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        report: ComparisonReport
    ):
        """Compare P&L calculations between live and backtest."""
        live_trades = live_data.get('trade_history', [])
        backtest_trades = backtest_data.get('trade_history', [])
        
        logger.debug(f"Comparing {len(live_trades)} live trades with {len(backtest_trades)} backtest trades")
        
        # Index by symbol and timestamp
        live_by_key = {}
        for trade in live_trades:
            key = (trade.get('symbol'), trade.get('timestamp'))
            live_by_key[key] = trade
        
        backtest_by_key = {}
        for trade in backtest_trades:
            key = (trade.get('symbol'), trade.get('timestamp'))
            backtest_by_key[key] = trade
        
        # Compare all trades
        all_keys = set(live_by_key.keys()) | set(backtest_by_key.keys())
        
        for key in all_keys:
            live_trade = live_by_key.get(key)
            backtest_trade = backtest_by_key.get(key)
            
            comparison = self._create_pnl_comparison(live_trade, backtest_trade, key)
            report.add_pnl_comparison(comparison)
    
    def _create_pnl_comparison(
        self,
        live_trade: Optional[Dict],
        backtest_trade: Optional[Dict],
        key: Tuple
    ) -> PnLComparison:
        """Create a P&L comparison."""
        symbol, timestamp_str = key
        
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str or datetime.now()
        
        trade_id = f"{symbol}_{timestamp.isoformat()}"
        
        live_pnl = live_trade.get('pnl', 0.0) if live_trade else 0.0
        backtest_pnl = backtest_trade.get('pnl', 0.0) if backtest_trade else 0.0
        
        live_fees = live_trade.get('fees', 0.0) if live_trade else 0.0
        backtest_fees = backtest_trade.get('fees', 0.0) if backtest_trade else 0.0
        
        # Calculate differences
        realized_pnl_diff = abs(live_pnl - backtest_pnl)
        fees_diff = abs(live_fees - backtest_fees)
        total_pnl_diff = realized_pnl_diff + fees_diff
        
        # Determine severity based on P&L difference relative to trade size
        avg_pnl = (abs(live_pnl) + abs(backtest_pnl)) / 2 if (abs(live_pnl) + abs(backtest_pnl)) > 0 else 1.0
        pnl_relative_diff = realized_pnl_diff / avg_pnl if avg_pnl > 0 else 0.0
        
        if pnl_relative_diff > self.tolerance.critical_threshold:
            severity = DivergenceSeverity.CRITICAL
        elif pnl_relative_diff > self.tolerance.moderate_threshold:
            severity = DivergenceSeverity.MODERATE
        elif pnl_relative_diff > self.tolerance.minor_threshold:
            severity = DivergenceSeverity.MINOR
        else:
            severity = DivergenceSeverity.NONE
        
        return PnLComparison(
            timestamp=timestamp,
            symbol=symbol,
            trade_id=trade_id,
            live_realized_pnl=live_pnl,
            backtest_realized_pnl=backtest_pnl,
            live_fees=live_fees,
            backtest_fees=backtest_fees,
            realized_pnl_diff=realized_pnl_diff,
            unrealized_pnl_diff=0.0,  # Not applicable for closed trades
            fees_diff=fees_diff,
            total_pnl_diff=total_pnl_diff,
            severity=severity,
            details={
                'live_entry': live_trade.get('entry_price') if live_trade else None,
                'live_exit': live_trade.get('exit_price') if live_trade else None,
                'backtest_entry': backtest_trade.get('entry_price') if backtest_trade else None,
                'backtest_exit': backtest_trade.get('exit_price') if backtest_trade else None
            }
        )
    
    def _compare_risk_checks(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        report: ComparisonReport
    ):
        """Compare risk management decisions between live and backtest."""
        live_checks = live_data.get('risk_checks', [])
        backtest_checks = backtest_data.get('risk_checks', [])
        
        logger.debug(f"Comparing {len(live_checks)} live risk checks with {len(backtest_checks)} backtest risk checks")
        
        # Index by symbol and timestamp
        live_by_key = {}
        for check in live_checks:
            key = (check.get('symbol'), check.get('timestamp'), check.get('check_type'))
            live_by_key[key] = check
        
        backtest_by_key = {}
        for check in backtest_checks:
            key = (check.get('symbol'), check.get('timestamp'), check.get('check_type'))
            backtest_by_key[key] = check
        
        # Compare all checks
        all_keys = set(live_by_key.keys()) | set(backtest_by_key.keys())
        
        for key in all_keys:
            live_check = live_by_key.get(key)
            backtest_check = backtest_by_key.get(key)
            
            comparison = self._create_risk_comparison(live_check, backtest_check, key)
            report.add_risk_comparison(comparison)
    
    def _create_risk_comparison(
        self,
        live_check: Optional[Dict],
        backtest_check: Optional[Dict],
        key: Tuple
    ) -> RiskComparison:
        """Create a risk comparison."""
        symbol, timestamp_str, check_type = key
        
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str or datetime.now()
        
        live_allowed = live_check.get('allowed', False) if live_check else False
        backtest_allowed = backtest_check.get('allowed', False) if backtest_check else False
        
        live_risk = live_check.get('risk_amount', 0.0) if live_check else 0.0
        backtest_risk = backtest_check.get('risk_amount', 0.0) if backtest_check else 0.0
        
        live_size = live_check.get('position_size', 0.0) if live_check else 0.0
        backtest_size = backtest_check.get('position_size', 0.0) if backtest_check else 0.0
        
        # Calculate differences
        decision_match = live_allowed == backtest_allowed
        risk_amount_diff = abs(live_risk - backtest_risk)
        position_size_diff = abs(live_size - backtest_size)
        
        live_reason = live_check.get('reason', '') if live_check else ''
        backtest_reason = backtest_check.get('reason', '') if backtest_check else ''
        
        # Determine severity
        if not decision_match:
            severity = DivergenceSeverity.CRITICAL
        elif position_size_diff > self.tolerance.position_size_tolerance:
            severity = DivergenceSeverity.MODERATE
        elif risk_amount_diff > self.tolerance.risk_amount_tolerance:
            severity = DivergenceSeverity.MINOR
        else:
            severity = DivergenceSeverity.NONE
        
        return RiskComparison(
            timestamp=timestamp,
            symbol=symbol,
            check_type=check_type,
            live_allowed=live_allowed,
            backtest_allowed=backtest_allowed,
            live_risk_amount=live_risk,
            backtest_risk_amount=backtest_risk,
            live_position_size=live_size,
            backtest_position_size=backtest_size,
            decision_match=decision_match,
            risk_amount_diff=risk_amount_diff,
            position_size_diff=position_size_diff,
            live_reason=live_reason,
            backtest_reason=backtest_reason,
            severity=severity,
            details={
                'live_exposure': live_check.get('current_exposure') if live_check else None,
                'backtest_exposure': backtest_check.get('current_exposure') if backtest_check else None
            }
        )
    
    def _compare_timing(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        report: ComparisonReport
    ):
        """Compare timing between live and backtest."""
        # Compare signal timing
        live_signals = live_data.get('signals', [])
        backtest_signals = backtest_data.get('signals', [])
        
        for live_sig in live_signals:
            # Find matching backtest signal
            for backtest_sig in backtest_signals:
                if (live_sig.get('symbol') == backtest_sig.get('symbol') and
                    live_sig.get('signal_type') == backtest_sig.get('signal_type')):
                    
                    live_time = datetime.fromisoformat(live_sig['timestamp'].replace('Z', '+00:00'))
                    backtest_time = datetime.fromisoformat(backtest_sig['timestamp'].replace('Z', '+00:00'))
                    
                    time_diff = abs((live_time - backtest_time).total_seconds())
                    
                    if time_diff > self.tolerance.timing_tolerance_seconds:
                        severity = DivergenceSeverity.MINOR if time_diff < 5.0 else DivergenceSeverity.MODERATE
                        
                        comparison = TimingComparison(
                            timestamp=live_time,
                            event_type='signal',
                            live_timestamp=live_time,
                            backtest_timestamp=backtest_time,
                            time_diff_seconds=time_diff,
                            severity=severity,
                            details={'symbol': live_sig.get('symbol')}
                        )
                        report.add_timing_comparison(comparison)
                    break
    
    def _compare_state_transitions(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        report: ComparisonReport
    ):
        """Compare state transitions between live and backtest."""
        # This is a simplified comparison - in practice you'd track
        # actual state machine transitions
        live_states = live_data.get('states', [])
        backtest_states = backtest_data.get('states', [])
        
        # Compare number of positions over time
        live_position_counts = [s.get('num_positions', 0) for s in live_states]
        backtest_position_counts = [s.get('num_positions', 0) for s in backtest_states]
        
        # Check if position count transitions match
        min_len = min(len(live_position_counts), len(backtest_position_counts))
        for i in range(min_len):
            if live_position_counts[i] != backtest_position_counts[i]:
                live_time = datetime.fromisoformat(live_states[i]['timestamp'].replace('Z', '+00:00'))
                
                comparison = StateTransitionComparison(
                    timestamp=live_time,
                    transition_type='position_count',
                    live_from_state=str(live_position_counts[i-1] if i > 0 else 0),
                    live_to_state=str(live_position_counts[i]),
                    backtest_from_state=str(backtest_position_counts[i-1] if i > 0 else 0),
                    backtest_to_state=str(backtest_position_counts[i]),
                    transition_match=False,
                    severity=DivergenceSeverity.MINOR,
                    details={'index': i}
                )
                report.add_state_transition(comparison)
    
    def save_report(
        self,
        report: ComparisonReport,
        output_dir: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Save comparison report to files.
        
        Args:
            report: ComparisonReport to save
            output_dir: Directory to save reports (uses self.output_dir if None)
            
        Returns:
            Tuple of (json_path, html_path)
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"validation_report_{report.report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        json_path = out_dir / f"{base_name}.json"
        html_path = out_dir / f"{base_name}.html"
        
        report.to_json(str(json_path))
        report.generate_html(str(html_path))
        
        logger.info(f"Report saved: {json_path}, {html_path}")
        
        return str(json_path), str(html_path)
    
    def validate_fidelity(
        self,
        live_data: Dict[str, Any],
        backtest_data: Dict[str, Any],
        symbols: List[str],
        max_divergence_rate: float = 0.05,
        max_critical_divergences: int = 0
    ) -> Tuple[bool, ComparisonReport]:
        """
        Validate backtest fidelity against live data.
        
        Args:
            live_data: Captured live trading data
            backtest_data: Backtest results
            symbols: List of symbols tested
            max_divergence_rate: Maximum acceptable divergence rate
            max_critical_divergences: Maximum acceptable critical divergences
            
        Returns:
            Tuple of (passed, report)
        """
        report = self.compare_results(live_data, backtest_data, symbols)
        
        # Calculate overall divergence rate
        total_comparisons = sum(m.total_comparisons for m in report.metrics.values())
        total_divergences = sum(m.divergences for m in report.metrics.values())
        divergence_rate = total_divergences / total_comparisons if total_comparisons > 0 else 0.0
        
        # Count critical divergences
        critical_count = sum(m.critical_divergences for m in report.metrics.values())
        
        # Determine if passed
        passed = (
            divergence_rate <= max_divergence_rate and
            critical_count <= max_critical_divergences
        )
        
        report.passed_validation = passed
        
        logger.info(
            f"Fidelity validation: {'PASSED' if passed else 'FAILED'} "
            f"(divergence_rate={divergence_rate:.2%}, critical={critical_count})"
        )
        
        return passed, report
