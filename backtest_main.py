#!/usr/bin/env python3
"""
Main entry point for running backtests.

This script provides a command-line interface for running
backtests on historical data.
"""

import asyncio
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestEngine
from src.backtest.trade_logger import TradeLogger
from src.backtest.equity_curve_generator import EquityCurveGenerator
from src.backtest.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for running backtests."""
    parser = argparse.ArgumentParser(
        description='Run backtests for the trading bot'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/backtest.yaml',
        help='Path to backtest configuration file (default: config/backtest.yaml)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--validation-mode',
        action='store_true',
        help='Run in validation mode with comparable output format'
    )
    parser.add_argument(
        '--validation-output',
        type=str,
        default='validation_backtest.json',
        help='Output file for validation mode (default: validation_backtest.json)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = BacktestConfig.from_yaml(args.config)
        
        # Validate configuration
        config.validate()
        
        # Initialize backtest engine
        logger.info("Initializing backtest engine...")
        engine = BacktestEngine(config)
        
        # Run backtest
        logger.info("Starting backtest...")
        results = await engine.run()
        
        # Print summary
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Total Fees: ${results['total_fees']:.2f}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Initial Balance: ${results['initial_balance']:.2f}")
        print("=" * 60)
        
        # Generate reports if configured
        if config.save_trade_log or config.save_equity_curve or config.generate_report:
            logger.info("Generating reports...")
            
            # Initialize output directory
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export trade log
            if config.save_trade_log and results.get('trade_history'):
                trade_logger = TradeLogger(config.output_dir)
                trade_logger.export_to_csv(results['trade_history'])
                trade_logger.export_to_json(results['trade_history'])
                trade_logger.export_summary(results['trade_history'])
                logger.info("Trade logs exported")
            
            # Export equity curve
            if config.save_equity_curve and results.get('equity_curve'):
                equity_generator = EquityCurveGenerator(config.output_dir)
                equity_generator.export_to_csv(results['equity_curve'])
                equity_generator.export_to_json(results['equity_curve'])
                equity_generator.export_drawdown_curve(results['equity_curve'])
                equity_generator.export_returns_curve(results['equity_curve'])
                equity_generator.export_summary(results['equity_curve'])
                logger.info("Equity curve exported")
            
            # Generate report
            if config.generate_report:
                report_generator = ReportGenerator(config.output_dir)
                report_generator.generate_html_report(results)
                report_generator.generate_json_report(results)
                report_generator.generate_text_report(results)
                logger.info("Reports generated")
            
            print(f"\nReports saved to: {output_dir.absolute()}")
        
        # Validation mode output
        if args.validation_mode:
            logger.info("Exporting validation mode output...")
            validation_output = generate_validation_output(results, engine)
            
            output_path = Path(args.validation_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(validation_output, f, indent=2, default=str)
            
            logger.info(f"Validation output saved to: {output_path}")
        
        logger.info("Backtest completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1
    
    return 0


def generate_validation_output(
    results: Dict[str, Any],
    engine: BacktestEngine
) -> Dict[str, Any]:
    """
    Generate validation mode output with all state transitions.
    
    Args:
        results: Backtest results
        engine: Backtest engine instance
        
    Returns:
        Dictionary with comparable data format
    """
    # Get detailed state from engine
    engine_stats = engine.get_stats()
    
    # Build state transitions
    state_transitions = []
    equity_curve = results.get('equity_curve', [])
    
    for i, point in enumerate(equity_curve):
        transition = {
            'timestamp': point.get('timestamp'),
            'state_type': 'equity_point',
            'equity': point.get('equity'),
            'balance': point.get('balance'),
            'free_balance': point.get('free_balance'),
            'used_balance': point.get('used_balance'),
            'unrealized_pnl': point.get('unrealized_pnl'),
            'realized_pnl': point.get('realized_pnl'),
            'num_positions': point.get('num_positions'),
            'total_risk': point.get('total_risk'),
            'sequence': i
        }
        state_transitions.append(transition)
    
    # Build signals from trade history
    signals = []
    for trade in results.get('trade_history', []):
        signal = {
            'timestamp': trade.get('timestamp'),
            'symbol': trade.get('symbol'),
            'signal_type': trade.get('side'),
            'strength': 1.0,  # Backtest signals are executed at full strength
            'price': trade.get('entry_price'),
            'reason': trade.get('reason', 'entry'),
            'details': {
                'exit_price': trade.get('exit_price'),
                'pnl': trade.get('pnl')
            }
        }
        signals.append(signal)
    
    # Build orders from trade history
    orders = []
    for trade in results.get('trade_history', []):
        # Entry order
        entry_order = {
            'timestamp': trade.get('timestamp'),
            'order_id': f"entry_{trade.get('symbol')}_{trade.get('timestamp')}",
            'symbol': trade.get('symbol'),
            'side': trade.get('side'),
            'order_type': 'market',
            'size': trade.get('size'),
            'price': trade.get('entry_price'),
            'filled_size': trade.get('size'),
            'fill_price': trade.get('entry_price'),
            'fees': 0.0,
            'status': 'filled',
            'slippage': 0.0
        }
        orders.append(entry_order)
    
    # Build positions
    positions = []
    for trade in results.get('trade_history', []):
        position = {
            'timestamp': trade.get('timestamp'),
            'position_id': f"pos_{trade.get('symbol')}_{trade.get('timestamp')}",
            'symbol': trade.get('symbol'),
            'side': trade.get('side'),
            'entry_price': trade.get('entry_price'),
            'current_price': trade.get('exit_price', trade.get('entry_price')),
            'size': trade.get('size'),
            'stop_loss': trade.get('stop_loss'),
            'take_profit': trade.get('take_profit'),
            'realized_pnl': trade.get('pnl', 0.0),
            'entry_time': trade.get('timestamp'),
            'exit_time': trade.get('exit_timestamp'),
            'exit_price': trade.get('exit_price'),
            'exit_reason': trade.get('reason')
        }
        positions.append(position)
    
    # Build risk checks from state summary
    risk_checks = []
    state_summary = engine_stats.get('state_manager', {})
    if state_summary:
        risk_check = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'portfolio',
            'check_type': 'summary',
            'allowed': True,
            'position_size': 0.0,
            'risk_amount': state_summary.get('total_risk', 0.0),
            'risk_percent': 0.0,
            'current_exposure': state_summary.get('total_exposure', 0.0),
            'current_positions': state_summary.get('num_positions', 0)
        }
        risk_checks.append(risk_check)
    
    # Build hedges (if available)
    hedges = results.get('hedges', [])
    
    # Construct final validation output
    validation_output = {
        'validation_timestamp': datetime.now().isoformat(),
        'total_return': results.get('total_return', 0.0),
        'win_rate': results.get('win_rate', 0.0),
        'total_trades': results.get('total_trades', 0),
        'final_equity': results.get('final_equity', 0.0),
        'initial_balance': results.get('initial_balance', 0.0),
        'equity_curve': equity_curve,
        'trade_history': results.get('trade_history', []),
        'signals': signals,
        'orders': orders,
        'positions': positions,
        'risk_checks': risk_checks,
        'hedges': hedges,
        'state_transitions': state_transitions,
        'engine_stats': engine_stats,
        'performance': results.get('performance', {}),
        'performance_summary': results.get('performance_summary', {}),
        'risk_summary': results.get('risk_summary', {}),
        'risk_report': results.get('risk_report', {}),
        'state_summary': results.get('state_summary', {})
    }
    
    return validation_output


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
