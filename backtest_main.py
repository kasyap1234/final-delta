#!/usr/bin/env python3
"""
Main entry point for running backtests.

This script provides a command-line interface for running
backtests on historical data.
"""

import asyncio
import argparse
import logging
from pathlib import Path

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


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
