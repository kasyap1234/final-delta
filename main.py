#!/usr/bin/env python3
"""
Main entry point for the cryptocurrency trading bot.

This script initializes and runs the trading bot with command-line
configuration options and proper signal handling.

Usage:
    python main.py --config config/config.yaml
    python main.py --config config/config.yaml --sandbox
    python main.py --config config/config.yaml --symbols BTC/USDT,ETH/USDT
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, TradingBotConfig
from src.bot import TradingBot
from src.utils import get_logger, LogCategory

logger = get_logger(__name__)


class BotRunner:
    """
    Manages the trading bot lifecycle and handles signals.
    """
    
    def __init__(self):
        self.bot: Optional[TradingBot] = None
        self._shutdown_event = asyncio.Event()
        self._running = False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        try:
            loop = asyncio.get_event_loop()
            
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_handler)
                
            logger.info("Signal handlers registered")
            
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning("Signal handlers not supported on this platform")
    
    def _signal_handler(self):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received, stopping bot...")
        self._shutdown_event.set()
        if self.bot:
            asyncio.create_task(self.bot.stop())
    
    async def run(self, config_path: str, sandbox: bool = False, symbols: Optional[list] = None):
        """
        Run the trading bot.
        
        Args:
            config_path: Path to configuration file
            sandbox: Whether to run in sandbox mode
            symbols: Optional list of symbols to override config
        """
        try:
            # Load configuration
            logger.info(f"Loading configuration from {config_path}")
            config = load_config(config_path)
            
            # Override with command line options
            if sandbox:
                config.exchange.sandbox = True
                logger.info("Running in SANDBOX mode")
            
            if symbols:
                config.trading.symbols = symbols
                logger.info(f"Using symbols from command line: {symbols}")
            
            # Create and initialize bot
            self.bot = TradingBot(config)
            
            logger.info("Initializing trading bot...")
            if not await self.bot.initialize():
                logger.error("Failed to initialize bot")
                return 1
            
            # Start the bot
            logger.info("Starting trading bot...")
            if not await self.bot.start():
                logger.error("Failed to start bot")
                return 1
            
            self._running = True
            
            # Wait for shutdown signal
            logger.info("Bot is running. Press Ctrl+C to stop.")
            await self._shutdown_event.wait()
            
            return 0
            
        except Exception as e:
            logger.error(f"Error running bot: {e}", exc_info=True)
            return 1
        finally:
            self._running = False
            if self.bot:
                await self.bot.stop()
    
    def get_status(self) -> dict:
        """Get current bot status."""
        if self.bot:
            return self.bot.get_status()
        return {'running': False, 'initialized': False}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Trading Bot for Delta Exchange India',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/config.yaml
  python main.py --config config/config.yaml --sandbox
  python main.py --config config/config.yaml --symbols BTC/USDT,ETH/USDT
  python main.py --config config/config.yaml --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--sandbox', '-s',
        action='store_true',
        help='Run in sandbox mode (no real trades)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of trading symbols (overrides config)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Create runner
    runner = BotRunner()
    runner.setup_signal_handlers()
    
    # Run bot
    exit_code = await runner.run(
        config_path=args.config,
        sandbox=args.sandbox,
        symbols=symbols
    )
    
    return exit_code


if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
