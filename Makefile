# Delta Exchange India Trading Bot - Makefile
# Common tasks for development and deployment

.PHONY: help install install-dev test lint format clean run sandbox shell db-clear

# Default target
help:
	@echo "Delta Trading Bot - Available Commands:"
	@echo ""
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run linters (flake8, mypy)"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean generated files"
	@echo "  make run          - Run the trading bot"
	@echo "  make sandbox      - Run bot in sandbox mode"
	@echo "  make shell        - Open Python shell with bot modules"
	@echo "  make setup        - Initial project setup"
	@echo "  make db-init      - Initialize database"
	@echo "  make db-clear     - Clear/delete SQLite database (for debugging)"
	@echo "  make db-backup    - Backup database"
	@echo "  make logs         - View recent logs"
	@echo "  make status       - Check bot status"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-asyncio black flake8 mypy

setup:
	@echo "Setting up Delta Trading Bot..."
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate     (Windows)"
	@echo ""
	@echo "Then run: make install"
	@echo ""
	@echo "Don't forget to:"
	@echo "  1. Copy .env.example to .env and add your API keys"
	@echo "  2. Copy config/config.example.yaml to config/config.yaml"

# Testing
test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code Quality
lint:
	flake8 src/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100

format-check:
	black src/ tests/ --line-length=100 --check

# Running the bot
run:
	python main.py --config config/config.yaml

sandbox:
	python main.py --config config/config.yaml --sandbox

debug:
	python main.py --config config/config.yaml --log-level DEBUG

# Database
db-init:
	@echo "Initializing database..."
	python -c "from src.database import DatabaseManager; import asyncio; asyncio.run(DatabaseManager('data/trading_bot.db').initialize())"

db-backup:
	@echo "Backing up database..."
	@mkdir -p backups
	@cp data/trading_bot.db backups/trading_bot_$$(date +%Y%m%d_%H%M%S).db
	@echo "Backup created in backups/"

db-query:
	@echo "Opening database shell..."
	sqlite3 data/trading_bot.db

db-clear:
	@echo "Clearing SQLite database..."
	@rm -f data/test_trading_bot.db
	@rm -f data/trading_bot.db
	@echo "Database cleared. Run 'make db-init' to reinitialize."

# Logs
logs:
	tail -f logs/trading_bot.log

logs-error:
	grep "ERROR" logs/trading_bot.log | tail -20

logs-trades:
	grep -E "(Position opened|Position closed|Trade executed)" logs/trading_bot.log | tail -20

# Maintenance
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true
	@echo "Cleaned up generated files"

clean-all: clean
	rm -rf venv/ .env config/config.yaml data/*.db logs/*.log
	@echo "Cleaned all generated and config files"

# Development shell
shell:
	python -c "import sys; sys.path.insert(0, '.'); from src.config import load_config; from src.exchange import ExchangeClient; import asyncio; print('Bot modules loaded. Use asyncio.run() for async operations.')" -i

# Docker (if using Docker)
docker-build:
	docker build -t delta-trading-bot .

docker-run:
	docker run -it --rm --env-file .env -v $$(pwd)/config:/app/config delta-trading-bot

# Git hooks
git-hooks:
	@echo "Installing git hooks..."
	@echo '#!/bin/bash\nmake lint\nmake test' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Git hooks installed"

# Performance report
report:
	@echo "Generating performance report..."
	@python scripts/generate_report.py

# Health check
status:
	@echo "Checking bot status..."
	@if pgrep -f "python main.py" > /dev/null; then \
		echo "Bot is running"; \
	else \
		echo "Bot is not running"; \
	fi
	@echo ""
	@echo "Recent log entries:"
	@tail -5 logs/trading_bot.log 2>/dev/null || echo "No log file found"
	@echo ""
	@echo "Database size:"
	@ls -lh data/trading_bot.db 2>/dev/null || echo "No database found"
