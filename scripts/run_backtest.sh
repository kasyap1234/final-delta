#!/bin/bash
#
# Quick Start Script for Running Backtests
#
# This script automates the process of:
# 1. Checking if historical data exists
# 2. Fetching historical data if needed
# 3. Running the backtest
# 4. Generating reports
#
# Usage:
#   ./scripts/run_backtest.sh [options]
#
# Options:
#   --force-fetch    Force re-fetching historical data even if it exists
#   --config FILE    Use custom backtest config file (default: config/backtest.yaml)
#   --verbose        Enable verbose logging
#   --help           Show this help message
#

set -e  # Exit on error

# Default values
FORCE_FETCH=false
CONFIG_FILE="config/backtest.yaml"
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-fetch)
            FORCE_FETCH=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --force-fetch    Force re-fetching historical data even if it exists"
            echo "  --config FILE    Use custom backtest config file (default: config/backtest.yaml)"
            echo "  --verbose        Enable verbose logging"
            echo "  --help           Show this help message"
            echo ""
            echo "Example:"
            echo "  $0                    # Run backtest with default settings"
            echo "  $0 --force-fetch      # Force re-fetch data and run backtest"
            echo "  $0 --config my_config.yaml --verbose"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

print_header "Delta Trading Bot - Backtest Quick Start"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

print_info "Python version: $(python3 --version)"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

print_info "Using config file: $CONFIG_FILE"

# Extract symbols and timeframe from config (using grep and sed)
SYMBOLS=$(grep -A 10 "^backtest:" "$CONFIG_FILE" | grep -A 5 "symbols:" | grep -E '^\s*-' | sed 's/.*"\(.*\)".*/\1/' | tr '\n' ',' | sed 's/,$//')
TIMEFRAME=$(grep -A 10 "^backtest:" "$CONFIG_FILE" | grep "timeframe:" | sed 's/.*: "\(.*\)".*/\1/')

if [ -z "$SYMBOLS" ]; then
    print_warning "Could not extract symbols from config, using defaults"
    SYMBOLS="BTC/USD,ETH/USD,SOL/USD"
fi

if [ -z "$TIMEFRAME" ]; then
    print_warning "Could not extract timeframe from config, using default"
    TIMEFRAME="15m"
fi

print_info "Symbols: $SYMBOLS"
print_info "Timeframe: $TIMEFRAME"

# Check if historical data exists
print_header "Checking Historical Data"

DATA_DIR="data/backtest"
DATA_EXISTS=true

for SYMBOL in ${SYMBOLS//,/ }; do
    # Convert symbol to filename format (BTC/USD -> BTC_USD_15m.csv)
    FILENAME="${SYMBOL//\/_}_${TIMEFRAME}.csv"
    FILEPATH="$DATA_DIR/$FILENAME"
    
    if [ -f "$FILEPATH" ]; then
        print_success "Data file exists: $FILENAME"
    else
        print_warning "Data file missing: $FILENAME"
        DATA_EXISTS=false
    fi
done

# Fetch data if needed
if [ "$FORCE_FETCH" = true ] || [ "$DATA_EXISTS" = false ]; then
    if [ "$FORCE_FETCH" = true ]; then
        print_info "Force fetch enabled, re-fetching data..."
    else
        print_info "Missing historical data, fetching..."
    fi
    
    print_header "Fetching Historical Data"
    
    # Run the data fetcher
    FETCH_CMD="python3 scripts/fetch_historical_data.py --config $CONFIG_FILE"
    
    if [ "$VERBOSE" = true ]; then
        FETCH_CMD="$FETCH_CMD --verbose"
    fi
    
    if [ "$FORCE_FETCH" = true ]; then
        FETCH_CMD="$FETCH_CMD --force"
    fi
    
    print_info "Running: $FETCH_CMD"
    
    if $FETCH_CMD; then
        print_success "Historical data fetched successfully"
    else
        print_error "Failed to fetch historical data"
        exit 1
    fi
else
    print_success "All historical data files exist, skipping fetch"
fi

# Run backtest
print_header "Running Backtest"

BACKTEST_CMD="python3 backtest_main.py --config $CONFIG_FILE"

if [ "$VERBOSE" = true ]; then
    BACKTEST_CMD="$BACKTEST_CMD --verbose"
fi

print_info "Running: $BACKTEST_CMD"
echo ""

if $BACKTEST_CMD; then
    print_success "Backtest completed successfully"
else
    print_error "Backtest failed"
    exit 1
fi

# Check if reports were generated
print_header "Backtest Results"

OUTPUT_DIR=$(grep -A 10 "^backtest:" "$CONFIG_FILE" | grep "output_dir:" | sed 's/.*: "\(.*\)".*/\1/')

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="backtest_results"
fi

if [ -d "$OUTPUT_DIR" ]; then
    print_success "Reports generated in: $OUTPUT_DIR"
    echo ""
    print_info "Generated files:"
    ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
else
    print_warning "Output directory not found: $OUTPUT_DIR"
fi

print_header "Done!"
print_info "Backtest completed successfully"
print_info "Review the results in the output directory"

exit 0
