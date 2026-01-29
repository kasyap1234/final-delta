## Stage 1: Fee Accounting Parity
**Goal**: Ensure backtest fee totals are computed and deducted consistently for entry/exit trades.
**Success Criteria**: `total_fees` in reports is non-zero when trades exist; account state reflects fee deductions.
**Tests**: Run a short backtest and confirm fees in JSON report; run existing test suite if available.
**Status**: Complete

## Stage 2: Market Regime Filters
**Goal**: Reduce choppy-market trades by adding ADX/EMA-spread filters and stronger signal gating.
**Success Criteria**: Trades are suppressed in low-trend regimes; strategy still trades in trending periods.
**Tests**: Run a 2025 backtest sample and inspect trade count and win rate.
**Status**: Complete

## Stage 3: Limit-Order Fee Assumptions
**Goal**: Apply maker/limit-order fee assumptions for both entry and exit calculations.
**Success Criteria**: Limit order fees are used for entry/exit calculations and in portfolio tracking.
**Tests**: Backtest report shows fees consistent with maker rates.
**Status**: Complete
