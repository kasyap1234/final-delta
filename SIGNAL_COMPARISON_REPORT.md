# Signal Detection Comparison Report

## Summary

This report compares the performance of **baseline continuous signal generation** vs **event-based signal detection** on available 2026 data (January 1-28, 2026).

**Important Note**: The requested 2023 and 2024 data is not available in the system. Only 2026 data exists in `data/backtest/`. The comparison below uses the full available 2026 dataset (Jan 1-28, 2026).

---

## Comparison Table

| Period | Signal Type | Total Return | Total Trades | Win Rate | Sharpe | Max DD | Total Profit |
|--------|-------------|--------------|--------------|----------|--------|--------|--------------|
| 2026-01-01 to 2026-01-28 | Baseline (Continuous) | 26.54% | 50 | 36.00% | 3.33 | 18.39% | $2,654.00 |
| 2026-01-01 to 2026-01-28 | Event-Based (EMA Crossover) | 33.55% | 53 | 45.28% | 4.03 | 18.97% | $3,354.90 |

---

## Detailed Results

### Baseline (Continuous Signal Generation)
- **Period**: January 1-28, 2026
- **Initial Balance**: $10,000.00
- **Final Equity**: $12,654.00
- **Total Return**: 26.54%
- **Sharpe Ratio**: 3.33
- **Maximum Drawdown**: 18.39%
- **Win Rate**: 36.00%
- **Total Trades**: 50
- **Total Fees**: $179.51

### Event-Based Signal Detection (EMA Crossover Only)
- **Period**: January 1-28, 2026
- **Initial Balance**: $10,000.00
- **Final Equity**: $13,354.90
- **Total Return**: 33.55%
- **Sharpe Ratio**: 4.03
- **Maximum Drawdown**: 18.97%
- **Win Rate**: 45.28%
- **Total Trades**: 53
- **Total Fees**: $190.70

---

## Key Improvements with Event-Based Signals

| Metric | Baseline | Event-Based | Improvement |
|--------|----------|-------------|-------------|
| Total Return | 26.54% | 33.55% | +7.01% |
| Sharpe Ratio | 3.33 | 4.03 | +21.0% |
| Win Rate | 36.00% | 45.28% | +9.28% |
| Total Profit | $2,654.00 | $3,354.90 | +$700.90 |
| Total Trades | 50 | 53 | +3 |
| Max Drawdown | 18.39% | 18.97% | +0.58% (worse) |

---

## Analysis

### What Changed
- **Baseline**: Generated signals continuously whenever conditions were met (close > ema_trend AND ema_short > ema_medium AND rsi < overbought)
- **Event-Based**: Only generates signals on EMA crossover events (when EMA9 crosses above/below EMA21)

### Performance Impact
1. **Return Improvement**: Event-based signals produced +7.01% higher returns
2. **Better Win Rate**: +9.28% improvement in win rate (45.28% vs 36.00%)
3. **Higher Sharpe**: Better risk-adjusted returns (4.03 vs 3.33)
4. **Similar Drawdown**: Slightly higher max drawdown (+0.58%) but comparable
5. **Trade Frequency**: Similar number of trades (53 vs 50)

---

## Recommendation

### Should We Adopt Event-Based Signals?

**YES** - Based on the available data, event-based signal detection shows:

1. **Improved Returns**: +7.01% higher total return
2. **Better Quality Trades**: Higher win rate (+9.28%) indicates better entry timing
3. **Better Risk-Adjusted Performance**: Sharpe ratio improved by 21%
4. **Similar Trade Frequency**: Not missing significant opportunities
5. **Comparable Drawdown**: Risk profile remains similar

### Limitations
- Only tested on 28 days of 2026 data (not full 2023/2024 years as requested)
- Cannot validate consistency across multiple years without additional historical data
- Single market condition (bullish crypto market in Jan 2026)

### Suggested Next Steps
1. **Obtain 2023-2024 data** to perform the originally requested multi-year validation
2. **Monitor live performance** if deploying event-based signals
3. **Consider hybrid approach** that combines both methods for different market conditions

---

## Files Generated

- `backtest_results_2026_baseline/` - Baseline continuous signal results
- `backtest_results_2026_event/` - Event-based signal results
- `config/backtest_2026_baseline.yaml` - Baseline configuration
- `config/backtest_2026_event.yaml` - Event-based configuration
- `src/backtest/strategy_engine.py` - Currently has event-based implementation

---

## Conclusion

Event-based signal detection shows promising improvements over continuous signal generation in the available test period. The 7% return improvement and 9% win rate increase suggest better entry timing and trade quality. However, validation on additional historical data (2023-2024) would provide more confidence in the consistency of these improvements across different market conditions.
