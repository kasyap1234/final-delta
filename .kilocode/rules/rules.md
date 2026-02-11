Strategy Consistency Rule
## The backtesting system and the live trading bot must always remain logically consistent.## 

The backtest exists to validate and measure the expected performance of the live bot. Therefore:

Both must use the same strategy logic.

Any change to the live trading strategy must first be implemented and verified in the backtest.

Backtest results should be treated as the primary performance reference for the live bot.

Divergence between backtest and live strategy behavior is not allowed.


## always write optimised code , avoid simple for loops , use polars instead of pandas ##

