# Crypto Quant Research Platform

A production-grade, automated market data pipeline and backtesting engine for crypto assets. Built to rigorously test trading hypotheses while preventing look-ahead bias and overfitting.

## Tech Stack
- **Data Engineering:** Polars, DuckDB, Parquet, CCXT
- **Backtesting & Analytics:** VectorBT, Pandas, Pytest
- **CI/CD & Tooling:** GitHub Actions, uv, Pytest

## 📊 Research Case Study: Time-Series Momentum (Moving Average Crossover)

### 1. Hypothesis
A dual Moving Average Crossover (Fast/Slow) strategy can capture time-series momentum in BTC/USDT hourly markets and generate alpha net of fees and slippage.

### 2. Data Pipeline & Quality Gates
- Fetched 30 days of 1H OHLCV data from Binance via CCXT.
- Stored in Parquet format for optimized I/O.
- Implemented `pytest` data quality gates to ensure OHLC logical consistency and time-series continuity (no missing candles).

### 3. In-Sample Optimization (The Trap)
A grid search was performed over 24 parameter combinations. 
**Top Result:** Fast=25, Slow=40.
- **Sharpe Ratio:** 3.68
- **Total Return:** 12.53%
- **Win Rate:** ~19% (Classic trend-following profile: many small losses, few massive wins).

### 4. Out-of-Sample Walk-Forward Validation (The Reality Check)
To test for overfitting, the strategy was locked and tested blindly on the final 20% of the data (Days 25-30).
**Result:** 
- **Total Return:** -2.65%
- **Sharpe Ratio:** -15.56

### 5. Conclusion & Next Steps
The strategy failed catastrophically out-of-sample. 
**Why?** Moving averages are highly susceptible to "whipsaws" in sideways, ranging markets. The In-Sample data likely featured strong directional trends, while the OOS data featured mean-reverting chop. 
**Future Research:** 
1. Add a regime filter (e.g., ADX or Volatility threshold) to only trade when a trend is mathematically confirmed.
2. Pivot to Crypto-Native signals (e.g., Funding Rate mean reversion or Order Book Imbalance) which are less reliant on traditional price-action momentum.




## 📊 Research Case Study 2: Derivatives Market Structure (Funding Rate Mean Reversion)

### 1. Hypothesis
Extreme perpetual funding rates indicate crowded leverage. Shorting extreme positive funding (Z > 1.5) and buying extreme negative funding (Z < -1.5) should capture mean-reversion flushes.

### 2. Data Engineering Challenge (Time-Series Alignment)
Funding rates print every 8 hours, while spot prices print every 1 hour. To prevent look-ahead bias and missing data, we aligned the datasets using Pandas `ffill()` (Forward Fill) to propagate the 8-hour funding state forward to the 1-hour candles.

### 3. Backtest Results (30-Day Period)
- **ETH:** -5.18% Return | Sharpe: -2.51 | Win Rate: 40%
- **SOL:** -5.22% Return | Sharpe: -2.34 | Win Rate: 40%

### 4. Post-Mortem: Why It Failed
While the Z-score correctly identified crowded positioning, a pure threshold lacks **regime context**. In strong directional trends, extreme funding can persist while price continues to trend, causing severe drawdowns on contrarian positions before reversion occurs ("fighting a steamroller").

### 5. Future Improvements & Next Steps
1. **Trend Filter:** Require price to break below a short-term Moving Average before entering the contrarian short.
2. **Risk Limits:** Implement Volatility-Adjusted Stop Losses (e.g., 2x ATR) to cut losses if the crowd remains irrational.
3. **Funding Arbitrage:** Pivot from directional betting to Delta-Neutral Cash & Carry arbitrage (Long Spot / Short Perp) to capture the funding yield.