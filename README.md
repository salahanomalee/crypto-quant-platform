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