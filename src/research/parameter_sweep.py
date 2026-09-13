import vectorbt as vbt
import polars as pl
from pathlib import Path
import pandas as pd
import itertools

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
SYMBOL = "BTC_USDT_1h"

FEES = 0.001       
SLIPPAGE = 0.0005  

def load_data():
    file_path = PROCESSED_DATA_DIR / f"{SYMBOL}.parquet"
    print(f"Loading {file_path}...")
    df = pl.read_parquet(file_path)
    pdf = df.to_pandas().set_index("datetime").sort_index()
    return pdf["close"]

def run_parameter_sweep(close_prices):
    """Test multiple moving average combinations to find robust parameters."""
    print("Running parameter sweep (Grid Search)...")
    
    # Define the windows we want to test
    fast_windows = range(5, 30, 5)   # 5, 10, 15, 20, 25
    slow_windows = range(40, 100, 10) # 40, 50, 60, 70, 80, 90
    
    # Create all combinations
    combinations = list(itertools.product(fast_windows, slow_windows))
    
    # Filter to ensure Fast is always smaller than Slow
    valid_combs = [(f, s) for f, s in combinations if f < s]
    
    results = []
    for fast_w, slow_w in valid_combs:
        # 1. Calculate MAs
        fast_ma = vbt.MA.run(close_prices, window=fast_w)
        slow_ma = vbt.MA.run(close_prices, window=slow_w)
        
        # 2. Generate Signals
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # 3. Simulate Portfolio
        pf = vbt.Portfolio.from_signals(
            close_prices,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=FEES,
            slippage=SLIPPAGE,
            freq="1h",
            direction="both"
        )
        
        # 4. Save Metrics
        results.append({
            "fast_window": fast_w,
            "slow_window": slow_w,
            "sharpe": pf.sharpe_ratio(),
            "total_return_pct": pf.total_return() * 100,
            "max_drawdown_pct": pf.max_drawdown() * 100,
            "total_trades": pf.trades.count()
        })

    # Convert to Pandas DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe Ratio to see the "best" strategies
    print("\n--- Top 5 Strategies by Sharpe Ratio ---")
    print(results_df.sort_values(by="sharpe", ascending=False).head(5).to_string())
    
    return results_df

if __name__ == "__main__":
    prices = load_data()
    sweep_results = run_parameter_sweep(prices)