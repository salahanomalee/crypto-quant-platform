import polars as pl
import numpy as np
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
HOURS_IN_YEAR = 24 * 365
ANNUALIZATION_FACTOR = np.sqrt(HOURS_IN_YEAR)
CAPITAL = 100000  # $100k hypothetical portfolio
CONFIDENCE_LEVEL = 0.95  # 95% VaR

def get_hourly_returns() -> pl.DataFrame:
    """Load data and calculate hourly log returns."""
    dfs = []
    for sym in SYMBOLS:
        file_path = PROCESSED_DATA_DIR / f"{sym}_1h.parquet"
        df = pl.read_parquet(file_path)
        df = df.select([pl.col("datetime"), pl.col("close").alias(sym)])
        dfs.append(df)
        
    # Inner join to ensure timestamps align perfectly
    wide_df = dfs[0]
    for df in dfs[1:]:
        wide_df = wide_df.join(df, on="datetime", how="inner").sort("datetime")
        
    exprs = [pl.col("datetime")]
    for sym in SYMBOLS:
        # Calculate hourly log returns
        exprs.append(pl.col(sym).log().diff().alias(f"{sym}_hourly_ret"))
        
    return wide_df.select(exprs).drop_nulls()

def calculate_daily_var_cvar(returns_df: pl.DataFrame):
    """Calculate 24-hour Historical Value at Risk (VaR) and CVaR."""
    print("--- 24-Hour Value at Risk (VaR) & CVaR (95% Confidence) ---")
    
    # Aggregate 24 hourly log returns to get daily log returns
    daily_ret_exprs = [pl.col("datetime")]
    for sym in SYMBOLS:
        daily_ret_exprs.append(
            pl.col(f"{sym}_hourly_ret").rolling_sum(window_size=24).alias(f"{sym}_daily_ret")
        )
        
    daily_df = returns_df.select(daily_ret_exprs).drop_nulls()
    
    alpha = 1 - CONFIDENCE_LEVEL # 0.05 (the 5% worst cases)
    
    for sym in SYMBOLS:
        col = f"{sym}_daily_ret"
        sorted_rets = daily_df[col].sort()
        
        # VaR is the 5th percentile (the cutoff for the worst 5% of days)
        var_idx = int(len(sorted_rets) * alpha)
        var = sorted_rets[var_idx]
        
        # CVaR (Expected Shortfall) is the average of all losses WORSE than the VaR
        cvar = sorted_rets[:var_idx].mean()
        
        print(f"{sym:10} | 24h 95% VaR: {var*100:6.2f}% | CVaR (Expected Shortfall): {cvar*100:6.2f}%")

def calculate_vol_targeting_weights(returns_df: pl.DataFrame):
    """Calculate Inverse-Volatility position sizing."""
    print(f"\n--- Volatility-Targeted Position Sizing (${CAPITAL:,.0f} Portfolio) ---")
    
    # Calculate annualized volatility for each asset
    vols = {}
    for sym in SYMBOLS:
        col = f"{sym}_hourly_ret"
        std_dev = returns_df[col].std()
        ann_vol = std_dev * ANNUALIZATION_FACTOR
        vols[sym] = ann_vol
        
    # Inverse Volatility Weighting: Lower vol = Higher allocation
    inv_vols = {sym: 1/vol for sym, vol in vols.items()}
    total_inv_vol = sum(inv_vols.values())
    
    # Normalize to 100%
    weights = {sym: (inv_vol / total_inv_vol) for sym, inv_vol in inv_vols.items()}
    
    print(f"{'Asset':<10} | {'Ann. Vol':<10} | {'Weight':<10} | {'Capital Allocation'}")
    print("-" * 55)
    
    for sym, weight in weights.items():
        allocation = CAPITAL * weight
        print(f"{sym:<10} | {vols[sym]*100:7.1f}% | {weight*100:7.1f}% | ${allocation:>12,.2f}")

if __name__ == "__main__":
    # 1. Load Returns
    hourly_rets = get_hourly_returns()
    
    # 2. Calculate Risk Limits (VaR)
    calculate_daily_var_cvar(hourly_rets)
    
    # 3. Calculate Position Sizes
    calculate_vol_targeting_weights(hourly_rets)