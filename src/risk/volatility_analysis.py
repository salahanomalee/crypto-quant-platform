import polars as pl
import numpy as np
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]

# Annualization factor for 1-hour crypto data (24 hours * 365 days = 8760)
# In TradFi daily data, this would be sqrt(252).
HOURS_IN_YEAR = 24 * 365
ANNUALIZATION_FACTOR = np.sqrt(HOURS_IN_YEAR)

def load_and_widen_data() -> pl.DataFrame:
    """Load all assets and pivot them into a 'wide' dataframe for correlation analysis."""
    print("Loading and aligning asset data...")
    dfs = []
    
    for sym in SYMBOLS:
        file_path = PROCESSED_DATA_DIR / f"{sym}_1h.parquet"
        df = pl.read_parquet(file_path)
        
        # Select only datetime and close, rename close to symbol name
        df = df.select([
            pl.col("datetime"),
            pl.col("close").alias(sym)
        ])
        dfs.append(df)
    
    # Join all dataframes on datetime (Inner join ensures no missing timestamps)
    wide_df = dfs[0]
    for df in dfs[1:]:
        wide_df = wide_df.join(df, on="datetime", how="inner")
        
    return wide_df.sort("datetime")

def calculate_returns_and_volatility(df: pl.DataFrame):
    """Calculate log returns and rolling annualized volatility."""
    
    # 1. Calculate Log Returns: ln(P_t / P_{t-1})
    exprs = [pl.col("datetime")]
    for sym in SYMBOLS:
        exprs.append(
            pl.col(sym).log().diff().alias(f"{sym}_log_ret")
        )
        
    df_ret = df.select(exprs)
    
    # 2. Calculate Rolling Volatility (e.g., 168 hours = 1 week lookback)
    window_size = 168 
    
    vol_exprs = [pl.col("datetime")]
    for sym in SYMBOLS:
        # Rolling standard deviation of the log returns
        rolling_std = pl.col(f"{sym}_log_ret").rolling_std(window_size=window_size)
        
        # Annualize it: Hourly StdDev * sqrt(8760)
        annualized_vol = (rolling_std * ANNUALIZATION_FACTOR).alias(f"{sym}_ann_vol_{window_size}h")
        vol_exprs.append(annualized_vol)
        
    df_vol = df_ret.select(vol_exprs)
    
    return df_ret, df_vol

def calculate_correlation_matrix(df_ret: pl.DataFrame):
    """Calculate the correlation matrix of the log returns."""
    # Polars doesn't have a direct .corr() matrix method, 
    # so we drop the datetime and convert to Pandas for the calculation.
    pdf = df_ret.drop("datetime").drop_nulls().to_pandas()
    corr_matrix = pdf.corr()
    return corr_matrix

if __name__ == "__main__":
    # 1. Load Data
    wide_df = load_and_widen_data()
    
    # 2. Calculate Metrics
    df_ret, df_vol = calculate_returns_and_volatility(wide_df)
    
    # 3. Display Current Volatility
    print("\n--- Latest Annualized Volatility (1-Week Rolling) ---")
    latest_vol = df_vol.drop_nulls().tail(1).transpose(include_header=True)
    print(latest_vol)
    
    # 4. Display Correlation Matrix
    print("\n--- Asset Correlation Matrix (Log Returns) ---")
    corr = calculate_correlation_matrix(df_ret)
    print(corr)
    
    # 5. Save Timeseries to CSV for Excel/Plotting
    output_path = PROCESSED_DATA_DIR / "volatility_metrics.csv"
    df_vol.drop_nulls().write_csv(output_path)
    print(f"\nSaved volatility timeseries to {output_path}")