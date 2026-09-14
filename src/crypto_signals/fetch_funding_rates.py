import ccxt
import polars as pl
import time
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
# CCXT requires ':USDT' to specify linear perpetual futures
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"] 
LOOKBACK_DAYS = 30

def setup_directories():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_funding_history(symbol: str, lookback_days: int) -> pl.DataFrame:
    print(f"Fetching funding history for {symbol}...")
    
    # Initialize exchange specifically for the Futures market
    exchange = ccxt.binance({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True
    })
    
    # Calculate start time in milliseconds
    since = int((time.time() - (lookback_days * 24 * 60 * 60)) * 1000)
    all_data = []
    
    while True:
        try:
            # Fetch funding rate history (max 1000 items per call on Binance)
            history = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not history:
                break
                
            all_data.extend(history)
            
            # Update 'since' to the timestamp of the last fetched record + 1ms
            since = history[-1]['timestamp'] + 1
            
            if since >= int(time.time() * 1000):
                break
                
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
            
    if not all_data:
        return pl.DataFrame()
        
    df = pl.DataFrame(all_data)
    
    # Select and format relevant columns
    df = df.select([
        pl.col("timestamp").alias("timestamp_ms"),
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
        pl.col("fundingRate").alias("funding_rate"),
        pl.col("symbol")
    ])
    
    # Sort chronologically
    return df.sort("timestamp_ms")

if __name__ == "__main__":
    setup_directories()
    all_dfs = []
    
    for sym in SYMBOLS:
        df = fetch_funding_history(sym, LOOKBACK_DAYS)
        if not df.is_empty():
            all_dfs.append(df)
        time.sleep(1) # Rate limit courtesy
        
    if all_dfs:
        # Concatenate all assets into one master dataframe
        final_df = pl.concat(all_dfs)
        print(f"\nSuccessfully fetched {final_df.height} total funding rate records.")
        
        # Show a quick summary using Polars group_by
        summary = final_df.group_by("symbol").agg([
            pl.col("funding_rate").mean().alias("avg_funding"),
            pl.col("funding_rate").max().alias("max_funding"),
            pl.col("funding_rate").min().alias("min_funding")
        ])
        print("\n--- 30-Day Funding Rate Summary ---")
        print(summary)
        
        # Save to Parquet
        file_path = PROCESSED_DATA_DIR / "funding_rates.parquet"
        final_df.write_parquet(file_path, compression="snappy")
        print(f"\nSaved master dataset to {file_path}")