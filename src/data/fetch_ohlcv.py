import ccxt
import polars as pl
import duckdb
import time
from pathlib import Path

# --- Configuration ---
EXCHANGE_ID = "binance"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
LOOKBACK_DAYS = 30
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def setup_directories():
    """Ensure data directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_exchange_data(symbol: str, timeframe: str, lookback_days: int) -> pl.DataFrame:
    """Fetch OHLCV data from exchange and return a Polars DataFrame."""
    print(f"Fetching {symbol} {timeframe} data...")

    exchange = ccxt.binance({"enableRateLimit": True})

    # Calculate start time in milliseconds
    since = int((time.time() - (lookback_days * 24 * 60 * 60)) * 1000)

    all_data = []

    while True:
        try:
            # Fetch batch of candles
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=1000
            )

            if not ohlcv:
                break

            all_data.extend(ohlcv)

            # Update 'since' to the timestamp of the last fetched candle + 1ms
            since = ohlcv[-1][0] + 1

            # Stop if we reached the current time
            if since >= int(time.time() * 1000):
                break

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break

    if not all_data:
        return pl.DataFrame()

    # Convert to Polars DataFrame
    # ccxt returns rows like:
    # [timestamp_ms, open, high, low, close, volume]
    df = pl.DataFrame(
        all_data,
        schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
        orient="row"  # Fixes Polars DataOrientationWarning
    )

    # Add symbol column and convert timestamp to datetime
    df = df.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").alias("datetime")
    )

    return df


def validate_data(df: pl.DataFrame) -> pl.DataFrame:
    """Quant-grade data validation: remove duplicates and bad rows."""
    initial_rows = df.height

    # 1. Drop exact duplicates
    df = df.unique(subset=["symbol", "timestamp_ms"])

    # 2. Sort by time
    df = df.sort("timestamp_ms")

    # 3. Filter out bad data
    df = df.filter(
        (pl.col("volume") > 0) &
        (pl.col("close") > 0)
    )

    final_rows = df.height
    print(f"Validation: Started with {initial_rows} rows, ended with {final_rows} rows.")

    return df


def save_to_parquet(df: pl.DataFrame, symbol: str) -> Path:
    """Save dataframe to a Parquet file."""
    clean_symbol = symbol.replace("/", "_")
    file_path = PROCESSED_DATA_DIR / f"{clean_symbol}_{TIMEFRAME}.parquet"

    # Write to parquet using snappy compression
    df.write_parquet(file_path, compression="snappy")

    print(f"Saved {symbol} to {file_path}")
    return file_path


def run_duckdb_analytics(file_path: Path):
    """Run a quick SQL query using DuckDB directly on the Parquet file."""
    print("\n--- DuckDB Analytics ---")

    # Use POSIX-style path for safer cross-platform compatibility
    parquet_path = file_path.as_posix()

    query = f"""
        SELECT 
            symbol,
            MIN(datetime) AS start_date,
            MAX(datetime) AS end_date,
            COUNT(*) AS total_candles,
            AVG(close) AS avg_price,
            MAX(high) - MIN(low) AS max_range
        FROM read_parquet('{parquet_path}')
        GROUP BY symbol
    """

    # Use .pl() to return a Polars DataFrame directly
    # This avoids needing pandas or numpy
    result = duckdb.query(query).pl()

    print(result)


if __name__ == "__main__":
    setup_directories()

    for sym in SYMBOLS:
        # 1. Fetch
        raw_df = fetch_exchange_data(sym, TIMEFRAME, LOOKBACK_DAYS)

        if raw_df.is_empty():
            continue

        # 2. Validate
        clean_df = validate_data(raw_df)

        # 3. Save
        file_path = save_to_parquet(clean_df, sym)

        # 4. Analyze
        run_duckdb_analytics(file_path)

        # Be nice to the exchange API
        time.sleep(2)