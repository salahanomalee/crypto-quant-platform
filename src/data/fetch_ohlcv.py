import logging
import sys
import time
from pathlib import Path

import ccxt
import duckdb
import polars as pl
from tqdm import tqdm

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Fetching {symbol} {timeframe} data...")

    exchange = ccxt.binance({"enableRateLimit": True})

    # Calculate start time in milliseconds
    since = int((time.time() - (lookback_days * 24 * 60 * 60)) * 1000)
    all_data = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since >= int(time.time() * 1000):
                break
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            break

    if not all_data:
        return pl.DataFrame()

    df = pl.DataFrame(
        all_data,
        schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
        orient="row",  # Fixes Polars DataOrientationWarning
    )

    df = df.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").alias("datetime"),
    )
    return df


def validate_data(df: pl.DataFrame) -> pl.DataFrame:
    """Quant-grade data validation: remove duplicates and bad rows."""
    initial_rows = df.height

    df = df.unique(subset=["symbol", "timestamp_ms"])
    df = df.sort("timestamp_ms")
    df = df.filter((pl.col("volume") > 0) & (pl.col("close") > 0))

    logger.info(f"Validation: {initial_rows} rows in -> {df.height} rows out.")
    return df


def save_to_parquet(df: pl.DataFrame, symbol: str) -> Path:
    """Save dataframe to a Parquet file."""
    clean_symbol = symbol.replace("/", "_")
    file_path = PROCESSED_DATA_DIR / f"{clean_symbol}_{TIMEFRAME}.parquet"
    df.write_parquet(file_path, compression="snappy")
    logger.info(f"Saved {symbol} to {file_path}")
    return file_path


def run_duckdb_analytics(file_path: Path):
    """Run a quick SQL query using DuckDB directly on the Parquet file."""
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
    result = duckdb.query(query).pl()
    logger.info(f"DuckDB analytics for {file_path.name}:\n{result}")


if __name__ == "__main__":
    setup_directories()

    saved = 0
    failed = 0
    logger.info(f"Starting OHLCV pipeline for {len(SYMBOLS)} symbols...")

    for sym in tqdm(SYMBOLS, desc="Fetching OHLCV", unit="sym"):
        try:
            raw_df = fetch_exchange_data(sym, TIMEFRAME, LOOKBACK_DAYS)

            if raw_df.is_empty():
                logger.warning(f"No data returned for {sym}. Skipping.")
                continue

            clean_df = validate_data(raw_df)
            file_path = save_to_parquet(clean_df, sym)
            run_duckdb_analytics(file_path)

            saved += 1
        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {sym}: {e}")

        time.sleep(2)  # Respect API rate limits

    logger.info("-" * 50)
    logger.info(f"Pipeline complete | success={saved} | failed={failed}")

    if saved == 0:
        logger.critical("FATAL: No OHLCV data fetched. Exchange unreachable or geo-blocked.")
        sys.exit(1)