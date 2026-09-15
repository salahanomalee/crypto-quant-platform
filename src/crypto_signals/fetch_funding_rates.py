import logging
import sys
import time
from pathlib import Path

import ccxt
import polars as pl

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
# CCXT requires ':USDT' to specify linear perpetual futures
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
LOOKBACK_DAYS = 30


def setup_directories():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_funding_history(symbol: str, lookback_days: int) -> pl.DataFrame:
    """Fetch perpetual funding rate history and return a Polars DataFrame."""
    logger.info(f"Fetching funding history for {symbol}...")

    exchange = ccxt.binance({
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })

    since = int((time.time() - (lookback_days * 24 * 60 * 60)) * 1000)
    all_data = []

    while True:
        try:
            history = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not history:
                break
            all_data.extend(history)
            since = history[-1]["timestamp"] + 1
            if since >= int(time.time() * 1000):
                break
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            break

    if not all_data:
        return pl.DataFrame()

    df = pl.DataFrame(all_data).select([
        pl.col("timestamp").alias("timestamp_ms"),
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
        pl.col("fundingRate").alias("funding_rate"),
        pl.col("symbol"),
    ])
    return df.sort("timestamp_ms")


if __name__ == "__main__":
    setup_directories()
    all_dfs = []

    for sym in SYMBOLS:
        df = fetch_funding_history(sym, LOOKBACK_DAYS)
        if not df.is_empty():
            all_dfs.append(df)
        time.sleep(1)  # Rate limit courtesy

    if all_dfs:
        try:
            final_df = pl.concat(all_dfs)
            logger.info(f"Successfully fetched {final_df.height} total funding rate records.")

            summary = final_df.group_by("symbol").agg([
                pl.col("funding_rate").mean().alias("avg_funding"),
                pl.col("funding_rate").max().alias("max_funding"),
                pl.col("funding_rate").min().alias("min_funding"),
            ])
            logger.info(f"30-day funding summary:\n{summary}")

            file_path = PROCESSED_DATA_DIR / "funding_rates.parquet"
            final_df.write_parquet(file_path, compression="snappy")
            logger.info(f"Saved master funding dataset to {file_path}")
        except Exception as e:
            logger.critical(f"Failed to concatenate or save funding data: {e}")
            sys.exit(1)
    else:
        logger.critical("FATAL: No funding data fetched. Exchange unreachable or geo-blocked.")
        sys.exit(1)