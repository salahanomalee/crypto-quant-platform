import polars as pl
from pathlib import Path
import requests
import os

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
FUNDING_FILE = PROCESSED_DATA_DIR / "funding_rates.parquet"

# Funding occurs roughly every 8 hours on Binance.
# 21 funding observations ≈ 7 days.
ROLLING_WINDOW = 21

# Z-score thresholds for signal classification
EXTREME_POSITIVE_Z = 1.5
EXTREME_NEGATIVE_Z = -1.5


def load_funding_data() -> pl.DataFrame:
    """Load funding rate data from Parquet."""
    if not FUNDING_FILE.exists():
        raise FileNotFoundError(
            f"{FUNDING_FILE} not found. Run fetch_funding_rates.py first."
        )

    df = pl.read_parquet(FUNDING_FILE)

    required_cols = {"timestamp_ms", "datetime", "funding_rate", "symbol"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort(["symbol", "timestamp_ms"])


def calculate_funding_zscores(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate rolling funding mean, rolling std, and Z-score per symbol.

    Z-score tells us how unusual the current funding rate is compared
    to recent funding history.
    """

    df = df.with_columns(
        [
            pl.col("funding_rate")
            .rolling_mean(window_size=ROLLING_WINDOW)
            .over("symbol")
            .alias("funding_rolling_mean"),

            pl.col("funding_rate")
            .rolling_std(window_size=ROLLING_WINDOW)
            .over("symbol")
            .alias("funding_rolling_std"),
        ]
    )

    df = df.with_columns(
        (
            (pl.col("funding_rate") - pl.col("funding_rolling_mean"))
            / pl.col("funding_rolling_std")
        ).alias("funding_zscore")
    )

    return df


def classify_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classify funding conditions.

    Positive extreme:
        Crowded longs / possible contrarian short research signal.

    Negative extreme:
        Crowded shorts / possible contrarian long research signal.
    """

    df = df.with_columns(
        pl.when(pl.col("funding_zscore") >= EXTREME_POSITIVE_Z)
        .then(pl.lit("crowded_long_possible_short_research"))
        .when(pl.col("funding_zscore") <= EXTREME_NEGATIVE_Z)
        .then(pl.lit("crowded_short_possible_long_research"))
        .otherwise(pl.lit("neutral"))
        .alias("funding_signal")
    )

    return df


def print_latest_snapshot(df: pl.DataFrame):
    """Print latest funding Z-score per asset."""
    print("\n--- Latest Funding Z-Score Snapshot ---")

    latest = (
        df.drop_nulls(["funding_zscore"])
        .sort(["symbol", "timestamp_ms"])
        .group_by("symbol")
        .tail(1)
        .select(
            [
                "symbol",
                "datetime",
                "funding_rate",
                "funding_rolling_mean",
                "funding_rolling_std",
                "funding_zscore",
                "funding_signal",
            ]
        )
        .sort("symbol")
    )

    print(latest)


def print_extreme_events(df: pl.DataFrame):
    """Print most extreme positive and negative funding observations."""
    clean = df.drop_nulls(["funding_zscore"])

    print("\n--- Top Positive Funding Z-Score Events ---")
    top_positive = (
        clean.sort("funding_zscore", descending=True)
        .select(
            [
                "symbol",
                "datetime",
                "funding_rate",
                "funding_zscore",
                "funding_signal",
            ]
        )
        .head(10)
    )
    print(top_positive)

    print("\n--- Top Negative Funding Z-Score Events ---")
    top_negative = (
        clean.sort("funding_zscore", descending=False)
        .select(
            [
                "symbol",
                "datetime",
                "funding_rate",
                "funding_zscore",
                "funding_signal",
            ]
        )
        .head(10)
    )
    print(top_negative)


def print_signal_counts(df: pl.DataFrame):
    """Count how many times each signal occurred."""
    print("\n--- Funding Signal Counts ---")

    counts = (
        df.drop_nulls(["funding_zscore"])
        .group_by(["symbol", "funding_signal"])
        .agg(pl.len().alias("count"))
        .sort(["symbol", "funding_signal"])
    )

    print(counts)


def save_results(df: pl.DataFrame):
    """Save Z-score-enriched funding dataset."""
    output_path = PROCESSED_DATA_DIR / "funding_zscores.parquet"
    df.write_parquet(output_path, compression="snappy")
    print(f"\nSaved funding Z-score dataset to {output_path}")


def send_telegram_alert(df: pl.DataFrame, bot_token: str, chat_id: str):
    """Send a Telegram message if any asset has an extreme funding signal."""
    # Get the latest snapshot
    latest = df.drop_nulls(["funding_zscore"]).sort(["symbol", "timestamp_ms"]).group_by("symbol").tail(1)
    
    alerts = latest.filter(pl.col("funding_signal") != "neutral")
    
    if alerts.is_empty():
        print("\nNo extreme signals detected. No alert sent.")
        return

    print(f"\n🚨 SENDING {alerts.height} ALERT(S) TO TELEGRAM...")
    
    # Telegram supports basic HTML formatting
    message_content = "🚨 <b>CRYPTO QUANT ALERT: EXTREME FUNDING DETECTED</b> 🚨\n\n"
    
    for row in alerts.iter_rows(named=True):
        sym = row['symbol']
        z = row['funding_zscore']
        signal = row['funding_signal']
        rate = row['funding_rate']
        
        message_content += f"<b>{sym}</b>\n"
        message_content += f"Z-Score: <code>{z:.2f}</code> | Raw Rate: <code>{rate:.5f}</code>\n"
        message_content += f"Signal: <code>{signal}</code>\n\n"
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_content,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200 and response.json().get("ok"):
            print("✅ Alert successfully sent to Telegram!")
        else:
            print(f"❌ Failed to send alert: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Telegram API error: {e}")


if __name__ == "__main__":
    funding_df = load_funding_data()
    zscore_df = calculate_funding_zscores(funding_df)
    signal_df = classify_signals(zscore_df)

    print_latest_snapshot(signal_df)
    print_extreme_events(signal_df)
    print_signal_counts(signal_df)

    save_results(signal_df)

    # ==========================================
    # SECURELY LOAD CREDENTIALS FROM ENVIRONMENT
    # ==========================================
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram_alert(signal_df, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    else:
        print("\n⚠️ Skipping Telegram alert: Environment variables not set.")