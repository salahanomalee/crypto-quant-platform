import vectorbt as vbt
import polars as pl
import pandas as pd
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
TARGET_SYMBOLS = ["ETH_USDT", "SOL_USDT"] 

# Realistic crypto trading costs
FEES = 0.001       # 0.1% exchange fee
SLIPPAGE = 0.0005  # 0.05% estimated slippage

def run_funding_backtest(symbol: str):
    print(f"\n--- Running Funding Mean Reversion Backtest for {symbol} ---")
    
    # 1. Load 1H Spot Prices
    price_path = PROCESSED_DATA_DIR / f"{symbol}_1h.parquet"
    if not price_path.exists():
        print(f"Missing price data for {symbol}. Run fetch_ohlcv.py first.")
        return
        
    price_df = pl.read_parquet(price_path).select(["datetime", "close"])
    price_pdf = price_df.to_pandas().set_index("datetime").sort_index()
    
    # 2. Load 8H Funding Signals
    funding_df = pl.read_parquet(PROCESSED_DATA_DIR / "funding_zscores.parquet")
    
    # Format the symbol name to match the funding data (e.g., ETH_USDT -> ETH/USDT:USDT)
    funding_symbol = f"{symbol.replace('_', '/')}:USDT" 
    sym_funding = funding_df.filter(pl.col("symbol") == funding_symbol).select(["datetime", "funding_zscore"])
    funding_pdf = sym_funding.to_pandas().set_index("datetime").sort_index()
    
    # 3. Align 8H Funding to 1H Prices using Forward Fill (ffill)
    # This is crucial: it propagates the 8H funding state to the subsequent 1H candles
    aligned_df = price_pdf.join(funding_pdf, how="left")
    aligned_df["funding_zscore"] = aligned_df["funding_zscore"].ffill()
    aligned_df = aligned_df.dropna()
    
    close_prices = aligned_df["close"]
    z_scores = aligned_df["funding_zscore"]
    
    # 4. Generate Mean Reversion Signals
    # LONG when Z < -1.5 (Crowded Short), EXIT LONG when Z > 0
    long_entries = z_scores < -1.5
    long_exits = z_scores > 0.0
    
    # SHORT when Z > 1.5 (Crowded Long), EXIT SHORT when Z < 0
    short_entries = z_scores > 1.5
    short_exits = z_scores < 0.0
    
    # 5. Simulate Portfolio
    print("Simulating trades with fees and slippage...")
    pf = vbt.Portfolio.from_signals(
        close_prices,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1h",
        direction="both"
    )
    
    # 6. Print Metrics
    print(f"\n--- {symbol} Performance ---")
    print(f"Total Return: {pf.total_return() * 100:.2f}%")
    print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
    print(f"Total Trades: {pf.trades.count()}")
    print(f"Win Rate:     {pf.trades.win_rate() * 100:.2f}%")
    
    # Save Interactive HTML
    report_path = PROCESSED_DATA_DIR / f"funding_backtest_{symbol}.html"
    pf.plot().write_html(report_path)
    print(f"Saved interactive chart to {report_path}")

if __name__ == "__main__":
    for sym in TARGET_SYMBOLS:
        run_funding_backtest(sym)