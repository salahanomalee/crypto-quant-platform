import vectorbt as vbt
import polars as pl
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
SYMBOL = "BTC_USDT_1h"

# Realistic crypto trading costs
FEES = 0.001       # 0.1% exchange fee (typical for Binance spot/perps)
SLIPPAGE = 0.0005  # 0.05% estimated slippage/market impact

def load_data(file_name: str):
    """Load Parquet data with Polars, then convert to Pandas for vectorbt."""
    file_path = PROCESSED_DATA_DIR / file_name
    print(f"Loading {file_path}...")
    
    df = pl.read_parquet(file_path)
    
    # vectorbt strictly requires a pandas Series with a DatetimeIndex
    pdf = df.to_pandas()
    pdf.set_index("datetime", inplace=True)
    
    # Ensure index is sorted (vectorbt requirement)
    pdf = pdf.sort_index()
    
    return pdf["close"]

def run_momentum_backtest(close_prices):
    """Run a dual moving average crossover strategy."""
    
    # 1. Calculate Fast and Slow Moving Averages
    fast_ma = vbt.MA.run(close_prices, window=10) # 10-hour MA
    slow_ma = vbt.MA.run(close_prices, window=50) # 50-hour MA
    
    # 2. Generate Entries and Exits
    # Entry when fast crosses ABOVE slow (Bullish momentum)
    entries = fast_ma.ma_crossed_above(slow_ma)
    # Exit when fast crosses BELOW slow (Bearish momentum)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    # 3. Run Portfolio Simulation
    print("Simulating trades with fees and slippage...")
    pf = vbt.Portfolio.from_signals(
        close_prices,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1h",
        direction="both" # Allow shorting as well (typical for crypto perps)
    )
    
    return pf

def generate_report(pf):
    """Print quant-grade performance metrics and save an interactive chart."""
    
    print("\n--- Portfolio Performance ---")
    print(f"Total Return:   {pf.total_return() * 100:.2f}%")
    print(f"Sharpe Ratio:   {pf.sharpe_ratio():.2f}")
    print(f"Sortino Ratio:  {pf.sortino_ratio():.2f}")
    print(f"Max Drawdown:   {pf.max_drawdown() * 100:.2f}%")
    print(f"Total Trades:   {pf.trades.count()}")
    print(f"Win Rate:       {pf.trades.win_rate() * 100:.2f}%")
    
    # Save an interactive HTML plot (Plotly)
    report_path = PROCESSED_DATA_DIR / "backtest_report.html"
    pf.plot().write_html(report_path)
    print(f"\nSaved interactive chart to {report_path}")

if __name__ == "__main__":
    prices = load_data(f"{SYMBOL}.parquet")
    
    # --- WALK-FORWARD VALIDATION ---
    # Split data: First 80% for training/finding parameters, Last 20% for blind testing
    split_idx = int(len(prices) * 0.80)
    
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]
    
    print(f"Training on {len(train_prices)} candles (Days 1-24)")
    train_pf = run_momentum_backtest(train_prices)
    
    print(f"\nBlind testing on {len(test_prices)} candles (Days 25-30)")
    test_pf = run_momentum_backtest(test_prices)
    
    print("\n--- OUT-OF-SAMPLE PERFORMANCE ---")
    print(f"Total Return: {test_pf.total_return() * 100:.2f}%")
    print(f"Sharpe Ratio: {test_pf.sharpe_ratio():.2f}")