import subprocess
import sys
import time

def run_stage(script_path: str):
    """Executes a python script and handles errors gracefully."""
    print(f"\n{'='*50}")
    print(f"🚀 STARTING STAGE: {script_path}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # sys.executable ensures we use the exact same Python environment (uv's .venv)
    result = subprocess.run(
        [sys.executable, script_path], 
        capture_output=False, # Print output directly to the console
        text=True
    )
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ FATAL ERROR in {script_path}. Pipeline halted.")
        sys.exit(1)
        
    print(f"✅ STAGE COMPLETE: {script_path} ({duration:.2f}s)")

if __name__ == "__main__":
    print("🌍 Initializing Crypto Quant Cloud Pipeline...")
    
    # 1. Market Data ETL
    run_stage("src/data/fetch_ohlcv.py")
    
    # 2. Derivatives Data ETL
    run_stage("src/crypto_signals/fetch_funding_rates.py")
    
    # 3. Signal Generation (Z-Scores)
    run_stage("src/crypto_signals/analyze_funding_zscore.py")
    
    # 4. Risk Analytics (Optional: uncomment if you want risk metrics every run)
    # run_stage("src/risk/portfolio_risk.py")
    
    print("\n" + "="*50)
    print("🏁 PIPELINE EXECUTION SUCCESSFUL")
    print("="*50 + "\n")