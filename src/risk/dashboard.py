import streamlit as st
import polars as pl
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="Crypto Risk Command Center", layout="wide")
PROCESSED_DATA_DIR = Path("data/processed")
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
HOURS_IN_YEAR = 24 * 365
ANNUALIZATION_FACTOR = np.sqrt(HOURS_IN_YEAR)

@st.cache_data
def load_data():
    """Load and align all parquet files."""
    dfs = []
    for sym in SYMBOLS:
        file_path = PROCESSED_DATA_DIR / f"{sym}_1h.parquet"
        if not file_path.exists():
            st.error(f"Missing {file_path}. Run fetch_ohlcv.py first!")
            st.stop()
            
        df = pl.read_parquet(file_path).select([
            pl.col("datetime"), 
            pl.col("close").alias(sym)
        ])
        dfs.append(df)
        
    wide_df = dfs[0]
    for df in dfs[1:]:
        wide_df = wide_df.join(df, on="datetime", how="inner").sort("datetime")
    return wide_df

def calculate_metrics(df: pl.DataFrame):
    """Calculate log returns and rolling volatility."""
    # Log returns (Note the "_ret" suffix added here!)
    exprs = [pl.col("datetime")]
    for sym in SYMBOLS:
        exprs.append(pl.col(sym).log().diff().alias(f"{sym}_ret"))
    df_ret = df.select(exprs).drop_nulls()
    
    # Rolling Volatility (168 hours = 1 week)
    vol_exprs = [pl.col("datetime")]
    for sym in SYMBOLS:
        # We use the "_ret" column to calculate volatility
        rolling_std = pl.col(f"{sym}_ret").rolling_std(window_size=168)
        ann_vol = (rolling_std * ANNUALIZATION_FACTOR).alias(sym)
        vol_exprs.append(ann_vol)
        
    df_vol = df_ret.select(vol_exprs).drop_nulls()
    return df_ret, df_vol

# --- UI LAYOUT ---
st.title("📊 Crypto Risk Command Center")
st.markdown("Institutional-grade analytics for BTC, ETH, and SOL.")

# Load data
df_wide = load_data()
df_ret, df_vol = calculate_metrics(df_wide)

# Convert to pandas for plotting
pdf_vol = df_vol.to_pandas().set_index("datetime")
pdf_ret = df_ret.drop("datetime").to_pandas()

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📈 Volatility Trends", "🔥 Correlation Matrix", "⚠️ VaR & Sizing"])

with tab1:
    st.subheader("Annualized Volatility (1-Week Rolling)")
    fig_vol = px.line(pdf_vol, title="Annualized Volatility over Time")
    fig_vol.update_layout(hovermode="x unified")
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.subheader("Asset Correlation Matrix (Log Returns)")
    corr_matrix = pdf_ret.corr()
    
    # Create interactive heatmap
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap (1.0 = Perfectly Correlated)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.warning("**Risk Insight:** Notice how high the correlation is between all three assets. Holding all three provides very little diversification benefit during market crashes.")

with tab3:
    st.subheader("24-Hour 95% Value at Risk (VaR)")
    
    # Calculate daily returns (24h rolling sum of hourly log returns)
    daily_rets = pdf_ret.rolling(window=24).sum().dropna()
    
    var_data = []
    for sym in SYMBOLS:
        # FIX: Append "_ret" to match the actual column name in the dataframe
        ret_col = f"{sym}_ret"
        sorted_rets = daily_rets[ret_col].sort_values()
        var_95 = sorted_rets.iloc[int(len(sorted_rets) * 0.05)]
        cvar_95 = sorted_rets[sorted_rets <= var_95].mean()
        
        var_data.append({
            "Asset": sym.replace("_USDT", ""),
            "95% VaR (Worst 5% Day)": f"{var_95*100:.2f}%",
            "CVaR (Expected Shortfall)": f"{cvar_95*100:.2f}%"
        })
        
    st.dataframe(pd.DataFrame(var_data), hide_index=True, use_container_width=True)
    
    st.subheader("Volatility-Targeted Position Sizing ($100k Portfolio)")
    
    # Calculate current vols and weights
    # FIX: Append "_ret" to match the actual column name
    current_vols = {sym: pdf_ret[f"{sym}_ret"].std() * ANNUALIZATION_FACTOR for sym in SYMBOLS}
    inv_vols = {sym: 1/v for sym, v in current_vols.items()}
    total_inv = sum(inv_vols.values())
    
    sizing_data = []
    for sym in SYMBOLS:
        weight = inv_vols[sym] / total_inv
        sizing_data.append({
            "Asset": sym.replace("_USDT", ""),
            "Current Volatility": f"{current_vols[sym]*100:.1f}%",
            "Optimal Weight": f"{weight*100:.1f}%",
            "Capital Allocation": f"${weight * 100000:,.2f}"
        })
        
    st.dataframe(pd.DataFrame(sizing_data), hide_index=True, use_container_width=True)