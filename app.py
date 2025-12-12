import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Page config
st.set_page_config(page_title="AI Stock Reversal Indicator", layout="wide")
st.title("AI Stock Reversal Detector")
st.markdown("Track reversals in/out of AI stocks (NASDAQ) vs Traditional (DJIA)")

# Sidebar controls
st.sidebar.header("Settings")
period = st.sidebar.selectbox("Data period", ["3mo", "6mo", "1y", "2y"], index=1)
threshold = st.sidebar.slider("Reversal threshold (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

# Fetch data
@st.cache_data
def fetch_data(period):
    try:
        nasdaq = yf.download("^IXIC", period=period, progress=False)["Close"]
        djia = yf.download("^DJI", period=period, progress=False)["Close"]
        return nasdaq, djia
    except:
        st.error("Failed to fetch data")
        return None, None

with st.spinner("Loading data..."):
    nasdaq, djia = fetch_data(period)

if nasdaq is not None and djia is not None:
    # Calculate ratio (NASDAQ / DJIA)
    ratio = nasdaq / djia
    
    # Calculate 30-day moving average of ratio
    ratio_ma = ratio.rolling(window=30).mean()
    
    # Calculate percentage change
    ratio_pct_change = ratio.pct_change() * 100
    
    # Detect reversals (when pct change exceeds threshold)
    reversal_signal = ratio_pct_change.abs() > threshold
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("NASDAQ vs DJIA Price", "NASDAQ/DJIA Ratio", "Reversal Signal"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Row 1: Price charts
    fig.add_trace(
        go.Scatter(x=nasdaq.index, y=nasdaq.values, name="NASDAQ (AI)", line=dict(color="blue"), mode="lines"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=djia.index, y=djia.values, name="DJIA (Traditional)", line=dict(color="red"), mode="lines"),
        row=1, col=1
    )
    
    # Row 2: Ratio chart
    fig.add_trace(
        go.Scatter(x=ratio.index, y=ratio.values, name="Ratio (NASDAQ/DJIA)", line=dict(color="purple"), mode="lines"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=ratio_ma.index, y=ratio_ma.values, name="30-day MA", line=dict(color="orange", dash="dash"), mode="lines"),
        row=2, col=1
    )
    
    # Row 3: Reversal signal
    reversal_dates = ratio_pct_change[reversal_signal].index
    for date in reversal_dates:
        fig.add_vline(x=date, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.add_trace(
        go.Scatter(x=ratio_pct_change.index, y=ratio_pct_change.values, name="Ratio % Change", 
                   fill='tozeroy', line=dict(color="teal"), mode="lines"),
        row=3, col=1
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="green", annotation_text=f"Threshold: +{threshold}%", row=3, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: -{threshold}%", row=3, col=1)
    
    # Update layout
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="% Change", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    fig.update_layout(height=1200, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current metrics
    st.subheader("Current Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("NASDAQ", str(round(nasdaq.iloc[-1], 2)))
    with col2:
        st.metric("DJIA", str(round(djia.iloc[-1], 2)))
    with col3:
        st.metric("Ratio", str(round(ratio.iloc[-1], 6)))
    with col4:
        st.metric("Ratio % Change", str(round(ratio_pct_change.iloc[-1], 2)) + "%")
    
    # Latest reversals
    st.subheader("Recent Reversals (Last 10)")
    recent_reversals = reversal_dates[-10:]
    if len(recent_reversals) > 0:
        reversal_values = [ratio_pct_change.loc[d] for d in recent_reversals]
        reversal_df = pd.DataFrame({
            "Date": recent_reversals,
            "Ratio % Change": reversal_values,
            "Direction": ["INTO AI" if v > 0 else "OUT OF AI" for v in reversal_values]
        })
        st.dataframe(reversal_df, use_container_width=True)
    else:
        st.info("No reversals detected with current threshold")
