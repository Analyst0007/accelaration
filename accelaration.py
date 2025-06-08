# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 14:40:03 2025

@author: Hemal
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Trading Strategy Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Core Functions
# -----------------------
@st.cache_data
def get_clean_financial_data(ticker, start_date, end_date):
    """Download and clean financial data"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data = data.ffill().bfill().dropna()
        
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data.dropna()
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def calculate_trend_slope(series, window=200):
    """Calculate trend slope using linear regression"""
    def linear_regression_slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    return series.rolling(window=window).apply(linear_regression_slope)

def calculate_slope_of_slope(slope_series, window=5):
    """Calculate acceleration (slope of slope)"""
    def slope_of_slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    return slope_series.rolling(window=window).apply(slope_of_slope)

def generate_signals(stock_data, window=200, acc_window=5):
    """Generate buy/sell signals based on slope and acceleration"""
    stock_data = stock_data.copy()
    stock_data['Slope_Long'] = calculate_trend_slope(stock_data['Close'], window=window)
    stock_data['Slope_Accel'] = calculate_slope_of_slope(stock_data['Slope_Long'], window=acc_window)

    stock_data['Signal'] = 0
    stock_data['Signal_Detail'] = ''
    buy_signal_triggered = False

    for i in range(2, len(stock_data)):
        slope = stock_data['Slope_Long'].iloc[i]
        accel = stock_data['Slope_Accel'].iloc[i]
        prev_accel = stock_data['Slope_Accel'].iloc[i-1]

        # Early Buy Signal: Acceleration crosses above 0 and slope is still negative
        if not buy_signal_triggered and accel > 0 and prev_accel <= 0 and slope < 0:
            stock_data.loc[stock_data.index[i], 'Signal'] = 1
            stock_data.loc[stock_data.index[i], 'Signal_Detail'] = 'Early Buy'
            buy_signal_triggered = True

        # Early Sell Signal: Acceleration crosses below 0 and slope is still positive
        elif buy_signal_triggered and accel < -0.01 and prev_accel >= -0.01 and slope > 0:
            stock_data.loc[stock_data.index[i], 'Signal'] = -1
            stock_data.loc[stock_data.index[i], 'Signal_Detail'] = 'Early Sell'
            buy_signal_triggered = False

    stock_data['Signal'] = stock_data['Signal'].astype('int8')
    return stock_data

def backtest_strategy(stock_data, initial_capital=10000.0):
    """Backtest the trading strategy"""
    stock_data = stock_data.copy()
    position = 0.0
    cash = initial_capital

    stock_data['Portfolio_Value'] = np.nan
    stock_data['Holdings'] = np.nan
    stock_data['Cash'] = np.nan

    stock_data['Portfolio_Value'] = stock_data['Portfolio_Value'].astype('float64')
    stock_data['Holdings'] = stock_data['Holdings'].astype('float64')
    stock_data['Cash'] = stock_data['Cash'].astype('float64')

    stock_data.loc[stock_data.index[0], 'Cash'] = initial_capital
    stock_data.loc[stock_data.index[0], 'Holdings'] = 0.0
    stock_data.loc[stock_data.index[0], 'Portfolio_Value'] = initial_capital

    for i in range(1, len(stock_data)):
        current_price = stock_data['Close'].iloc[i]
        signal = stock_data['Signal'].iloc[i]

        if signal == 1 and position == 0:
            position = cash / current_price
            cash = 0.0

        elif signal == -1 and position > 0:
            cash = position * current_price
            position = 0.0

        current_holdings = position * current_price
        current_portfolio_value = cash + current_holdings

        stock_data.loc[stock_data.index[i], 'Holdings'] = current_holdings
        stock_data.loc[stock_data.index[i], 'Cash'] = cash
        stock_data.loc[stock_data.index[i], 'Portfolio_Value'] = current_portfolio_value

    return stock_data

def create_interactive_plots(stock_data):
    """Create interactive plots using Plotly"""
    fig = sp.make_subplots(
        rows=4, cols=1,
        subplot_titles=('Price with Buy/Sell Signals', 'Trend Slope', 'Slope Acceleration', 'Portfolio Value'),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.2, 0.2, 0.3]
    )

    # Price with signals
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Close'], 
                  name='Price', line=dict(color='blue', width=2)),
        row=1, col=1
    )

    buy_signals = stock_data[stock_data['Signal'] == 1]
    sell_signals = stock_data[stock_data['Signal'] == -1]

    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                      mode='markers', name='Buy Signal',
                      marker=dict(symbol='triangle-up', size=12, color='green')),
            row=1, col=1
        )

    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                      mode='markers', name='Sell Signal',
                      marker=dict(symbol='triangle-down', size=12, color='red')),
            row=1, col=1
        )

    # Slope
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Slope_Long'],
                  name='Trend Slope', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Acceleration
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Slope_Accel'],
                  name='Slope Acceleration', line=dict(color='orange', width=2)),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # Portfolio Value
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Portfolio_Value'],
                  name='Portfolio Value', line=dict(color='darkblue', width=3)),
        row=4, col=1
    )

    # Buy and hold comparison
    initial_value = stock_data['Portfolio_Value'].iloc[0]
    buy_hold_values = initial_value * (stock_data['Close'] / stock_data['Close'].iloc[0])
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=buy_hold_values,
                  name='Buy & Hold', line=dict(color='lightblue', width=2, dash='dot')),
        row=4, col=1
    )

    fig.update_layout(height=800, showlegend=True, title_text="Trading Strategy Analysis")
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Slope", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration", row=3, col=1)
    fig.update_yaxes(title_text="Value ($)", row=4, col=1)

    return fig

# -----------------------
# Streamlit App
# -----------------------
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Trading Strategy Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Configuration")
    
    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", value="TSLA", help="Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT)")
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    slope_window = st.sidebar.slider("Slope Window", min_value=10, max_value=300, value=50, 
                                   help="Window for trend slope calculation")
    acc_window = st.sidebar.slider("Acceleration Window", min_value=3, max_value=20, value=5,
                                 help="Window for acceleration calculation")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, 
                                            value=10000, step=1000)
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download data
        status_text.text("Downloading stock data...")
        progress_bar.progress(20)
        
        stock_data = get_clean_financial_data(ticker.upper(), start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            st.error(f"Could not download data for {ticker.upper()}. Please check the ticker symbol and date range.")
            return
        
        # Generate signals
        status_text.text("Generating trading signals...")
        progress_bar.progress(40)
        stock_data = generate_signals(stock_data, slope_window, acc_window)
        
        # Backtest strategy
        status_text.text("Running backtest...")
        progress_bar.progress(60)
        stock_data = backtest_strategy(stock_data, initial_capital)
        
        # Calculate metrics
        status_text.text("Calculating performance metrics...")
        progress_bar.progress(80)
        
        buy_hold_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
        strategy_return = (stock_data['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
        
        # Count signals
        num_buy_signals = len(stock_data[stock_data['Signal'] == 1])
        num_sell_signals = len(stock_data[stock_data['Signal'] == -1])
        
        # Create plots
        status_text.text("Creating visualizations...")
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"Analysis completed for {ticker.upper()}!")
        
        # Performance metrics
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Strategy Return",
                value=f"{strategy_return:.2f}%",
                delta=f"{strategy_return - buy_hold_return:.2f}% vs B&H"
            )
        
        with col2:
            st.metric(
                label="Buy & Hold Return",
                value=f"{buy_hold_return:.2f}%"
            )
        
        with col3:
            st.metric(
                label="Buy Signals",
                value=num_buy_signals
            )
        
        with col4:
            st.metric(
                label="Sell Signals",
                value=num_sell_signals
            )
        
        # Additional metrics
        st.subheader("📈 Additional Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_days = len(stock_data)
            st.info(f"**Trading Days:** {total_days}")
        
        with col2:
            data_quality = stock_data.isnull().sum().sum()
            st.info(f"**Missing Values:** {data_quality}")
        
        with col3:
            date_range = f"{stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}"
            st.info(f"**Date Range:** {date_range}")
        
        # Interactive plots
        st.subheader("📊 Interactive Analysis")
        fig = create_interactive_plots(stock_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Signal Details"):
            signal_data = stock_data[stock_data['Signal'] != 0][['Close', 'Signal', 'Signal_Detail', 'Slope_Long', 'Slope_Accel']]
            if not signal_data.empty:
                st.dataframe(signal_data, use_container_width=True)
            else:
                st.info("No trading signals generated for the selected parameters.")
        
        # Download data
        with st.expander("💾 Download Results"):
            csv = stock_data.to_csv()
            st.download_button(
                label="Download Complete Dataset (CSV)",
                data=csv,
                file_name=f"{ticker.upper()}_{start_date}_{end_date}_analysis.csv",
                mime="text/csv"
            )

    # Information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        This app implements a trading strategy based on:
        
        • **Trend Slope**: Linear regression slope of price
        • **Acceleration**: Rate of change of slope
        
        **Signals:**
        • Buy when acceleration turns positive while slope is negative
        • Sell when acceleration turns negative while slope is positive
        """
    )

if __name__ == "__main__":
    main()