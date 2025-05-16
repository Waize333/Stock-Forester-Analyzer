import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import model
import top_performer as tp
import stockData as sk
import news_pipeline as npp
import main
from typing import Dict, Any, Tuple

st.set_page_config(page_title="Smart Stock Prediction & Analysis", layout="wide")

st.title("Smart Stock Prediction & Analysis")
st.write("This app automatically identifies top-performing stocks and uses RNN to predict prices with market sentiment analysis.")

# Current stock info section
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "AAPL"

# Thin separator
st.markdown("<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: none; background-color: rgba(128, 128, 128, 0.2);'>", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    
    # Add auto-selection options
    selection_mode = st.radio("Ticker Selection Mode", ["Auto (Top Performers)", "Manual"])
    st.session_state.selection_mode = selection_mode
    
    if selection_mode == "Auto (Top Performers)":
        try:
            with st.spinner("Finding top performing stocks..."):
                top_tickers = main.select_top_stocks(max_stocks=3)
                if not top_tickers:
                    st.warning("No top stocks found. Using default tickers.")
                    top_tickers = ["AAPL", "MSFT", "GOOGL"]
            ticker = st.selectbox("Selected Top Performers", options=top_tickers, index=0)
            st.info(f"Top performers identified: {', '.join(top_tickers)}")
            st.session_state.top_tickers = top_tickers
        except Exception as e:
            st.error(f"Error finding top stocks: {str(e)}")
            ticker = st.text_input("Stock Ticker (fallback)", value="AAPL")
    else:
        ticker = st.text_input("Stock Ticker", value="AAPL")
    
    interval = st.selectbox("Time Interval", options=["15m", "30m", "1h", "1d"], index=0)
    num_candles = st.slider("Number of Candles", min_value=100, max_value=2000, value=1280, step=100)
    
    run_button = st.button("Run Analysis")

# Selected stock indicator 
if ticker != st.session_state.current_ticker:
    st.session_state.current_ticker = ticker

# Display selection method info
if st.session_state.selection_mode == "Auto (Top Performers)":
    stock_info_col1, stock_info_col2 = st.columns([1, 3])
    with stock_info_col1:
        st.write("🏆 **Auto-selected:**")
    with stock_info_col2:
        st.write(f"**{ticker}** was selected based on volatility, volume, liquidity, and trend metrics.")

# Initialize session state for tracking progress
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
    st.session_state.messages = []
    st.session_state.prediction_data = None
    st.session_state.sentiment_data = None
    st.session_state.recommendation = None
    st.session_state.stock_data = None
    st.session_state.selection_mode = "Manual"
    st.session_state.top_tickers = ["AAPL", "MSFT", "GOOGL"]
    st.session_state.current_ticker = "AAPL"

# Function to add a message to the log
def add_log(message, is_header=False):
    st.session_state.messages.append({"text": message, "is_header": is_header})

# Main process function
def run_analysis(ticker, interval, num_candles):
    progress_bar = st.progress(0)
    
    # Step 1: Reset logs
    st.session_state.messages = []
    st.session_state.current_step = 1
    progress_bar.progress(0.1)
    
    # Show information about ticker selection mode
    if st.session_state.selection_mode == "Auto (Top Performers)":
        add_log(f"Using auto-selected top performer: {ticker}", True)
        add_log(f"Top performers identified: {', '.join(st.session_state.top_tickers)}")
    else:
        add_log(f"Using manually selected ticker: {ticker}", True)
    
    # Run the analysis using main.py functions
    try:
        # Call the main analysis function with our logger
        analysis_results = main.smart_stock_analysis(ticker, interval, num_candles, add_log)
        
        # Update session state with results
        st.session_state.stock_data = analysis_results['stock_data']
        st.session_state.prediction_data = analysis_results['price_prediction']
        st.session_state.sentiment_data = analysis_results['sentiment_data']
        st.session_state.recommendation = analysis_results['recommendation']
        
        progress_bar.progress(1.0)
        add_log("Analysis complete!")
    except Exception as e:
        add_log(f"Error during analysis: {str(e)}")
        progress_bar.progress(1.0)

# Display progress log
def display_logs():
    log_container = st.container()
    with log_container:
        st.subheader("Process Log")
        for message in st.session_state.messages:
            if message["is_header"]:
                st.markdown(f"### {message['text']}")
            else:
                st.text(message['text'])

# Display stock data visualization
def display_stock_data():
    if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
        st.subheader("Stock Price History")
        
        # Create a modified dataframe to remove gaps
        df = st.session_state.stock_data.copy()
        
        # Sort by datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        # Replace datetime with continuous indices to remove gaps
        df['candle_index'] = range(len(df))
        
        # Create candlestick chart using Plotly with continuous indices
        fig = go.Figure(data=[go.Candlestick(
            x=df['candle_index'],  # Using index instead of datetime
            open=df['open'], 
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Historical Data'
        )])
        
        # Create a secondary x-axis with some reference dates
        reference_dates = []
        if len(df) > 0:
            # Create 5 reference points with their indices and dates
            step = len(df) // 5
            for i in range(0, len(df), step):
                if i < len(df):
                    reference_dates.append({
                        'index': df['candle_index'].iloc[i],
                        'date': df['datetime'].iloc[i].strftime('%Y-%m-%d %H:%M')
                    })
        
        # Add prediction point if prediction data exists
        if st.session_state.prediction_data is not None:
            pred = st.session_state.prediction_data
            
            # Get the last candle index and add 1 for prediction
            last_index = df['candle_index'].iloc[-1]
            prediction_index = last_index + 1
            
            # Determine color based on prediction direction
            prediction_color = '#00FF7F' if pred['direction'] > 0 else '#FF4500'
            
            # Add a longer prediction line (extending past the prediction point)
            extended_index = prediction_index + 2  # Extend line 2 points beyond prediction
            fig.add_trace(go.Scatter(
                x=[last_index - 1, last_index, prediction_index, extended_index],  # Start one candle earlier and extend two candles further
                y=[
                   float(df['close'].iloc[-2]) if len(df) > 1 else float(df['close'].iloc[-1]),  # Previous candle or repeat last if not enough data
                   float(df['close'].iloc[-1]),  # Last known price
                   float(pred['next_price']),  # Prediction point
                   float(pred['next_price']) + (float(pred['next_price']) - float(df['close'].iloc[-1]))  # Extend the trend
                ],
                mode='lines+text',
                line=dict(
                    color=prediction_color,
                    width=3,
                    dash='dot'
                ),
                text=['', '', 'PREDICTION', ''],
                textposition='top center',
                textfont=dict(
                    color='gold',
                    size=16
                ),
                name=f"Prediction: ${pred['next_price']:.2f} ({pred['change_percent']*100:+.2f}%)",
                showlegend=True
            ))
            
            # Add shaded area for prediction
            # Convert pandas min/max to float values to avoid sequence multiplication error
            try:
                min_low = float(df['low'].min())
                max_high = float(df['high'].max())
                next_price = float(pred['next_price'])
                
                # Calculate trend continuation for prediction
                trend_continuation = next_price + (next_price - float(df['close'].iloc[-1]))
                
                # Create extended prediction area
                fig.add_shape(
                    type="rect",
                    x0=last_index + 0.5,  # Start slightly after last candle
                    x1=extended_index + 0.5,  # End after the extended prediction point
                    y0=min(float(min_low) * 0.995, next_price * 0.995, trend_continuation * 0.995),  # Extend below chart
                    y1=max(float(max_high) * 1.005, next_price * 1.005, trend_continuation * 1.005),  # Extend above chart
                    line=dict(
                        color=prediction_color,
                        width=0,
                    ),
                    fillcolor=prediction_color,
                    opacity=0.1,
                    layer="below",
                    name="Prediction Zone"
                )
            except (TypeError, ValueError) as e:
                st.caption(f"Note: Could not add prediction shading due to data issue. Using simplified display.")
            
        # Update layout with custom x-axis ticks and range
        pred_range = 0
        if st.session_state.prediction_data is not None:
            pred_range = 3  # Increased to show the extended prediction line
            
        fig.update_layout(
            title=f'{ticker} Stock Price with Prediction',
            yaxis_title='Price (USD)',
            xaxis_title='Candle Index',
            height=500,
            xaxis=dict(
                tickmode='array',
                tickvals=[d['index'] for d in reference_dates],
                ticktext=[d['date'] for d in reference_dates],
                range=[0, df['candle_index'].iloc[-1] + pred_range + 0.5]  # Extend range for prediction
            ),
            legend_title="Price Data",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a note about the prediction visualization
        if st.session_state.prediction_data is not None:
            pred_direction = "increase" if pred['direction'] > 0 else "decrease"
            st.caption(f"Note: The dotted line shows the predicted price trend with an expected {pred_direction} of {abs(pred['change_percent']*100):.2f}% from the last known price.")
    else:
        # Display fallback when no data is available
        st.warning("⚠️ No stock data available to display. This may be due to API rate limits or network issues.")
        if st.session_state.prediction_data is not None:
            pred = st.session_state.prediction_data
            # Create a simple bar chart for prediction only
            fig = go.Figure()
            prediction_color = '#00FF7F' if pred['direction'] > 0 else '#FF4500'
            
            fig.add_trace(go.Bar(
                x=['Current', 'Predicted'],
                y=[pred['last_price'], pred['next_price']],
                marker_color=['gray', prediction_color],
                text=[f"${pred['last_price']:.2f}", f"${pred['next_price']:.2f}"],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f'{ticker} Price Prediction',
                yaxis_title='Price (USD)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Note: Using simplified chart due to data fetching limitations. Prediction shows a {abs(pred['change_percent']*100):.2f}% {'increase' if pred['direction'] > 0 else 'decrease'}.")

# Display sentiment data
def display_sentiment():
    if st.session_state.sentiment_data is not None:
        sentiment = st.session_state.sentiment_data
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment Score", f"{sentiment.get('score', 0):.2f}")
        with col2:
            st.metric("News Volume", sentiment.get('volume', 0))
        with col3:
            st.metric("Positive Ratio", f"{sentiment.get('positive_ratio', 0):.1%}")

# Display prediction and recommendation
def display_recommendation():
    if st.session_state.recommendation is not None:
        rec = st.session_state.recommendation
        pred = st.session_state.prediction_data
        
        # Create a highlighted box for prediction
        st.markdown("""
        <style>
        .prediction-box {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid gold;
            margin-bottom: 20px;
        }
        .prediction-title {
            color: gold;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }
        .prediction-value {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
        .prediction-up {
            color: #00FF7F;
        }
        .prediction-down {
            color: #FF4500;
        }
        .current-price {
            font-size: 20px;
            text-align: center;
            color: #CCCCCC;
        }
        .percent-change {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Get prediction color based on direction
        prediction_color_class = "prediction-up" if pred['direction'] > 0 else "prediction-down"
        direction_text = "📈 PRICE INCREASE PREDICTED" if pred['direction'] > 0 else "📉 PRICE DECREASE PREDICTED"
        direction_icon = "📈" if pred['direction'] > 0 else "📉"
        
        # Create the highlighted prediction box
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-title">{direction_icon} PRICE PREDICTION {direction_icon}</div>
            <div class="current-price">Current: ${pred['last_price']:.2f}</div>
            <div class="prediction-value {prediction_color_class}">${pred['next_price']:.2f}</div>
            <div class="percent-change {prediction_color_class}">{pred['change_percent']:+.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Also keep the metrics for more detail
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${pred['last_price']:.2f}")
        with col2:
            st.metric("Predicted Price", f"${pred['next_price']:.2f}", 
                     f"{pred['change_percent']:.2%}")
        with col3:
            st.metric("Direction", "BULLISH" if pred['direction'] > 0 else "BEARISH")
        
        st.subheader("Investment Recommendation")
        rec_color = "green" if rec['recommendation'] in ["Strong Buy", "Buy"] else \
                   "orange" if rec['recommendation'] in ["Weak Buy", "Hold", "Speculative Buy", "Technical Buy"] else "red"
        
        st.markdown(f"<h2 style='color:{rec_color};'>{rec['recommendation']}</h2>", unsafe_allow_html=True)
        st.write(rec['explanation'])
        
        # Signal strength indicators
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Technical Signal")
            if rec['price_direction'] > 0:
                direction_icon = "📈" 
            else:
                direction_icon = "📉"
            
            if rec['price_strength'] == "strong":
                strength_meter = "🟢🟢🟢"
            elif rec['price_strength'] == "moderate":
                strength_meter = "🟢🟢⚪"
            else:
                strength_meter = "🟢⚪⚪"
                
            st.markdown(f"### {direction_icon} {strength_meter}")
            
        with col2:
            st.subheader("Sentiment Signal")
            if rec['sentiment_score'] > 0:
                sentiment_icon = "😀" 
            else:
                sentiment_icon = "😟"
            
            if rec['sentiment_strength'] == "strong":
                strength_meter = "🟢🟢🟢"
            elif rec['sentiment_strength'] == "moderate":
                strength_meter = "🟢🟢⚪"
            else:
                strength_meter = "🟢⚪⚪"
                
            st.markdown(f"### {sentiment_icon} {strength_meter}")
        
        # Full report in expander
        with st.expander("Full Analysis Report"):
            st.text(rec['report'])

# Run the analysis when button is clicked
if run_button:
    run_analysis(ticker, interval, num_candles)

# Reorganize the layout: prediction first, then stock chart, then logs in expandable section
if st.session_state.recommendation is not None:
    st.markdown("---")
    display_recommendation()

if st.session_state.stock_data is not None:
    st.markdown("---")
    display_stock_data()

if st.session_state.sentiment_data is not None:
    st.markdown("---")
    st.subheader("Sentiment Analysis")
    display_sentiment()

# Move logs to an expandable section at the bottom
st.markdown("---")
with st.expander("Process Log (Click to expand)", expanded=False):
    display_logs()

# Footer
st.markdown("---")
st.caption("RNN Stock Prediction & Analysis Tool - Built with Streamlit") 