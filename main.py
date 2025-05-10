import pandas as pd
import numpy as np
import time
import os
from typing import Dict, Any, Tuple
import model
import top_performer as tp
import stockData as sk
import news_pipeline as npp

def create_sentiment_csv():
        api_key = "d0abq39r01qm3l9kf2vgd0abq39r01qm3l9kf300"
        ticker = tp.select_stocks(max_stocks=3)[0]  # Select the first ticker
        npp.main(api_key, [ticker])


def load_sentiment_from_csv(symbol: str, filepath: str = "news_sentiment_20250505.csv") -> Dict[str, Any]:
    """
    Load sentiment data from CSV file for a specific symbol
    
    Args:
        symbol: The ticker/symbol to get sentiment for
        filepath: Path to the sentiment CSV file
    
    Returns:
        Dictionary with sentiment data
    """
    try:
        df = pd.read_csv(filepath)
        
        # Print available tickers in the CSV for debugging
        available_tickers = df['ticker'].unique()
        print(f"Available tickers in {filepath}: {available_tickers}")
        
        # Case-insensitive filter by ticker/symbol to be more flexible
        symbol_df = df[df['ticker'].str.upper() == symbol.upper()]
        
        # If no exact match, try to use AAPL as fallback for demonstration
        if len(symbol_df) == 0:
            print(f"No sentiment data found for {symbol} in {filepath}")
            print("Attempting to use AAPL sentiment data as a fallback...")
            symbol_df = df[df['ticker'].str.upper() == "AAPL"]
            
            # If still no match, use the first ticker in the file
            if len(symbol_df) == 0 and len(df) > 0:
                first_ticker = df['ticker'].iloc[0]
                print(f"Using {first_ticker} sentiment data as fallback")
                symbol_df = df[df['ticker'] == first_ticker]
        
        if len(symbol_df) == 0:
            print(f"No usable sentiment data found in {filepath}")
            return {'score': 0, 'volume': 0, 'positive_count': 0, 'negative_count': 0, 'positive_ratio': 0}
        
        # Calculate average sentiment score
        avg_score = symbol_df['compound'].mean()
        
        # Count news items
        news_count = len(symbol_df)
        
        # Calculate positive vs negative ratio
        positive_count = len(symbol_df[symbol_df['sentiment'] == 'positive'])
        negative_count = len(symbol_df[symbol_df['sentiment'] == 'negative'])
        
        print(f"Found {news_count} news items for {symbol} with average sentiment: {avg_score:.2f}")
        
        return {
            'score': avg_score,
            'volume': news_count,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_count / news_count if news_count > 0 else 0
        }
        
    except Exception as e:
        print(f"Error loading sentiment data from {filepath}: {e}")
        # Print file contents for debugging
        try:
            with open(filepath, 'r') as f:
                print(f"First few lines of {filepath}:")
                for i, line in enumerate(f):
                    if i < 5:  # Print just the first 5 lines
                        print(line.strip())
                    else:
                        break
        except:
            pass
        return {'score': 0, 'volume': 0, 'positive_count': 0, 'negative_count': 0, 'positive_ratio': 0}


def generate_recommendation(price_direction: float, 
                          price_strength: str,
                          sentiment_score: float, 
                          sentiment_strength: str) -> Tuple[str, str]:
    """
    Generate investment recommendation based on technical and sentiment signals
    """
    # Case 1: Both positive (price up, positive sentiment)
    if price_direction > 0 and sentiment_score > 0:
        if price_strength == "strong" and sentiment_strength == "strong":
            return "Strong Buy", "Technical analysis predicts significant price increase with strong confidence, supported by very positive market sentiment. Both technical and fundamental indicators align for a high-conviction opportunity."
        
        elif price_strength == "strong" and sentiment_strength != "strong":
            return "Buy", "Technical indicators strongly suggest price increase, with moderately positive sentiment support. Technical analysis leads this recommendation, though sentiment is only moderately reinforcing."
        
        elif price_strength != "strong" and sentiment_strength == "strong":
            return "Buy", "Highly positive sentiment indicates strong market confidence despite only moderate technical signals. Consider buying with awareness that this is more fundamentals-driven than technically driven."
        
        else:
            return "Weak Buy", "Both technical indicators and sentiment are mildly positive, suggesting a potential but not compelling opportunity. Consider a small position with careful monitoring."
    
    # Case 2: Price up but sentiment negative
    elif price_direction > 0 and sentiment_score < 0:
        if price_strength == "strong" and sentiment_strength == "strong":
            return "Cautious Technical Buy", "Technical indicators strongly predict price increase, but strongly negative sentiment presents significant fundamental risk. This divergence suggests caution - the price may rise short-term but face headwinds due to poor fundamentals."
        
        elif price_strength == "strong" and sentiment_strength != "strong":
            return "Technical Buy", "Strong technical signals suggest price increase despite mild negative sentiment. The technical case outweighs the minor fundamental concerns."
        
        elif price_strength != "strong" and sentiment_strength == "strong":
            return "Hold/Watch", "Technical indicators show mild upward potential but strongly negative sentiment presents substantial risk. Wait for sentiment improvement or stronger technical confirmation."
        
        else:
            return "Neutral/Wait", "Mixed signals with mild technical positivity offset by mild negative sentiment. Better opportunities likely exist elsewhere."
    
    # Case 3: Price down but sentiment positive
    elif price_direction < 0 and sentiment_score > 0:
        if price_strength == "strong" and sentiment_strength == "strong":
            return "Wait for Reversal", "Technical indicators currently bearish despite strong positive sentiment. This divergence suggests a potential reversal opportunity - watch closely for technical confirmation of the positive fundamentals."
        
        elif price_strength != "strong" and sentiment_strength == "strong":
            return "Speculative Buy", "Strong positive sentiment may overcome mild technical weakness. Consider a small position if you have higher risk tolerance, as fundamentals may soon shift technical direction."
        
        else:
            return "Hold", "Positive sentiment provides some optimism despite technical weakness. Not recommended for new positions, but existing positions may benefit from holding through this period."
    
    # Case 4: Both negative
    else:  # price_direction <= 0 and sentiment_score <= 0
        if price_strength == "strong" and sentiment_strength == "strong":
            return "Strong Sell/Avoid", "Both technical analysis and sentiment are strongly negative. Avoid this investment completely or consider shorting/selling existing positions."
        
        elif price_strength == "strong" or sentiment_strength == "strong":
            return "Sell/Avoid", "Significant negative signals from both technicals and fundamentals. Not recommended for investment at this time."
        
        else:
            return "Weak Sell/Caution", "Mild negative signals suggest caution. Better opportunities exist elsewhere."


def generate_analysis_report(symbol: str, price_prediction: Dict[str, Any], 
                           sentiment_data: Dict[str, Any], timeframe: str = "short") -> str:
    """
    Generate a human-readable report of the analysis
    
    Args:
        symbol: The ticker symbol
        price_prediction: Dictionary with price prediction data
        sentiment_data: Dictionary with sentiment analysis data
        timeframe: Analysis timeframe
        
    Returns:
        Formatted analysis report as string
    """
    # Extract metrics
    price_direction = price_prediction.get('direction', 0)
    price_change_pct = price_prediction.get('change_percent', 0)
    last_price = price_prediction.get('last_price', 0)
    next_price = price_prediction.get('next_price', 0)
    confidence = price_prediction.get('confidence', 0.5)
    
    sentiment_score = sentiment_data.get('score', 0)
    sentiment_volume = sentiment_data.get('volume', 0)
    
    # Calculate signal strengths
    price_strength = model.evaluate_signal_strength(price_change_pct, confidence)
    sentiment_strength = model.evaluate_signal_strength(sentiment_score, sentiment_volume/100 if sentiment_volume > 0 else 0)
    
    # Generate recommendation
    recommendation, explanation = generate_recommendation(
        price_direction, price_strength, sentiment_score, sentiment_strength
    )
    
    # Format the report
    report = [
        f"==== Investment Analysis Report: {symbol.upper()} ====",
        f"Timeframe: {timeframe}",
        f"\nRECOMMENDATION: {recommendation}",
        f"\n{explanation}",
        "\nANALYSIS SUMMARY:",
        f"• Technical Forecast: {'BULLISH' if price_direction > 0 else 'BEARISH'}",
        f"  Current Price: ${last_price:.2f}",
        f"  Predicted Price: ${next_price:.2f}",
        f"  Predicted Change: {price_change_pct:.2%} ({price_strength} signal)",
        f"• Market Sentiment: {'POSITIVE' if sentiment_score > 0 else 'NEGATIVE'} ({sentiment_strength} signal)",
        f"  Sentiment Score: {sentiment_score:.2f} (based on {sentiment_volume} news items)",
        f"  Positive News Ratio: {sentiment_data.get('positive_ratio', 0):.1%}"
    ]
    
    return "\n".join(report)


def main():
    # Fetch stock data
    interval = "15m"          # Changed to daily for more reliable data
    num_candles = 1280       # Reasonable number of candles

    try:
        top_tickers = tp.select_stocks()
    except Exception as e:
        print(f"Error calling select_stocks(): {e}")
        top_tickers = ["AAPL", "MSFT", "GOOGL"]  # Default tickers if selection fails
        
    if not top_tickers:
        print("No tickers returned from select_stocks()")
        top_tickers = ["AAPL"]  # Default ticker
        
    print(f"Selected tickers: {', '.join(top_tickers)}")
    ticker = top_tickers[0]  # Use first ticker
    
    # Try to fetch data
    print(f"Fetching {num_candles} {interval} candles for: {ticker}")
    data_df = sk.fetch_candles(ticker, interval, num_candles)

    create_sentiment_csv()  # Create sentiment CSV if not already done
    
    # Wait briefly to ensure data fetching is complete
    print("Waiting 5 seconds for data processing and sentiment collection")
    time.sleep(5)

    # Read data and prepare for training
    try:
        data = pd.read_csv("data.csv")
        data = data.ffill()  # Forward fill any missing values
        print(f"Loaded data with {len(data)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Define predictors and target
    PREDICTORS = ["open", "high", "low"]
    TARGET = "close"

    # Save original data for denormalization
    original_data = data.copy()
    
    # Scale the data
    data[PREDICTORS + [TARGET]] = model.standard_scale(data, PREDICTORS + [TARGET])

    # Split into train, valid, test sets
    np.random.seed(0)
    split_data = np.split(data, [int(.7*len(data)), int(.85*len(data))])
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [
        [d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] 
        for d in split_data
    ]

    # RNN configuration
    layer_conf = [
        {"type": "input", "units": len(PREDICTORS)},
        {"type": "rnn", "hidden": 3, "output": 1},
    ]

    # Check if we should load or train model
    model_path = f"model_{ticker}.npy"
    if os.path.exists(model_path):
        print(f"Found existing model for {ticker}. Loading...")
        layers = model.load_model(ticker)
        if layers is None:
            print("Failed to load model. Training new model...")
            layers = model.train_model(train_x, train_y, valid_x, valid_y, layer_conf)
    else:
        print(f"No existing model found for {ticker}. Training new model...")
        layers = model.train_model(train_x, train_y, valid_x, valid_y, layer_conf)
        
    # Save the trained model
    model.save_model(layers, ticker)
        
    # Make prediction
    print("\n--- Making Prediction for Next Candle ---")
    
    # Get the most recent sequence for prediction
    sequence_len = 100
    if test_x.shape[0] >= sequence_len:
        last_sequence = test_x[-sequence_len:]
    else:
        # If not enough test data, use the last available data
        last_sequence = np.vstack([train_x[-sequence_len + test_x.shape[0]:], test_x])
    
    # Predict the next candle
    next_price = model.predict_next_candle(layers, last_sequence, original_data, PREDICTORS, TARGET)
    
    # Get the last known price for comparison
    last_known_price = original_data[TARGET].iloc[-1]
    
    # Calculate expected movement
    price_change = next_price - last_known_price
    price_change_pct = price_change / last_known_price
    
    print(f"Last known price: ${last_known_price:.2f}")
    print(f"Predicted next price: ${next_price:.2f}")
    print(f"Expected change: ${price_change:.2f} ({price_change_pct:.2%})")
    
    direction = 1 if price_change > 0 else -1
    print(f"Prediction: {'📈 PRICE INCREASE' if direction > 0 else '📉 PRICE DECREASE'}")
    
    # Create price prediction dictionary
    price_prediction = {
        'next_price': next_price,
        'last_price': last_known_price,
        'change': price_change,
        'change_percent': price_change_pct,
        'direction': direction,
        'confidence': 0.6  # Base confidence - can be adjusted
    }
    
    # Get sentiment data for current symbol
    print(f"\n--- Analyzing Sentiment for {ticker} ---")
    
    # Try different date formats for sentiment files
    sentiment_files = [
        f"news_sentiment_{time.strftime('%Y%m%d')}.csv",  # today's date
        "news_sentiment.csv",  # default filename
        "news_sentiment_20250505.csv"  # example provided file
    ]
    
    sentiment_data = None
    for file in sentiment_files:
        if os.path.exists(file):
            sentiment_data = load_sentiment_from_csv(ticker, file)
            print(f"Loaded sentiment data from: {file}")
            break
    
    if sentiment_data is None:
        print("No sentiment data files found. Using neutral sentiment.")
        sentiment_data = {'score': 0, 'volume': 0, 'positive_count': 0, 'negative_count': 0, 'positive_ratio': 0}
    
    # Generate the full analysis report
    print("\n" + "=" * 50)
    analysis_report = generate_analysis_report(ticker, price_prediction, sentiment_data)
    print(analysis_report)
    print("=" * 50)
    
    # Save the analysis report to a file
    report_file = f"analysis_report_{ticker}.txt"
    with open(report_file, "w") as f:
        f.write(analysis_report)
    print(f"\nAnalysis report saved to {report_file}")


if __name__ == "__main__":
    main()