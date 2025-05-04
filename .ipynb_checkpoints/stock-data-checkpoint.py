import yfinance as yf
import pandas as pd
import top_performer  

def fetch_candles(ticker, interval, num_candles):
    interval_mapping = {
        '1m': '1d', '5m': '5d', '15m': '5d', '30m': '5d',
        '1h': '7d', '1d': '60d', '1wk': '1y', '1mo': '2y'
    }

    if interval not in interval_mapping:
        raise ValueError(f"Unsupported interval: {interval}")

    try:
        df = yf.Ticker(ticker).history(period=interval_mapping[interval], interval=interval)
        df = df.tail(num_candles).reset_index()
        df.columns = [col.lower() for col in df.columns]

        filename = f"{ticker}_{interval}_{num_candles}_candles.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} candles for {ticker} to {filename}")
        return df

    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # --- USER SETTINGS ---
    interval = "1h"         # e.g., "1d", "1h", "15m"
    num_candles = 48        # Number of candles

    # --- FETCH TOP STOCKS ---
    try:
        top_tickers = top_performer.select_stocks()
    except Exception as e:
        print(f"Error calling select_stocks(): {e}")
        top_tickers = []

    if not top_tickers:
        print("No tickers returned from top_performer.select_stocks()")
    else:
        print(f"Fetching {num_candles} {interval} candles for:", ", ".join(top_tickers))
        for ticker in top_tickers:
            fetch_candles(ticker, interval, num_candles)
