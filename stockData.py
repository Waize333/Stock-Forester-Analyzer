import requests
import pandas as pd
import top_performer

API_KEY = "1E59L4GUHYAO8B3K"

def fetch_candles(ticker, interval, num_candles):
    interval_mapping = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '60min', '1d': 'daily', '1wk': 'weekly', '1mo': 'monthly'
    }

    if interval not in interval_mapping:
        raise ValueError(f"Unsupported interval: {interval}")

    try:
        endpoint = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY" if interval in ['1m', '5m', '15m', '30m', '1h'] else "TIME_SERIES_DAILY",
            "symbol": ticker,
            "interval": interval_mapping[interval] if interval in ['1m', '5m', '15m', '30m', '1h'] else None,
            "apikey": API_KEY,
            "outputsize": "full"
        }

        # Make the API request
        response = requests.get(endpoint, params=params)
        data = response.json()

        # Parse the response
        if interval in ['1m', '5m', '15m', '30m', '1h']:
            time_series_key = f"Time Series ({interval_mapping[interval]})"
        else:
            time_series_key = "Time Series (Daily)"

        if time_series_key not in data:
            raise ValueError(f"Error fetching data: {data.get('Note', 'Unknown error')}")

        df = pd.DataFrame(data[time_series_key]).T
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.tail(num_candles).reset_index()
        df.rename(columns={'index': 'datetime'}, inplace=True)

        # Save to CSV
        filename = "data.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} candles for {ticker} to {filename}")
        return df

    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    interval = "15m"  
    num_candles = 1738     

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
