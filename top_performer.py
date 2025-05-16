import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Alpha Vantage API key
# Old keys:
# API_KEY = "1E59L4GUHYAO8B3K"
# API_KEY = "FVGKGAH0YR15ISRS"
# API_KEY = "1IOHS5A8MEGP1RBA"
# API_KEY = "XF1WJ0RX54CZJ0IO"
API_KEY = " ZTWP7FZ2RTWDY3EG"

class DynamicStockSelector:
    def __init__(self, tickers, max_stocks=3):
        self.tickers = tickers
        self.max_stocks = max_stocks

    def fetch_stock_data(self, ticker):
        """Fetch and process data for a single stock using Alpha Vantage."""
        try:
            # Alpha Vantage API endpoint
            endpoint = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": ticker,
                "interval": "60min",
                "apikey": API_KEY,
                "outputsize": "compact"
            }

            # Make the API request
            response = requests.get(endpoint, params=params)
            data = response.json()

            # Parse the response
            time_series_key = "Time Series (60min)"
            if time_series_key not in data:
                raise ValueError(f"Error fetching data for {ticker}: {data.get('Note', 'Unknown error')}")

            df = pd.DataFrame(data[time_series_key]).T
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float).reset_index()
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

            # Calculate metrics
            volatility = df['close'].pct_change().std()
            volume = df['volume'].mean()
            liquidity = df['close'].iloc[-1] * df['volume'].iloc[-1]
            trend = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

            return {
                'ticker': ticker,
                'volatility': volatility,
                'volume': volume,
                'liquidity': liquidity,
                'trend': trend
            }

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None

    def apply_filters(self, df):
        """Apply filters to remove low-performing stocks."""
        min_volume = df['volume'].quantile(0.25)
        df = df[df['volume'] > min_volume]

        min_liquidity = df['liquidity'].quantile(0.25)
        df = df[df['liquidity'] > min_liquidity]

        vol_lower, vol_upper = df['volatility'].quantile([0.25, 0.75])
        df = df[(df['volatility'] > vol_lower) & (df['volatility'] < vol_upper)]

        return df

    def select_stocks(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            stock_data = list(filter(None, executor.map(self.fetch_stock_data, self.tickers)))

        df_stocks = pd.DataFrame(stock_data)
        if df_stocks.empty:
            print("No valid stock data collected.")
            return []

        df_stocks = self.apply_filters(df_stocks)

        # Normalize and score
        for column in ['volatility', 'volume', 'liquidity', 'trend']:
            df_stocks[column] = (df_stocks[column] - df_stocks[column].min()) / (
                df_stocks[column].max() - df_stocks[column].min() + 1e-9)

        df_stocks['score'] = (
            df_stocks['volatility'] * 0.3 +
            df_stocks['volume'] * 0.3 +
            df_stocks['liquidity'] * 0.2 +
            df_stocks['trend'] * 0.2
        )

        top_stocks = df_stocks.nlargest(self.max_stocks, 'score')
        print("Top selected stocks:")
        print(top_stocks[['ticker', 'score']])

        return top_stocks['ticker'].tolist()

def select_stocks(tickers=None, max_stocks=3):

    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JNJ', 'JPM', 'V', 'NVDA']
    
    selector = DynamicStockSelector(tickers=tickers, max_stocks=max_stocks)
    return selector.select_stocks()
