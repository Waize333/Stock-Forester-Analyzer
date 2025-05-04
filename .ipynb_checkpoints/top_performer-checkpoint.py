import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class DynamicStockSelector:
    def __init__(self, tickers, max_stocks=3):
        self.tickers = tickers
        self.max_stocks = max_stocks

    def fetch_stock_data(self, ticker):
        """Fetch and process data for a single NYSE stock."""
        try:
            df = yf.Ticker(ticker).history(period="2d", interval="1h")
            if df.empty or len(df) < 2:
                raise ValueError("Insufficient data")

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

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
    """
    Module-level function to select top-performing stocks.
    
    Args:
        tickers (list, optional): List of stock tickers to analyze. Defaults to popular tech stocks.
        max_stocks (int, optional): Maximum number of stocks to return. Defaults to 3.
        
    Returns:
        list: Ticker symbols of top-performing stocks
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JNJ', 'JPM', 'V', 'NVDA']
    
    selector = DynamicStockSelector(tickers=tickers, max_stocks=max_stocks)
    return selector.select_stocks()
