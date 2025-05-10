import requests
import pandas as pd
from datetime import datetime, timedelta
import os

class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1/news"
    
    def fetch_articles(self, ticker, days_back=1):
        """Fetch news for a single stock ticker"""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).timestamp()
        
        try:
            response = requests.get(
                f"{self.base_url}?symbol={ticker}&token={self.api_key}"
            )
            articles = [
                a for a in response.json() 
                if datetime.fromtimestamp(a['datetime']).date() >= datetime.fromtimestamp(cutoff_date).date()
            ]
            
            return pd.DataFrame([{
                'ticker': ticker,
                'title': a.get('headline', ''),
                'summary': a.get('summary', ''),
                'published_at': datetime.fromtimestamp(a['datetime']).isoformat(),
                'url': a.get('url', ''),
                'source': a.get('source', '')
            } for a in articles])
            
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            return pd.DataFrame()

    def fetch_for_tickers(self, tickers, days_back=1):
        """Fetch news for multiple tickers"""
        return pd.concat([self.fetch_articles(ticker, days_back) for ticker in tickers])