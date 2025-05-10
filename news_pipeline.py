from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
from top_performer import select_stocks
import pandas as pd
import sys

def main(api_key, tickers):
    # Step 1: Fetch news
    print("🔄 Fetching news articles...")
    news_df = NewsFetcher(api_key).fetch_for_tickers(tickers)
    
    if news_df.empty:
        print("❌ No articles found")
        return
    
    # Step 2: Analyze sentiment
    print("🔍 Analyzing sentiment...")
    analyzed_df = SentimentAnalyzer().analyze_articles(news_df)
    
    # Step 3: Save results
    filename = "news_sentiment.csv"
    analyzed_df.to_csv(filename, index=False)
    
    print(f"✅ Saved {len(analyzed_df)} articles to {filename}")
    print("Columns:", analyzed_df.columns.tolist())

if __name__ == "__main__":
    api_key = "d0abq39r01qm3l9kf2vgd0abq39r01qm3l9kf300"
    
    tickers = select_stocks(max_stocks=3)  # Adjust max_stocks as needed
    
    if not tickers:
        print("❌ No tickers selected by top_performer.")
    else:
        print(f"✅ Using tickers: {', '.join(tickers)}")
        main(api_key, tickers)