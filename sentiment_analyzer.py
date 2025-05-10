from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd

nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, text):
        """Simple sentiment analysis using VADER"""
        if not text or pd.isna(text):
            return {'compound': 0, 'sentiment': 'neutral'}
        
        scores = self.sia.polarity_scores(text)
        sentiment = 'positive' if scores['compound'] > 0.05 else \
                  'negative' if scores['compound'] < -0.05 else 'neutral'
        
        return {
            'compound': scores['compound'],
            'sentiment': sentiment
        }
    
    def analyze_articles(self, df):
        """Add sentiment columns to news DataFrame"""
        df['full_text'] = df['title'].fillna('') + ". " + df['summary'].fillna('')
        sentiment = df['full_text'].apply(self.analyze).apply(pd.Series)
        return pd.concat([df, sentiment], axis=1)