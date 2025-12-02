"""Analyze sentiment from financial news"""
import requests
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class NewsSentimentAnalyzer:
    """Analyze sentiment from financial news"""
    
    def __init__(self):
        """Initialize news sentiment analyzer"""
        self.news_cache = {}
    
    def fetch_news_headlines(self, symbol: str, lookback_days: int = 30) -> List[Dict]:
        """
        Fetch news headlines for a stock
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
        
        Returns:
            List of news articles
        """
        try:
            # Extract base symbol (remove .NS suffix)
            base_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # Simulate news fetching (in production, use actual news APIs)
            logger.info(f"Fetching news for {symbol} (last {lookback_days} days)")
            
            # Placeholder - In production, integrate with:
            # - News API
            # - Google News RSS
            # - Financial news websites
            news_items = [
                {
                    'title': f'Stock analysis for {base_symbol}',
                    'date': datetime.now() - timedelta(days=i),
                    'source': 'Market News'
                }
                for i in range(min(5, lookback_days))
            ]
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
            subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = 'positive'
            elif polarity < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment_label
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
    
    def get_stock_sentiment_score(self, symbol: str, lookback_days: int = 30) -> Dict:
        """
        Get overall sentiment score for a stock
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
        
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            news_items = self.fetch_news_headlines(symbol, lookback_days)
            
            if not news_items:
                logger.warning(f"No news found for {symbol}")
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.5,  # Neutral
                    'news_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0
                }
            
            sentiments = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for news in news_items:
                sentiment = self.analyze_sentiment(news['title'])
                sentiments.append(sentiment['polarity'])
                
                if sentiment['sentiment'] == 'positive':
                    positive_count += 1
                elif sentiment['sentiment'] == 'negative':
                    negative_count += 1
                else:
                    neutral_count += 1
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Normalize to 0-1 scale (from -1 to 1)
            sentiment_score = (avg_sentiment + 1) / 2
            
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'avg_polarity': avg_sentiment,
                'news_count': len(news_items),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment score for {symbol}: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0.5,
                'news_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
    
    def analyze_multiple_stocks(self, symbols: List[str], lookback_days: int = 30) -> pd.DataFrame:
        """
        Analyze sentiment for multiple stocks
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days to look back
        
        Returns:
            DataFrame with sentiment analysis
        """
        results = []
        
        for symbol in symbols:
            sentiment_data = self.get_stock_sentiment_score(symbol, lookback_days)
            results.append(sentiment_data)
        
        df = pd.DataFrame(results)
        logger.info(f"Analyzed sentiment for {len(df)} stocks")
        return df
