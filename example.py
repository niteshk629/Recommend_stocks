"""
Example usage of the Stock Recommendation System
"""
from src.utils.config_loader import ConfigLoader
from src.data_collection.stock_data_fetcher import StockDataFetcher
from src.fundamental_analysis.fundamental_metrics import FundamentalAnalyzer
from src.sentiment_analysis.news_sentiment import NewsSentimentAnalyzer
from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.recommendation_engine.recommender import StockRecommender

# Initialize components
config = ConfigLoader()
data_fetcher = StockDataFetcher(period='1y', interval='1d')
fundamental_analyzer = FundamentalAnalyzer()
sentiment_analyzer = NewsSentimentAnalyzer()
technical_indicators = TechnicalIndicators()
recommender = StockRecommender()

# Example 1: Fetch historical data for a single stock
print("\n=== Example 1: Fetch Historical Data ===")
symbol = 'RELIANCE.NS'
historical_data = data_fetcher.fetch_stock_data(symbol)
print(f"\nFetched {len(historical_data)} records for {symbol}")
print(historical_data.tail())

# Example 2: Add technical indicators
print("\n=== Example 2: Technical Analysis ===")
historical_data = technical_indicators.add_all_indicators(historical_data)
print("\nTechnical Indicators (last 5 days):")
print(historical_data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']].tail())

# Example 3: Fundamental analysis
print("\n=== Example 3: Fundamental Analysis ===")
fundamental_metrics = fundamental_analyzer.get_fundamental_metrics(symbol)
print(f"\nFundamental Metrics for {symbol}:")
for key, value in fundamental_metrics.items():
    if value is not None:
        print(f"  {key}: {value}")

valuation_score = fundamental_analyzer.calculate_valuation_score(fundamental_metrics)
print(f"\nValuation Score: {valuation_score:.3f}")

# Example 4: Sentiment analysis
print("\n=== Example 4: Sentiment Analysis ===")
sentiment_data = sentiment_analyzer.get_stock_sentiment_score(symbol, lookback_days=30)
print(f"\nSentiment Data for {symbol}:")
for key, value in sentiment_data.items():
    print(f"  {key}: {value}")

# Example 5: Generate recommendation
print("\n=== Example 5: Generate Recommendation ===")
recommendation = recommender.generate_recommendation(
    symbol=symbol,
    technical_score=0.65,
    fundamental_score=valuation_score,
    sentiment_score=sentiment_data['sentiment_score'],
    additional_data={
        'current_price': historical_data['Close'].iloc[-1],
        'market_cap': fundamental_metrics.get('market_cap')
    }
)

print(f"\nRecommendation for {symbol}:")
print(f"  Recommendation: {recommendation['recommendation']}")
print(f"  Confidence: {recommendation['confidence']}")
print(f"  Composite Score: {recommendation['composite_score']:.3f}")
print(f"  Technical Score: {recommendation['technical_score']:.3f}")
print(f"  Fundamental Score: {recommendation['fundamental_score']:.3f}")
print(f"  Sentiment Score: {recommendation['sentiment_score']:.3f}")

# Example 6: Analyze multiple stocks
print("\n=== Example 6: Multiple Stock Analysis ===")
symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
print(f"\nAnalyzing {len(symbols)} stocks...")

fundamental_results = fundamental_analyzer.analyze_multiple_stocks(symbols)
print("\nFundamental Analysis Results:")
print(fundamental_results[['symbol', 'pe_ratio', 'roe', 'valuation_score']].to_string(index=False))

print("\n" + "="*60)
print("Example completed successfully!")
print("="*60)
