"""
Main entry point for Stock Recommendation System
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data_collection.stock_data_fetcher import StockDataFetcher
from src.fundamental_analysis.fundamental_metrics import FundamentalAnalyzer
from src.sentiment_analysis.news_sentiment import NewsSentimentAnalyzer
from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.models.ml_models import StockMLModels
from src.recommendation_engine.recommender import StockRecommender

logger = setup_logger(__name__, log_file='logs/stock_recommender.log')


class StockRecommendationSystem:
    """Main stock recommendation system"""
    
    def __init__(self, config_path=None):
        """
        Initialize the recommendation system
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Stock Recommendation System")
        
        self.config = ConfigLoader(config_path)
        self.data_fetcher = StockDataFetcher(
            period=self.config.get('data_collection.historical_period', '5y'),
            interval=self.config.get('data_collection.interval', '1d')
        )
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.technical_indicators = TechnicalIndicators()
        self.ml_models = StockMLModels()
        self.recommender = StockRecommender(
            thresholds=self.config.get('recommendation.thresholds')
        )
        
        logger.info("Stock Recommendation System initialized successfully")
    
    def collect_data(self, symbols=None):
        """
        Collect data for stocks
        
        Args:
            symbols: List of stock symbols (uses config if None)
        
        Returns:
            DataFrame with historical data
        """
        if symbols is None:
            symbols = self.config.get_stock_universe()
        
        logger.info(f"Collecting data for {len(symbols)} stocks")
        
        # Fetch historical data
        historical_data = self.data_fetcher.fetch_multiple_stocks(symbols)
        
        if historical_data.empty:
            logger.error("Failed to fetch historical data")
            return pd.DataFrame()
        
        logger.info(f"Collected {len(historical_data)} data points")
        return historical_data
    
    def analyze_stocks(self, symbols=None):
        """
        Perform complete analysis on stocks
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            DataFrame with analysis results
        """
        if symbols is None:
            symbols = self.config.get_stock_universe()
        
        logger.info(f"Analyzing {len(symbols)} stocks")
        
        results = []
        
        for symbol in symbols:
            try:
                logger.info(f"Analyzing {symbol}")
                
                # Fetch historical data
                historical_data = self.data_fetcher.fetch_stock_data(symbol)
                
                if historical_data is None or historical_data.empty:
                    logger.warning(f"Skipping {symbol} - no historical data")
                    continue
                
                # Add technical indicators
                historical_data = self.technical_indicators.add_all_indicators(historical_data)
                
                # Get latest data point for technical score
                latest_data = historical_data.iloc[-1]
                
                # Calculate technical score based on indicators
                technical_score = self._calculate_technical_score(latest_data)
                
                # Get fundamental metrics
                fundamental_metrics = self.fundamental_analyzer.get_fundamental_metrics(symbol)
                fundamental_score = self.fundamental_analyzer.calculate_valuation_score(fundamental_metrics)
                
                # Get sentiment score
                sentiment_data = self.sentiment_analyzer.get_stock_sentiment_score(
                    symbol,
                    lookback_days=self.config.get('data_collection.sentiment_lookback_days', 30)
                )
                sentiment_score = sentiment_data['sentiment_score']
                
                # Get stock info
                stock_info = self.data_fetcher.get_stock_info(symbol)
                
                results.append({
                    'symbol': symbol,
                    'name': stock_info.get('name', ''),
                    'sector': stock_info.get('sector', ''),
                    'current_price': latest_data.get('Close'),
                    'market_cap': stock_info.get('market_cap'),
                    'technical_score': technical_score,
                    'fundamental_score': fundamental_score,
                    'sentiment_score': sentiment_score,
                    'pe_ratio': fundamental_metrics.get('pe_ratio'),
                    'rsi': latest_data.get('RSI'),
                })
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        df = pd.DataFrame(results)
        logger.info(f"Analysis complete for {len(df)} stocks")
        return df
    
    def _calculate_technical_score(self, data_point) -> float:
        """
        Calculate technical score from indicators
        
        Args:
            data_point: Series with technical indicators
        
        Returns:
            Technical score (0-1)
        """
        score = 0.5  # Default neutral
        
        try:
            # RSI scoring
            rsi = data_point.get('RSI')
            if rsi and rsi < 30:
                score += 0.2  # Oversold - bullish
            elif rsi and rsi > 70:
                score -= 0.2  # Overbought - bearish
            
            # MACD scoring
            macd = data_point.get('MACD')
            macd_signal = data_point.get('MACD_Signal')
            if macd and macd_signal:
                if macd > macd_signal:
                    score += 0.15  # Bullish crossover
                else:
                    score -= 0.15  # Bearish crossover
            
            # Moving average scoring
            close = data_point.get('Close')
            sma_20 = data_point.get('SMA_20')
            sma_50 = data_point.get('SMA_50')
            
            if close and sma_20 and close > sma_20:
                score += 0.1  # Above 20-day MA
            
            if close and sma_50 and close > sma_50:
                score += 0.1  # Above 50-day MA
            
            # Bollinger Bands scoring
            bb_upper = data_point.get('BB_Upper')
            bb_lower = data_point.get('BB_Lower')
            
            if close and bb_lower and close < bb_lower:
                score += 0.1  # Below lower band - potential reversal
            elif close and bb_upper and close > bb_upper:
                score -= 0.1  # Above upper band - overbought
            
            # Ensure score is between 0 and 1
            score = max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
        
        return score
    
    def generate_recommendations(self, analysis_df=None, symbols=None):
        """
        Generate recommendations for stocks
        
        Args:
            analysis_df: DataFrame with analysis results (performs analysis if None)
            symbols: List of stock symbols
        
        Returns:
            DataFrame with recommendations
        """
        if analysis_df is None:
            analysis_df = self.analyze_stocks(symbols)
        
        if analysis_df.empty:
            logger.error("No data to generate recommendations")
            return pd.DataFrame()
        
        logger.info("Generating recommendations")
        
        recommendations_df = self.recommender.generate_portfolio_recommendations(analysis_df)
        
        return recommendations_df
    
    def run(self, output_file=None):
        """
        Run the complete recommendation system
        
        Args:
            output_file: Path to save recommendations (optional)
        
        Returns:
            DataFrame with recommendations
        """
        logger.info("=" * 60)
        logger.info("Starting Stock Recommendation System")
        logger.info("=" * 60)
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        if recommendations.empty:
            logger.error("No recommendations generated")
            return recommendations
        
        # Generate report
        report = self.recommender.generate_report(recommendations)
        
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total stocks analyzed: {report['total_stocks']}")
        logger.info(f"Recommendation distribution: {report['recommendation_distribution']}")
        logger.info(f"Average composite score: {report['average_composite_score']:.3f}")
        logger.info(f"Top 5 BUY recommendations: {', '.join(report['top_5_buy'])}")
        logger.info(f"High confidence recommendations: {report['high_confidence_count']}")
        logger.info("=" * 60 + "\n")
        
        # Display top recommendations
        logger.info("\nTop 10 Stock Recommendations:")
        logger.info(recommendations.head(10).to_string(index=False))
        
        # Save to file
        if output_file:
            recommendations.to_csv(output_file, index=False)
            logger.info(f"\nRecommendations saved to {output_file}")
        
        return recommendations


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Stock Recommendation System for Indian Markets (NSE/BSE)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file',
        default=None
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for recommendations',
        default='output/recommendations.csv'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Specific stock symbols to analyze (e.g., RELIANCE.NS TCS.NS)',
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize and run system
        system = StockRecommendationSystem(config_path=args.config)
        
        if args.symbols:
            logger.info(f"Analyzing specific symbols: {args.symbols}")
            recommendations = system.generate_recommendations(symbols=args.symbols)
        else:
            recommendations = system.run(output_file=args.output)
        
        if recommendations.empty:
            logger.error("No recommendations generated")
            sys.exit(1)
        
        logger.info("\nStock Recommendation System completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running Stock Recommendation System: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
