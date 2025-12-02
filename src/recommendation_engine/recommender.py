"""Stock recommendation engine"""
import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class StockRecommender:
    """Generate buy/sell/hold recommendations for stocks"""
    
    def __init__(self, thresholds: Dict = None):
        """
        Initialize stock recommender
        
        Args:
            thresholds: Dictionary with recommendation thresholds
        """
        self.thresholds = thresholds or {
            'strong_buy': 0.75,
            'buy': 0.60,
            'hold': 0.40,
            'sell': 0.25,
            'strong_sell': 0.0
        }
    
    def calculate_composite_score(self, technical_score: float, 
                                   fundamental_score: float,
                                   sentiment_score: float,
                                   weights: Dict = None) -> float:
        """
        Calculate composite score from multiple signals
        
        Args:
            technical_score: Score from technical analysis (0-1)
            fundamental_score: Score from fundamental analysis (0-1)
            sentiment_score: Score from sentiment analysis (0-1)
            weights: Dictionary with weights for each component
        
        Returns:
            Composite score (0-1)
        """
        weights = weights or {
            'technical': 0.4,
            'fundamental': 0.4,
            'sentiment': 0.2
        }
        
        composite_score = (
            technical_score * weights['technical'] +
            fundamental_score * weights['fundamental'] +
            sentiment_score * weights['sentiment']
        )
        
        return composite_score
    
    def get_recommendation(self, score: float) -> str:
        """
        Get recommendation based on composite score
        
        Args:
            score: Composite score (0-1)
        
        Returns:
            Recommendation string
        """
        if score >= self.thresholds['strong_buy']:
            return 'STRONG BUY'
        elif score >= self.thresholds['buy']:
            return 'BUY'
        elif score >= self.thresholds['hold']:
            return 'HOLD'
        elif score >= self.thresholds['sell']:
            return 'SELL'
        else:
            return 'STRONG SELL'
    
    def get_recommendation_confidence(self, score: float) -> str:
        """
        Get confidence level for recommendation
        
        Args:
            score: Composite score
        
        Returns:
            Confidence level
        """
        # Calculate distance from threshold boundaries
        thresholds = sorted(self.thresholds.values(), reverse=True)
        
        # Find closest threshold
        min_distance = min([abs(score - t) for t in thresholds])
        
        if min_distance > 0.15:
            return 'HIGH'
        elif min_distance > 0.08:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_recommendation(self, symbol: str, technical_score: float,
                               fundamental_score: float, sentiment_score: float,
                               additional_data: Dict = None) -> Dict:
        """
        Generate complete recommendation for a stock
        
        Args:
            symbol: Stock symbol
            technical_score: Technical analysis score
            fundamental_score: Fundamental analysis score
            sentiment_score: Sentiment analysis score
            additional_data: Additional data to include
        
        Returns:
            Dictionary with recommendation details
        """
        composite_score = self.calculate_composite_score(
            technical_score, fundamental_score, sentiment_score
        )
        
        recommendation = self.get_recommendation(composite_score)
        confidence = self.get_recommendation_confidence(composite_score)
        
        result = {
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': confidence,
            'composite_score': composite_score,
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'sentiment_score': sentiment_score,
        }
        
        if additional_data:
            result.update(additional_data)
        
        logger.info(f"{symbol}: {recommendation} (Score: {composite_score:.3f}, Confidence: {confidence})")
        
        return result
    
    def generate_portfolio_recommendations(self, stocks_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate recommendations for multiple stocks
        
        Args:
            stocks_data: DataFrame with stock data and scores
        
        Returns:
            DataFrame with recommendations
        """
        recommendations = []
        
        for _, row in stocks_data.iterrows():
            rec = self.generate_recommendation(
                symbol=row['symbol'],
                technical_score=row.get('technical_score', 0.5),
                fundamental_score=row.get('fundamental_score', 0.5),
                sentiment_score=row.get('sentiment_score', 0.5),
                additional_data={
                    'current_price': row.get('current_price'),
                    'market_cap': row.get('market_cap'),
                    'sector': row.get('sector')
                }
            )
            recommendations.append(rec)
        
        df = pd.DataFrame(recommendations)
        
        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False)
        
        logger.info(f"Generated recommendations for {len(df)} stocks")
        return df
    
    def get_top_recommendations(self, recommendations_df: pd.DataFrame, 
                               n: int = 10, recommendation_type: str = 'BUY') -> pd.DataFrame:
        """
        Get top N recommendations of a specific type
        
        Args:
            recommendations_df: DataFrame with recommendations
            n: Number of recommendations to return
            recommendation_type: Type of recommendation to filter
        
        Returns:
            DataFrame with top recommendations
        """
        filtered = recommendations_df[
            recommendations_df['recommendation'].str.contains(recommendation_type, case=False)
        ]
        
        return filtered.head(n)
    
    def generate_report(self, recommendations_df: pd.DataFrame) -> Dict:
        """
        Generate summary report of recommendations
        
        Args:
            recommendations_df: DataFrame with recommendations
        
        Returns:
            Dictionary with report summary
        """
        report = {
            'total_stocks': len(recommendations_df),
            'recommendation_distribution': recommendations_df['recommendation'].value_counts().to_dict(),
            'average_composite_score': recommendations_df['composite_score'].mean(),
            'top_5_buy': recommendations_df[
                recommendations_df['recommendation'].isin(['BUY', 'STRONG BUY'])
            ].head(5)['symbol'].tolist(),
            'top_5_avoid': recommendations_df[
                recommendations_df['recommendation'].isin(['SELL', 'STRONG SELL'])
            ].head(5)['symbol'].tolist(),
            'high_confidence_count': len(recommendations_df[recommendations_df['confidence'] == 'HIGH'])
        }
        
        return report
