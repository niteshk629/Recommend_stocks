"""Tests for recommendation engine"""
import pytest
from src.recommendation_engine.recommender import StockRecommender


def test_recommender_initialization():
    """Test recommender initialization"""
    recommender = StockRecommender()
    assert recommender.thresholds is not None
    assert 'strong_buy' in recommender.thresholds


def test_calculate_composite_score():
    """Test composite score calculation"""
    recommender = StockRecommender()
    
    score = recommender.calculate_composite_score(
        technical_score=0.7,
        fundamental_score=0.8,
        sentiment_score=0.6
    )
    
    assert 0 <= score <= 1
    assert isinstance(score, float)


def test_get_recommendation():
    """Test recommendation generation"""
    recommender = StockRecommender()
    
    # Test strong buy
    rec = recommender.get_recommendation(0.80)
    assert rec == 'STRONG BUY'
    
    # Test buy
    rec = recommender.get_recommendation(0.65)
    assert rec == 'BUY'
    
    # Test hold
    rec = recommender.get_recommendation(0.50)
    assert rec == 'HOLD'
    
    # Test sell
    rec = recommender.get_recommendation(0.30)
    assert rec == 'SELL'
    
    # Test strong sell
    rec = recommender.get_recommendation(0.10)
    assert rec == 'STRONG SELL'


def test_get_recommendation_confidence():
    """Test confidence level calculation"""
    recommender = StockRecommender()
    
    confidence = recommender.get_recommendation_confidence(0.90)
    assert confidence in ['HIGH', 'MEDIUM', 'LOW']


def test_generate_recommendation():
    """Test complete recommendation generation"""
    recommender = StockRecommender()
    
    rec = recommender.generate_recommendation(
        symbol='TEST.NS',
        technical_score=0.7,
        fundamental_score=0.6,
        sentiment_score=0.5
    )
    
    assert rec['symbol'] == 'TEST.NS'
    assert 'recommendation' in rec
    assert 'confidence' in rec
    assert 'composite_score' in rec
    assert 0 <= rec['composite_score'] <= 1
