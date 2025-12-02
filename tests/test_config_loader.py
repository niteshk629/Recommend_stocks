"""Tests for configuration loader"""
import pytest
from src.utils.config_loader import ConfigLoader


def test_config_loader_initialization():
    """Test configuration loader initialization"""
    config = ConfigLoader()
    assert config.config is not None
    assert isinstance(config.config, dict)


def test_get_stock_universe():
    """Test getting stock universe"""
    config = ConfigLoader()
    universe = config.get_stock_universe()
    assert isinstance(universe, list)
    assert len(universe) > 0
    assert 'RELIANCE.NS' in universe


def test_get_config_value():
    """Test getting configuration values"""
    config = ConfigLoader()
    
    # Test existing key
    period = config.get('data_collection.historical_period')
    assert period is not None
    
    # Test non-existing key with default
    value = config.get('non.existing.key', 'default')
    assert value == 'default'


def test_get_recommendation_settings():
    """Test getting recommendation settings"""
    config = ConfigLoader()
    settings = config.get_recommendation_settings()
    assert isinstance(settings, dict)
    assert 'thresholds' in settings
