"""Configuration loader utility"""
import yaml
import os
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration settings"""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "config.yaml"
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def get_stock_universe(self):
        """Get list of stocks to analyze"""
        return self.config.get('stock_universe', {}).get('nse', [])
    
    def get_data_collection_settings(self):
        """Get data collection settings"""
        return self.config.get('data_collection', {})
    
    def get_model_settings(self, model_name):
        """Get settings for specific model"""
        return self.config.get('models', {}).get(model_name, {})
    
    def get_recommendation_settings(self):
        """Get recommendation settings"""
        return self.config.get('recommendation', {})
