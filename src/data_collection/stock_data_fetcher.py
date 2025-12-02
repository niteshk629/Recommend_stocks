"""Fetch historical stock data from NSE/BSE"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class StockDataFetcher:
    """Fetch historical stock data for Indian markets"""
    
    def __init__(self, period='5y', interval='1d'):
        """
        Initialize stock data fetcher
        
        Args:
            period: Historical period to fetch (e.g., '1y', '5y')
            interval: Data interval (e.g., '1d', '1wk')
        """
        self.period = period
        self.interval = interval
    
    def fetch_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
        
        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            stock = yf.Ticker(symbol)
            df = stock.history(period=self.period, interval=self.interval)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            df['Symbol'] = symbol
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str], delay: float = 0.5) -> pd.DataFrame:
        """
        Fetch historical data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            delay: Delay between requests to avoid rate limiting
        
        Returns:
            Combined DataFrame with all stock data
        """
        all_data = []
        
        for symbol in symbols:
            df = self.fetch_stock_data(symbol)
            if df is not None:
                all_data.append(df)
            time.sleep(delay)  # Rate limiting
        
        if all_data:
            combined_df = pd.concat(all_data, axis=0)
            logger.info(f"Fetched data for {len(all_data)} stocks")
            return combined_df
        else:
            logger.warning("No data fetched for any stock")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> dict:
        """
        Get basic information about a stock
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'INR')
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """
        Save stock data to CSV
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
        """
        try:
            df.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
