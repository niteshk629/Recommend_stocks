"""Calculate technical indicators for stocks"""
import pandas as pd
import numpy as np
from typing import Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock analysis"""
    
    def __init__(self):
        """Initialize technical indicators calculator"""
        pass
    
    def calculate_sma(self, df: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            df: DataFrame with stock data
            window: Window size for SMA
            column: Column to calculate SMA on
        
        Returns:
            Series with SMA values
        """
        return df[column].rolling(window=window).mean()
    
    def calculate_ema(self, df: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            df: DataFrame with stock data
            window: Window size for EMA
            column: Column to calculate EMA on
        
        Returns:
            Series with EMA values
        """
        return df[column].ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, df: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            df: DataFrame with stock data
            window: Window size for RSI
            column: Column to calculate RSI on
        
        Returns:
            Series with RSI values
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                       signal: int = 9, column: str = 'Close') -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with stock data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on
        
        Returns:
            DataFrame with MACD, signal, and histogram
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, 
                                   num_std: int = 2, column: str = 'Close') -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with stock data
            window: Window size for moving average
            num_std: Number of standard deviations
            column: Column to calculate bands on
        
        Returns:
            DataFrame with upper, middle, and lower bands
        """
        sma = df[column].rolling(window=window).mean()
        std = df[column].rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band
        })
    
    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with stock data
            window: Window size for ATR
        
        Returns:
            Series with ATR values
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def calculate_volume_ma(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate Volume Moving Average
        
        Args:
            df: DataFrame with stock data
            window: Window size
        
        Returns:
            Series with volume MA
        """
        return df['Volume'].rolling(window=window).mean()
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            DataFrame with all indicators
        """
        try:
            df = df.copy()
            
            # Moving Averages
            df['SMA_20'] = self.calculate_sma(df, 20)
            df['SMA_50'] = self.calculate_sma(df, 50)
            df['SMA_200'] = self.calculate_sma(df, 200)
            df['EMA_12'] = self.calculate_ema(df, 12)
            df['EMA_26'] = self.calculate_ema(df, 26)
            
            # RSI
            df['RSI'] = self.calculate_rsi(df)
            
            # MACD
            macd_data = self.calculate_macd(df)
            df['MACD'] = macd_data['MACD']
            df['MACD_Signal'] = macd_data['Signal']
            df['MACD_Histogram'] = macd_data['Histogram']
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(df)
            df['BB_Upper'] = bb_data['BB_Upper']
            df['BB_Middle'] = bb_data['BB_Middle']
            df['BB_Lower'] = bb_data['BB_Lower']
            
            # ATR
            df['ATR'] = self.calculate_atr(df)
            
            # Volume MA
            df['Volume_MA'] = self.calculate_volume_ma(df)
            
            # Price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
            df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
            
            logger.info(f"Added technical indicators to DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
