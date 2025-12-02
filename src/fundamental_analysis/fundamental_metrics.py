"""Calculate fundamental metrics for stocks"""
import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FundamentalAnalyzer:
    """Analyze fundamental metrics of stocks"""
    
    def __init__(self):
        """Initialize fundamental analyzer"""
        pass
    
    def get_fundamental_metrics(self, symbol: str) -> Dict:
        """
        Get fundamental metrics for a stock
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            logger.info(f"Fetching fundamental metrics for {symbol}")
            stock = yf.Ticker(symbol)
            info = stock.info
            
            metrics = {
                'symbol': symbol,
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'eps': info.get('trailingEps', None),
                'forward_eps': info.get('forwardEps', None),
                'book_value': info.get('bookValue', None),
                'market_cap': info.get('marketCap', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'beta': info.get('beta', None),
                'dividend_yield': info.get('dividendYield', None),
                'payout_ratio': info.get('payoutRatio', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
            }
            
            logger.info(f"Successfully fetched fundamental metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching fundamental metrics for {symbol}: {e}")
            return {'symbol': symbol}
    
    def get_financial_statements(self, symbol: str) -> Dict:
        """
        Get financial statements for a stock
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with financial statements
        """
        try:
            stock = yf.Ticker(symbol)
            
            return {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow
            }
        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {e}")
            return {}
    
    def calculate_valuation_score(self, metrics: Dict) -> float:
        """
        Calculate valuation score based on fundamental metrics
        
        Args:
            metrics: Dictionary with fundamental metrics
        
        Returns:
            Valuation score (0-1)
        """
        score = 0.5  # Default neutral score
        
        try:
            # PE Ratio scoring (lower is better, but not too low)
            pe = metrics.get('pe_ratio')
            if pe and 0 < pe < 15:
                score += 0.15
            elif pe and 15 <= pe < 25:
                score += 0.10
            
            # PB Ratio scoring (lower is better)
            pb = metrics.get('pb_ratio')
            if pb and 0 < pb < 1.5:
                score += 0.10
            elif pb and 1.5 <= pb < 3:
                score += 0.05
            
            # ROE scoring (higher is better)
            roe = metrics.get('roe')
            if roe and roe > 0.15:
                score += 0.15
            elif roe and roe > 0.10:
                score += 0.10
            
            # Profit Margin scoring
            profit_margin = metrics.get('profit_margin')
            if profit_margin and profit_margin > 0.15:
                score += 0.10
            elif profit_margin and profit_margin > 0.10:
                score += 0.05
            
            # Debt to Equity scoring (lower is better)
            de = metrics.get('debt_to_equity')
            if de and de < 0.5:
                score += 0.10
            elif de and de < 1.0:
                score += 0.05
            
            # Ensure score is between 0 and 1
            score = max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error calculating valuation score: {e}")
        
        return score
    
    def analyze_multiple_stocks(self, symbols: list) -> pd.DataFrame:
        """
        Analyze fundamental metrics for multiple stocks
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            DataFrame with fundamental analysis
        """
        results = []
        
        for symbol in symbols:
            metrics = self.get_fundamental_metrics(symbol)
            metrics['valuation_score'] = self.calculate_valuation_score(metrics)
            results.append(metrics)
        
        df = pd.DataFrame(results)
        logger.info(f"Analyzed {len(df)} stocks")
        return df
