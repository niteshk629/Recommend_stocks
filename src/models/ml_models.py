"""Machine learning models for stock recommendation"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
from typing import Tuple, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class StockMLModels:
    """Machine learning models for stock prediction"""
    
    def __init__(self):
        """Initialize ML models"""
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training
        
        Args:
            df: DataFrame with stock data and indicators
        
        Returns:
            Tuple of (features, target)
        """
        # Define features to use
        technical_features = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI',
            'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
            'ATR', 'Volume_MA', 'Price_Change_5d'
        ]
        
        # Create target variable (1 if price increases in next 5 days, 0 otherwise)
        df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
        
        # Select features that exist in the dataframe
        available_features = [f for f in technical_features if f in df.columns]
        
        # Drop rows with NaN values
        df_clean = df[available_features + ['Target']].dropna()
        
        X = df_clean[available_features]
        y = df_clean['Target']
        
        self.feature_names = available_features
        logger.info(f"Prepared {len(available_features)} features with {len(X)} samples")
        
        return X, y
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           n_estimators: int = 100, max_depth: int = 10) -> RandomForestClassifier:
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
        
        Returns:
            Trained model
        """
        logger.info("Training Random Forest model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        
        logger.info("Random Forest model trained successfully")
        return self.rf_model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      n_estimators: int = 100, max_depth: int = 6,
                      learning_rate: float = 0.1) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
        
        Returns:
            Trained model
        """
        logger.info("Training XGBoost model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_train_scaled, y_train)
        
        logger.info("XGBoost model trained successfully")
        return self.xgb_model
    
    def predict_random_forest(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using Random Forest
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        if self.rf_model is None:
            raise ValueError("Random Forest model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict_proba(X_scaled)[:, 1]
    
    def predict_xgboost(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using XGBoost
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.xgb_model.predict_proba(X_scaled)[:, 1]
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                       model_type: str = 'rf') -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            model_type: Type of model ('rf' or 'xgb')
        
        Returns:
            Dictionary with evaluation metrics
        """
        if model_type == 'rf':
            predictions = (self.predict_random_forest(X_test) > 0.5).astype(int)
        else:
            predictions = (self.predict_xgboost(X_test) > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        logger.info(f"{model_type.upper()} Model Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def get_feature_importance(self, model_type: str = 'rf') -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            model_type: Type of model ('rf' or 'xgb')
        
        Returns:
            DataFrame with feature importance
        """
        if model_type == 'rf' and self.rf_model:
            importance = self.rf_model.feature_importances_
        elif model_type == 'xgb' and self.xgb_model:
            importance = self.xgb_model.feature_importances_
        else:
            raise ValueError(f"Model {model_type} not trained")
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, filepath: str, model_type: str = 'rf'):
        """
        Save trained model
        
        Args:
            filepath: Path to save model
            model_type: Type of model to save
        """
        if model_type == 'rf' and self.rf_model:
            joblib.dump((self.rf_model, self.scaler, self.feature_names), filepath)
        elif model_type == 'xgb' and self.xgb_model:
            joblib.dump((self.xgb_model, self.scaler, self.feature_names), filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_type: str = 'rf'):
        """
        Load trained model
        
        Args:
            filepath: Path to load model from
            model_type: Type of model to load
        """
        model, scaler, feature_names = joblib.load(filepath)
        
        if model_type == 'rf':
            self.rf_model = model
        else:
            self.xgb_model = model
        
        self.scaler = scaler
        self.feature_names = feature_names
        
        logger.info(f"Model loaded from {filepath}")
