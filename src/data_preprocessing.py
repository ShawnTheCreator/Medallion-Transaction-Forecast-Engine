"""
Data preprocessing pipeline for transaction forecasting.
Handles target transformation, temporal feature engineering, and data cleaning.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TargetTransformer:
    """
    Handles log1p transformation for RMSLE optimization.
    Transform: log1p(y) = log(y + 1)
    Inverse: expm1(y) = exp(y) - 1
    """
    
    def __init__(self, use_transform: bool = True):
        self.use_transform = use_transform
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply log1p transformation."""
        if not self.use_transform:
            return y
        y = np.asarray(y)
        # Ensure non-negative
        y = np.maximum(y, 0)
        return np.log1p(y)
    
    def inverse_transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply expm1 to get back to original scale."""
        if not self.use_transform:
            return y_pred
        y_pred = np.asarray(y_pred)
        result = np.expm1(y_pred)
        # Ensure non-negative predictions
        return np.maximum(result, 0)


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates temporal features from transaction history.
    Captures seasonality, velocity, and trends.
    """
    
    def __init__(self, 
                 customer_id_col: str = 'customer_id',
                 date_col: str = 'transaction_date',
                 target_col: str = 'txn_count',
                 seasonal_lags: List[int] = None,
                 short_term_lags: List[int] = None,
                 rolling_windows: List[int] = None,
                 rolling_stats: List[str] = None,
                 velocity_windows: List[Tuple[int, int]] = None):
        
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.target_col = target_col
        self.seasonal_lags = seasonal_lags or [12, 11, 13]
        self.short_term_lags = short_term_lags or [1, 2, 3, 6]
        self.rolling_windows = rolling_windows or [1, 3, 6, 12]
        self.rolling_stats = rolling_stats or ['mean', 'std', 'min', 'max']
        self.velocity_windows = velocity_windows or [(3, 12), (1, 12), (3, 6)]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features."""
        df = X.copy()
        
        # Ensure date is datetime
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values([self.customer_id_col, self.date_col])
        
        # Create month and quarter features for seasonality
        if self.date_col in df.columns:
            df['month'] = df[self.date_col].dt.month
            df['quarter'] = df[self.date_col].dt.quarter
            df['is_nov_jan'] = df['month'].isin([11, 12, 1]).astype(int)
            df['is_holiday_season'] = df['month'].isin([10, 11, 12, 1]).astype(int)
        
        # Generate lag features
        df = self._create_lag_features(df)
        
        # Generate rolling statistics
        df = self._create_rolling_features(df)
        
        # Generate velocity ratios
        df = self._create_velocity_features(df)
        
        # Generate trend features
        df = self._create_trend_features(df)
        
        logger.info(f"Created {len(df.columns)} total features")
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged transaction counts."""
        # Short-term lags (recent history)
        for lag in self.short_term_lags:
            col_name = f'{self.target_col}_lag_{lag}m'
            df[col_name] = df.groupby(self.customer_id_col)[self.target_col].shift(lag)
        
        # Seasonal lags (year-over-year comparison)
        for lag in self.seasonal_lags:
            col_name = f'{self.target_col}_lag_{lag}m_seasonal'
            df[col_name] = df.groupby(self.customer_id_col)[self.target_col].shift(lag)
        
        logger.info(f"Created {len(self.short_term_lags) + len(self.seasonal_lags)} lag features")
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        for window in self.rolling_windows:
            for stat in self.rolling_stats:
                col_name = f'{self.target_col}_rolling_{stat}_{window}m'
                
                if stat == 'mean':
                    df[col_name] = df.groupby(self.customer_id_col)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean())
                elif stat == 'std':
                    df[col_name] = df.groupby(self.customer_id_col)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).std())
                elif stat == 'min':
                    df[col_name] = df.groupby(self.customer_id_col)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).min())
                elif stat == 'max':
                    df[col_name] = df.groupby(self.customer_id_col)[self.target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).max())
        
        logger.info(f"Created {len(self.rolling_windows) * len(self.rolling_stats)} rolling features")
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity ratio features (recent vs historical)."""
        for recent, historical in self.velocity_windows:
            recent_col = f'{self.target_col}_rolling_mean_{recent}m'
            historical_col = f'{self.target_col}_lag_{historical}m_seasonal' if historical == 12 else f'{self.target_col}_lag_{historical}m'
            
            if recent_col in df.columns and historical_col in df.columns:
                col_name = f'velocity_{recent}m_vs_{historical}m'
                # Add small constant to avoid division by zero
                df[col_name] = df[recent_col] / (df[historical_col] + 1)
        
        logger.info(f"Created {len(self.velocity_windows)} velocity features")
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend-based features."""
        # Recent trend (last 3 months vs previous 3 months)
        df['trend_3m'] = (
            df.groupby(self.customer_id_col)[self.target_col].shift(0) - 
            df.groupby(self.customer_id_col)[self.target_col].shift(3)
        )
        
        # Year-over-year growth rate
        lag_12 = f'{self.target_col}_lag_12m_seasonal'
        if lag_12 in df.columns:
            df['yoy_growth'] = (
                (df[self.target_col] - df[lag_12]) / (df[lag_12] + 1)
            )
        
        logger.info("Created trend features")
        return df


class NumericTransformer(BaseEstimator, TransformerMixin):
    """
    Handles skewed numeric features with log transformation.
    Also handles missing values and outliers.
    """
    
    def __init__(self, 
                 skewed_cols: List[str] = None,
                 log_transform: bool = True,
                 impute_strategy: str = 'median'):
        self.skewed_cols = skewed_cols or []
        self.log_transform = log_transform
        self.impute_strategy = impute_strategy
        self.impute_values_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Compute imputation values from training data."""
        for col in self.skewed_cols:
            if col in X.columns:
                if self.impute_strategy == 'median':
                    self.impute_values_[col] = X[col].median()
                elif self.impute_strategy == 'mean':
                    self.impute_values_[col] = X[col].mean()
                else:
                    self.impute_values_[col] = 0
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations."""
        df = X.copy()
        
        for col in self.skewed_cols:
            if col not in df.columns:
                continue
                
            # Impute missing values
            if col in self.impute_values_:
                df[col] = df[col].fillna(self.impute_values_[col])
            
            # Log transform for skewed features (add 1 to handle zeros)
            if self.log_transform:
                log_col = f'{col}_log'
                df[log_col] = np.log1p(df[col].clip(lower=0))
                
                # Add is_missing indicator
                df[f'{col}_is_missing'] = (df[col] == 0).astype(int)
        
        return df


class NumericIdToCategorical(BaseEstimator, TransformerMixin):
    """
    Converts numeric ID columns to categorical strings.
    
    Banking datasets often use numeric IDs (e.g., branch_code=101, region_id=5)
    that should be treated as categorical, not continuous linear variables.
    This ensures target encoding or CatBoost handle them correctly.
    """
    
    def __init__(self,
                 id_cols: List[str] = None,
                 prefix: str = 'id_'):
        """
        Args:
            id_cols: List of column names that are numeric IDs but categorical
            prefix: Prefix to add to converted string values
        """
        self.id_cols = id_cols or []
        self.prefix = prefix
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric ID columns to categorical strings."""
        df = X.copy()
        
        for col in self.id_cols:
            if col not in df.columns:
                continue
            
            # Convert to string with prefix (e.g., 101 -> "id_101")
            # This ensures model treats it as categorical, not continuous
            df[col] = self.prefix + df[col].astype(str)
            logger.info(f"Converted numeric ID '{col}' to categorical string")
        
        return df


def create_preprocessing_pipeline(config: Dict) -> Tuple:
    """
    Create the full preprocessing pipeline.
    
    Returns:
        Tuple of (temporal_engineer, numeric_transformer, id_transformer, target_transformer)
    """
    data_cfg = config['data']
    feature_cfg = config['features']
    model_cfg = config['model']
    
    # Target transformer
    target_transformer = TargetTransformer(use_transform=model_cfg.use_log_transform)
    
    # Temporal feature engineer
    temporal_engineer = TemporalFeatureEngineer(
        customer_id_col=data_cfg.customer_id_col,
        date_col=data_cfg.date_col,
        target_col=data_cfg.target_col,
        seasonal_lags=feature_cfg.seasonal_lags,
        short_term_lags=feature_cfg.short_term_lags,
        rolling_windows=feature_cfg.rolling_windows,
        rolling_stats=feature_cfg.rolling_stats,
        velocity_windows=feature_cfg.velocity_windows
    )
    
    # Numeric transformer for skewed features
    numeric_transformer = NumericTransformer(
        skewed_cols=data_cfg.skewed_numeric_cols,
        log_transform=True
    )
    
    # ID transformer to convert numeric IDs to categorical strings
    id_transformer = NumericIdToCategorical(
        id_cols=data_cfg.numeric_id_cols,
        prefix='id_'
    )
    
    return temporal_engineer, numeric_transformer, id_transformer, target_transformer
