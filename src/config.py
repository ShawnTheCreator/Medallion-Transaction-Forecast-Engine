"""
Configuration settings for the Nedbank Transaction Forecast pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data paths and loading configuration."""
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    sample_submission_path: str = "data/sample_submission.csv"
    output_dir: str = "outputs"
    model_dir: str = "models"
    
    # Target column
    target_col: str = "next_3m_txn_count"
    
    # ID columns
    customer_id_col: str = "customer_id"
    
    # Date column
    date_col: str = "transaction_date"
    
    # High cardinality categorical columns
    high_cardinality_cols: List[str] = field(default_factory=lambda: [
        "transaction_description",
        "merchant_category",
        "transaction_channel"
    ])
    
    # Numeric columns requiring transformation
    skewed_numeric_cols: List[str] = field(default_factory=lambda: [
        "income",
        "transaction_amount"
    ])
    
    # Numeric ID columns that should be treated as categorical (not continuous)
    # e.g., branch_code=101, region_id=5 - these are IDs, not linear quantities
    numeric_id_cols: List[str] = field(default_factory=lambda: [
        "branch_code",
        "region_id",
        "area_code",
        "district_id",
        "office_code",
        "segment_code",
        "sector_id"
    ])


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    
    # Temporal lag features
    seasonal_lags: List[int] = field(default_factory=lambda: [12, 11, 13])  # Annual seasonality
    short_term_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 6])
    
    # Rolling window sizes
    rolling_windows: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    
    # Rolling statistics to compute
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    
    # Velocity ratios (current vs historical)
    velocity_windows: List[tuple] = field(default_factory=lambda: [
        (3, 12),   # 3m avg vs 12m ago
        (1, 12),   # 1m vs 12m ago (YoY)
        (3, 6),    # 3m vs 6m ago
    ])
    
    # Target encoding smoothing parameter
    target_encoding_min_samples: int = 30
    target_encoding_noise: float = 0.01
    
    # Feature hashing for text features
    hash_vector_size: int = 1000


@dataclass
class ModelConfig:
    """Model training configuration."""
    
    # Model type: 'xgboost', 'lightgbm', 'catboost'
    model_type: str = "lightgbm"
    
    # Target transformation
    use_log_transform: bool = True
    
    # Cross-validation
    n_splits: int = 5
    cv_gap_months: int = 3  # Purge gap between train and validation
    
    # LightGBM hyperparameters
    lgb_params: dict = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 2000,
        'early_stopping_rounds': 100,
        'random_state': 42,
        'n_jobs': -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
    })
    
    # XGBoost hyperparameters
    xgb_params: dict = field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 2000,
        'early_stopping_rounds': 100,
        'random_state': 42,
        'n_jobs': -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_weight': 3,
        'gamma': 0.1,
    })
    
    # CatBoost hyperparameters
    catboost_params: dict = field(default_factory=lambda: {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'depth': 8,
        'learning_rate': 0.05,
        'iterations': 2000,
        'early_stopping_rounds': 100,
        'random_seed': 42,
        'verbose': 100,
        'l2_leaf_reg': 3,
        'bagging_temperature': 0.5,
        'border_count': 128,
    })


def get_config():
    """Get default configuration."""
    return {
        'data': DataConfig(),
        'features': FeatureConfig(),
        'model': ModelConfig(),
    }
