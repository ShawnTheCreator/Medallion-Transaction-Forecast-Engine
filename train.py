"""
Training script for Nedbank Transaction Forecasting.
Orchestrates the full ML pipeline from data loading to model training.
"""

import os
import sys
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.data_preprocessing import (
    create_preprocessing_pipeline,
    TargetTransformer,
    TemporalFeatureEngineer,
    NumericTransformer
)
from src.categorical_encoder import create_categorical_encoders
from src.validation import TimeSeriesCrossValidator, compute_rmsle
from src.model import get_model_from_config, train_with_cv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(config):
    """Load training and test data."""
    data_cfg = config['data']
    
    logger.info(f"Loading training data from {data_cfg.train_path}")
    train_df = pd.read_csv(data_cfg.train_path)
    
    logger.info(f"Loaded {len(train_df)} training samples")
    logger.info(f"Columns: {list(train_df.columns)}")
    
    # Load test data if available
    test_df = None
    if os.path.exists(data_cfg.test_path):
        logger.info(f"Loading test data from {data_cfg.test_path}")
        test_df = pd.read_csv(data_cfg.test_path)
        logger.info(f"Loaded {len(test_df)} test samples")
    
    return train_df, test_df


def prepare_features(train_df, test_df, config):
    """
    Prepare features using the preprocessing pipeline.
    Handles temporal features, categorical encoding, and target transformation.
    """
    data_cfg = config['data']
    
    # Separate target
    y = train_df[data_cfg.target_col].values
    
    # Create preprocessing components
    temporal_engineer, numeric_transformer, id_transformer, target_transformer = create_preprocessing_pipeline(config)
    
    # Fit and transform on training data
    logger.info("Converting numeric IDs to categorical...")
    train_processed = id_transformer.fit_transform(train_df)
    
    logger.info("Creating temporal features...")
    train_processed = temporal_engineer.fit_transform(train_processed)
    
    logger.info("Transforming numeric features...")
    train_processed = numeric_transformer.fit_transform(train_processed)
    
    # Categorical encoding - use TF-IDF for text descriptions (captures semantic meaning)
    logger.info("Encoding categorical features...")
    cat_encoder = create_categorical_encoders(config, text_encoding_method='tfidf')
    cat_encoder.fit(train_processed, y)
    train_processed = cat_encoder.transform(train_processed)
    
    # Drop non-feature columns
    drop_cols = [data_cfg.target_col, data_cfg.date_col, data_cfg.customer_id_col]
    feature_cols = [col for col in train_processed.columns if col not in drop_cols]
    X = train_processed[feature_cols]
    
    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Features: {list(feature_cols)[:10]}... (showing first 10)")
    
    # Process test data if available
    X_test = None
    if test_df is not None:
        test_processed = id_transformer.transform(test_df)
        test_processed = temporal_engineer.transform(test_processed)
        test_processed = numeric_transformer.transform(test_processed)
        test_processed = cat_encoder.transform(test_processed)
        X_test = test_processed[feature_cols]
        logger.info(f"Test feature matrix shape: {X_test.shape}")
    
    return X, y, X_test, target_transformer, {
        'temporal_engineer': temporal_engineer,
        'numeric_transformer': numeric_transformer,
        'id_transformer': id_transformer,
        'cat_encoder': cat_encoder,
        'feature_cols': feature_cols
    }


def train_model(X, y, config, target_transformer):
    """Train model with time series cross-validation."""
    model_cfg = config['model']
    
    # Create time series CV splitter
    cv_splitter = TimeSeriesCrossValidator(
        n_splits=model_cfg.n_splits,
        date_col=config['data'].date_col,
        gap_months=model_cfg.cv_gap_months
    )
    
    # Train with CV
    model, cv_results = train_with_cv(X, y, config, cv_splitter, target_transformer)
    
    return model, cv_results


def save_artifacts(model, preprocessors, target_transformer, feature_importance, config, output_dir='outputs'):
    """Save model and preprocessing artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config['data'].model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(config['data'].model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessors
    preprocessors_path = os.path.join(config['data'].model_dir, 'preprocessors.pkl')
    with open(preprocessors_path, 'wb') as f:
        pickle.dump(preprocessors, f)
    logger.info(f"Preprocessors saved to {preprocessors_path}")
    
    # Save target transformer
    target_transformer_path = os.path.join(config['data'].model_dir, 'target_transformer.pkl')
    with open(target_transformer_path, 'wb') as f:
        pickle.dump(target_transformer, f)
    logger.info(f"Target transformer saved to {target_transformer_path}")
    
    # Save feature importance
    if feature_importance is not None:
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Save OOF predictions
    if 'oof_predictions' in cv_results:
        oof_path = os.path.join(output_dir, 'oof_predictions.csv')
        pd.DataFrame({
            'oof_prediction': cv_results['oof_predictions'],
            'actual': y
        }).to_csv(oof_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Train transaction forecasting model')
    parser.add_argument('--train-path', type=str, help='Path to training data')
    parser.add_argument('--test-path', type=str, help='Path to test data')
    parser.add_argument('--model-type', type=str, default='lightgbm', 
                       choices=['xgboost', 'lightgbm', 'catboost'],
                       help='Model type to use')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Override config with command line args
    if args.train_path:
        config['data'].train_path = args.train_path
    if args.test_path:
        config['data'].test_path = args.test_path
    if args.model_type:
        config['model'].model_type = args.model_type
    if args.n_splits:
        config['model'].n_splits = args.n_splits
    
    logger.info("="*60)
    logger.info("Transaction Forecasting Model Training")
    logger.info("="*60)
    logger.info(f"Model type: {config['model'].model_type}")
    logger.info(f"CV folds: {config['model'].n_splits}")
    logger.info(f"Log transform: {config['model'].use_log_transform}")
    logger.info("="*60)
    
    # Load data
    train_df, test_df = load_data(config)
    
    # Prepare features
    X, y, X_test, target_transformer, preprocessors = prepare_features(train_df, test_df, config)
    
    # Train model
    model, cv_results = train_model(X, y, config, target_transformer)
    
    # Get feature importance
    feature_importance = None
    if hasattr(model, 'get_feature_importance'):
        feature_importance = model.get_feature_importance(preprocessors['feature_cols'])
        logger.info("\nTop 10 most important features:")
        logger.info(feature_importance.head(10).to_string())
    
    # Save artifacts
    save_artifacts(model, preprocessors, target_transformer, feature_importance, config, args.output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info(f"Final CV RMSLE: {cv_results['overall_rmsle']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
