"""
Prediction script for generating submissions.
Loads trained model and generates predictions for test data.
"""

import os
import sys
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_artifacts(model_dir='models'):
    """Load trained model and preprocessors."""
    artifacts = {}
    
    # Load model
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        artifacts['model'] = pickle.load(f)
    logger.info(f"Loaded model from {model_path}")
    
    # Load preprocessors
    preprocessors_path = os.path.join(model_dir, 'preprocessors.pkl')
    with open(preprocessors_path, 'rb') as f:
        artifacts['preprocessors'] = pickle.load(f)
    logger.info(f"Loaded preprocessors from {preprocessors_path}")
    
    # Load target transformer
    target_transformer_path = os.path.join(model_dir, 'target_transformer.pkl')
    with open(target_transformer_path, 'rb') as f:
        artifacts['target_transformer'] = pickle.load(f)
    logger.info(f"Loaded target transformer from {target_transformer_path}")
    
    return artifacts


def make_predictions(test_df, artifacts):
    """
    Generate predictions for test data.
    
    CRITICAL: Submission must be in np.log1p format as per competition requirements.
    The scoring platform expects log-transformed predictions, not raw counts.
    
    Args:
        test_df: Test DataFrame with raw features
        artifacts: Dictionary containing model and preprocessors
    
    Returns:
        Predictions in np.log1p format (REQUIRED by submission platform)
    """
    model = artifacts['model']
    preprocessors = artifacts['preprocessors']
    
    # Extract preprocessors
    id_transformer = preprocessors['id_transformer']
    temporal_engineer = preprocessors['temporal_engineer']
    numeric_transformer = preprocessors['numeric_transformer']
    cat_encoder = preprocessors['cat_encoder']
    feature_cols = preprocessors['feature_cols']
    
    # Store original row count for validation
    original_row_count = len(test_df)
    logger.info(f"CRITICAL: Original test data has {original_row_count} rows. Must maintain this count.")
    
    # Apply preprocessing
    logger.info("Converting numeric IDs to categorical...")
    test_processed = id_transformer.transform(test_df)
    
    logger.info("Applying temporal feature engineering...")
    test_processed = temporal_engineer.transform(test_processed)
    
    logger.info("Applying numeric transformations...")
    test_processed = numeric_transformer.transform(test_processed)
    
    logger.info("Encoding categorical features...")
    test_processed = cat_encoder.transform(test_processed)
    
    # Select features
    X_test = test_processed[feature_cols]
    logger.info(f"Test feature matrix shape: {X_test.shape}")
    
    # Generate predictions (model outputs in log space)
    logger.info("Generating predictions...")
    y_pred_log = model.predict(X_test)
    
    # CRITICAL: Keep predictions in np.log1p format for submission
    # The scoring platform expects log-transformed values, not raw counts
    # Do NOT inverse transform with expm1
    y_pred_final = y_pred_log  # Already in log1p format
    
    logger.info(f"Predictions generated (log1p format): min={y_pred_final.min():.4f}, max={y_pred_final.max():.4f}, mean={y_pred_final.mean():.4f}")
    logger.info("WARNING: Submission is in np.log1p format as required by platform. Do not apply expm1.")
    
    return y_pred_final


def create_submission(test_df, predictions, customer_id_col='customer_id', output_path='submission.csv', expected_count=3584):
    """
    Create submission file in the required format.
    
    CRITICAL: Validates row count matches expected (3,584 for Nedbank challenge).
    An incomplete submission is automatic disqualification.
    """
    # Validate row count
    actual_count = len(predictions)
    if actual_count != expected_count:
        logger.error(f"ROW COUNT MISMATCH! Expected {expected_count}, got {actual_count}")
        logger.error("Some rows were dropped during feature engineering. CHECK IMMEDIATELY.")
        raise ValueError(f"Submission must have exactly {expected_count} rows. Got {actual_count}.")
    
    logger.info(f"CRITICAL CHECK PASSED: Submission has correct row count ({actual_count})")
    
    submission = pd.DataFrame({
        customer_id_col: test_df[customer_id_col],
        'next_3m_txn_count': predictions
    })
    
    # Note: For log1p format, values can be negative (log of small numbers)
    # The platform expects np.log1p(y_pred), so no clipping needed
    logger.info("Submission is in np.log1p format as required by scoring platform")
    
    # Save submission
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Prediction statistics:\n{submission['next_3m_txn_count'].describe()}")
    
    return submission


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    parser.add_argument('--test-path', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing trained model')
    parser.add_argument('--output-path', type=str, default='submission.csv', help='Output submission file path')
    parser.add_argument('--customer-id-col', type=str, default='customer_id', help='Name of customer ID column')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Transaction Forecasting - Prediction")
    logger.info("="*60)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_path}")
    test_df = pd.read_csv(args.test_path)
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Load model artifacts
    artifacts = load_artifacts(args.model_dir)
    
    # Generate predictions
    predictions = make_predictions(test_df, artifacts)
    
    # Create submission
    submission = create_submission(test_df, predictions, args.customer_id_col, args.output_path)
    
    logger.info("="*60)
    logger.info("Prediction completed successfully!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
