"""
Gradient Boosting Machine models for transaction forecasting.
Supports XGBoost, LightGBM, and CatBoost with unified interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseEstimator, RegressorMixin):
    """
    Unified interface for Gradient Boosting models.
    Supports XGBoost, LightGBM, and CatBoost.
    """
    
    def __init__(self,
                 model_type: str = 'lightgbm',
                 params: Optional[Dict] = None,
                 categorical_features: Optional[List[str]] = None,
                 early_stopping_rounds: int = 100,
                 verbose: bool = True):
        """
        Args:
            model_type: 'xgboost', 'lightgbm', or 'catboost'
            params: Model hyperparameters
            categorical_features: List of categorical feature names (for CatBoost)
            early_stopping_rounds: Rounds for early stopping
            verbose: Whether to print training progress
        """
        self.model_type = model_type.lower()
        self.params = params or {}
        self.categorical_features = categorical_features or []
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.model_ = None
        self.feature_importances_ = None
        
    def _get_model_class(self):
        """Import and return the appropriate model class."""
        if self.model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                return XGBRegressor
            except ImportError:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        elif self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor
            except ImportError:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        elif self.model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor
            except ImportError:
                raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def fit(self, 
            X: pd.DataFrame, 
            y: np.ndarray,
            eval_set: Optional[List[Tuple]] = None,
            sample_weight: Optional[np.ndarray] = None):
        """
        Fit the gradient boosting model.
        
        Args:
            X: Feature matrix
            y: Target values (should be log-transformed for RMSLE)
            eval_set: List of (X, y) tuples for early stopping
            sample_weight: Optional sample weights
        """
        ModelClass = self._get_model_class()
        
        # Prepare parameters
        fit_params = {}
        
        if self.model_type == 'xgboost':
            self.model_ = ModelClass(**self.params)
            
            if eval_set:
                fit_params['eval_set'] = eval_set
                fit_params['early_stopping_rounds'] = self.early_stopping_rounds
                fit_params['verbose'] = self.verbose
            
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
        
        elif self.model_type == 'lightgbm':
            # Extract early stopping from params
            early_stopping = self.params.pop('early_stopping_rounds', self.early_stopping_rounds)
            n_estimators = self.params.get('n_estimators', 1000)
            
            self.model_ = ModelClass(**self.params)
            
            if eval_set:
                fit_params['eval_set'] = eval_set
                fit_params['callbacks'] = [
                    # Early stopping callback
                ]
                
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            
            # LightGBM specific: handle categorical features
            # They should be encoded as integers before passing
            
        elif self.model_type == 'catboost':
            # CatBoost handles categorical features natively
            cat_features_indices = [
                X.columns.get_loc(col) for col in self.categorical_features 
                if col in X.columns
            ]
            
            catboost_params = self.params.copy()
            catboost_params['cat_features'] = cat_features_indices
            catboost_params['use_best_model'] = True
            
            self.model_ = ModelClass(**catboost_params)
            
            if eval_set:
                fit_params['eval_set'] = eval_set
        
        # Fit the model
        logger.info(f"Training {self.model_type} model on {X.shape[0]} samples, {X.shape[1]} features...")
        
        try:
            self.model_.fit(X, y, **fit_params)
        except TypeError:
            # Fallback: some versions have different parameter names
            if 'callbacks' in fit_params:
                del fit_params['callbacks']
            self.model_.fit(X, y, **fit_params)
        
        # Store feature importances
        if hasattr(self.model_, 'feature_importances_'):
            self.feature_importances_ = self.model_.feature_importances_
        
        logger.info(f"Model training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        
        predictions = self.model_.predict(X)
        return predictions
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importances as a DataFrame.
        
        Returns:
            DataFrame with 'feature' and 'importance' columns
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has no feature importances available")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(self.feature_importances_)],
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class ModelEnsemble:
    """
    Ensemble of multiple gradient boosting models.
    Supports weighted averaging of predictions.
    """
    
    def __init__(self, models: List[GradientBoostingModel], weights: Optional[List[float]] = None):
        """
        Args:
            models: List of fitted GradientBoostingModel instances
            weights: Optional weights for each model (defaults to equal weighting)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions as weighted average.
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        # Weighted sum
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred


def get_model_from_config(config: Dict) -> GradientBoostingModel:
    """
    Factory function to create a model from configuration.
    """
    model_cfg = config['model']
    
    # Select parameters based on model type
    if model_cfg.model_type == 'xgboost':
        params = model_cfg.xgb_params.copy()
    elif model_cfg.model_type == 'lightgbm':
        params = model_cfg.lgb_params.copy()
    elif model_cfg.model_type == 'catboost':
        params = model_cfg.catboost_params.copy()
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")
    
    return GradientBoostingModel(
        model_type=model_cfg.model_type,
        params=params,
        early_stopping_rounds=params.get('early_stopping_rounds', 100)
    )


def train_with_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    config: Dict,
    cv_splitter: Any,
    target_transformer: Optional[Any] = None
) -> Tuple[GradientBoostingModel, Dict]:
    """
    Train model with time series cross-validation.
    
    Returns:
        Tuple of (fitted_model, cv_results)
    """
    from sklearn.base import clone
    from .validation import compute_rmsle
    
    model_cfg = config['model']
    
    # Initialize model
    base_model = get_model_from_config(config)
    
    # Store OOF predictions
    oof_preds = np.zeros(len(y))
    fold_scores = []
    fold_models = []
    
    logger.info(f"Starting {model_cfg.model_type} training with {cv_splitter.get_n_splits()}-fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X)):
        logger.info(f"\n=== Fold {fold + 1}/{cv_splitter.get_n_splits()} ===")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply target transformation
        if target_transformer:
            y_train_t = target_transformer.transform(y_train)
            y_val_t = target_transformer.transform(y_val)
        else:
            y_train_t, y_val_t = y_train, y_val
        
        # Clone model for this fold
        fold_model = clone(base_model)
        
        # Fit with early stopping
        fold_model.fit(
            X_train, y_train_t,
            eval_set=[(X_val, y_val_t)]
        )
        
        # Predict
        y_pred_t = fold_model.predict(X_val)
        
        # Inverse transform
        if target_transformer:
            y_pred = target_transformer.inverse_transform(y_pred_t)
        else:
            y_pred = y_pred_t
        
        # Store OOF
        oof_preds[val_idx] = y_pred
        
        # Score
        fold_rmsle = compute_rmsle(y_val, y_pred)
        fold_scores.append(fold_rmsle)
        fold_models.append(fold_model)
        
        logger.info(f"Fold {fold + 1} RMSLE: {fold_rmsle:.4f}")
    
    # Overall CV score
    overall_rmsle = compute_rmsle(y, oof_preds)
    logger.info(f"\n{'='*50}")
    logger.info(f"Overall CV RMSLE: {overall_rmsle:.4f}")
    logger.info(f"Mean Fold RMSLE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    logger.info(f"{'='*50}\n")
    
    # Retrain on full data with best iteration from CV
    logger.info("Retraining on full dataset...")
    final_model = clone(base_model)
    
    if target_transformer:
        y_t = target_transformer.transform(y)
    else:
        y_t = y
    
    final_model.fit(X, y_t)
    
    cv_results = {
        'oof_predictions': oof_preds,
        'fold_scores': fold_scores,
        'overall_rmsle': overall_rmsle,
        'mean_rmsle': np.mean(fold_scores),
        'std_rmsle': np.std(fold_scores),
        'fold_models': fold_models
    }
    
    return final_model, cv_results
