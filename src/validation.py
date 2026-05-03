"""
Time-Series Cross-Validation with Purging.
Prevents temporal data leakage in financial time series.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional
from sklearn.model_selection import BaseCrossValidator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator(BaseCrossValidator):
    """
    Time Series Cross-Validation with Purging.
    
    Implements Rolling Forecasting Origin with purging to prevent
    temporal data leakage between train and validation sets.
    
    The validation set must always strictly follow the training set
    chronologically, and we drop (purge) any overlapping observations.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 date_col: str = 'transaction_date',
                 gap_months: int = 3,
                 min_train_months: int = 6,
                 purge_overlap: bool = True):
        """
        Args:
            n_splits: Number of cross-validation folds
            date_col: Name of the date column
            gap_months: Months to purge between train and validation
            min_train_months: Minimum months of training data for first split
            purge_overlap: Whether to drop overlapping observations
        """
        self.n_splits = n_splits
        self.date_col = date_col
        self.gap_months = gap_months
        self.min_train_months = min_train_months
        self.purge_overlap = purge_overlap
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for time series cross-validation.
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Ensure date column exists and is datetime
        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")
        
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Sort by date
        df = df.sort_values(self.date_col).reset_index(drop=True)
        
        # Create date-based index for splitting
        date_range = pd.date_range(
            start=df[self.date_col].min(),
            end=df[self.date_col].max(),
            freq='MS'  # Month Start
        )
        
        if len(date_range) < self.min_train_months + self.n_splits:
            raise ValueError(
                f"Not enough months of data. Need at least "
                f"{self.min_train_months + self.n_splits} months, got {len(date_range)}"
            )
        
        # Calculate split points
        total_months = len(date_range)
        test_size = (total_months - self.min_train_months) // self.n_splits
        
        logger.info(f"Time Series CV: {total_months} months, "
                   f"min_train={self.min_train_months}, test_size={test_size}")
        
        for i in range(self.n_splits):
            # Calculate split dates
            train_end_idx = self.min_train_months + i * test_size
            test_start_idx = train_end_idx + self.gap_months
            test_end_idx = min(test_start_idx + test_size, total_months)
            
            if test_start_idx >= total_months:
                break
            
            train_end_date = date_range[train_end_idx - 1]
            test_start_date = date_range[test_start_idx]
            test_end_date = date_range[test_end_idx - 1]
            
            # Create masks for train and test
            train_mask = df[self.date_col] <= train_end_date
            
            # Apply purging: exclude gap months from training
            if self.purge_overlap:
                purge_start = train_end_date + pd.DateOffset(months=1)
                purge_end = test_start_date - pd.DateOffset(days=1)
                purge_mask = (df[self.date_col] >= purge_start) & (df[self.date_col] <= purge_end)
                train_mask = train_mask & ~purge_mask
            
            test_mask = (df[self.date_col] >= test_start_date) & (df[self.date_col] <= test_end_date)
            
            train_indices = df[train_mask].index.values
            test_indices = df[test_mask].index.values
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                logger.warning(f"Split {i+1}: Empty train or test set, skipping")
                continue
            
            logger.info(
                f"Split {i+1}/{self.n_splits}: "
                f"Train: {len(train_indices)} samples ({df[train_mask][self.date_col].min().date()} to "
                f"{df[train_mask][self.date_col].max().date()}), "
                f"Test: {len(test_indices)} samples ({df[test_mask][self.date_col].min().date()} to "
                f"{df[test_mask][self.date_col].max().date()})"
            )
            
            yield train_indices, test_indices


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged Group Time Series Cross-Validation.
    
    For datasets with multiple time series (e.g., multiple customers),
    this splitter respects both temporal ordering and group boundaries.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 date_col: str = 'transaction_date',
                 group_col: str = 'customer_id',
                 gap_months: int = 3):
        self.n_splits = n_splits
        self.date_col = date_col
        self.group_col = group_col
        self.gap_months = gap_months
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices respecting groups and time.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Get unique groups and their date ranges
        group_info = df.groupby(self.group_col)[self.date_col].agg(['min', 'max', 'count'])
        group_info = group_info.sort_values('min')
        
        n_groups = len(group_info)
        test_size = n_groups // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start_idx = (i + 1) * test_size
            test_end_idx = min(test_start_idx + test_size, n_groups)
            
            if test_start_idx >= n_groups:
                break
            
            # Get test groups
            test_groups = group_info.index[test_start_idx:test_end_idx]
            test_mask = df[self.group_col].isin(test_groups)
            
            # Get train groups (earlier groups)
            train_groups = group_info.index[:test_start_idx]
            train_mask = df[self.group_col].isin(train_groups)
            
            # Purge: remove recent observations from train that might leak
            if test_groups.any():
                test_min_date = df[test_mask][self.date_col].min()
                purge_date = test_min_date - pd.DateOffset(months=self.gap_months)
                train_mask = train_mask & (df[self.date_col] < purge_date)
            
            train_indices = df[train_mask].index.values
            test_indices = df[test_mask].index.values
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Logarithmic Error.
    
    RMSLE = sqrt(mean((log(y_pred + 1) - log(y_true + 1))^2))
    
    This metric is less sensitive to outliers than RMSE and
    doesn't penalize underestimation as heavily as overestimation.
    """
    # Clip negative predictions to zero
    y_pred = np.maximum(y_pred, 0)
    
    # Apply log1p transformation
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    # Compute RMSE of log values
    squared_errors = (log_pred - log_true) ** 2
    mse = np.mean(squared_errors)
    rmsle = np.sqrt(mse)
    
    return rmsle


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: TimeSeriesCrossValidator,
    target_transformer=None
) -> dict:
    """
    Perform time series cross-validation with optional target transformation.
    
    Returns:
        Dictionary with CV scores and out-of-fold predictions
    """
    from sklearn.base import clone
    
    oof_predictions = np.zeros(len(y))
    fold_scores = []
    
    logger.info(f"Starting {cv.get_n_splits()}-fold time series cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply target transformation
        if target_transformer:
            y_train_transformed = target_transformer.transform(y_train)
        else:
            y_train_transformed = y_train
        
        # Clone model and fit
        fold_model = clone(model)
        fold_model.fit(X_train, y_train_transformed)
        
        # Predict
        y_pred_transformed = fold_model.predict(X_val)
        
        # Inverse transform predictions
        if target_transformer:
            y_pred = target_transformer.inverse_transform(y_pred_transformed)
        else:
            y_pred = y_pred_transformed
        
        # Store OOF predictions
        oof_predictions[val_idx] = y_pred
        
        # Compute fold score
        fold_rmsle = compute_rmsle(y_val, y_pred)
        fold_scores.append(fold_rmsle)
        
        logger.info(f"Fold {fold + 1} RMSLE: {fold_rmsle:.4f}")
    
    # Overall CV score
    overall_rmsle = compute_rmsle(y, oof_predictions)
    mean_fold_rmsle = np.mean(fold_scores)
    std_fold_rmsle = np.std(fold_scores)
    
    logger.info(f"Overall CV RMSLE: {overall_rmsle:.4f}")
    logger.info(f"Mean Fold RMSLE: {mean_fold_rmsle:.4f} (+/- {std_fold_rmsle:.4f})")
    
    return {
        'oof_predictions': oof_predictions,
        'fold_scores': fold_scores,
        'overall_rmsle': overall_rmsle,
        'mean_rmsle': mean_fold_rmsle,
        'std_rmsle': std_fold_rmsle
    }
