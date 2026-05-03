# Nedbank Transaction Forecasting ML Pipeline

A comprehensive machine learning pipeline designed for the Nedbank Data Masters challenge to predict `next_3m_txn_count` (3-month transaction counts).

## Key Features

### 1. Target Optimization: RMSLE with Log Transform
- **Problem**: Transaction counts are right-skewed with outliers
- **Solution**: `np.log1p` transformation converts RMSE → RMSLE optimization
- **Benefit**: Stabilizes variance, prevents overfitting to high-frequency outliers

### 2. Temporal Feature Engineering
Captures seasonality and velocity from 34 months of history:
- **Seasonal lags**: 12, 11, 13 months (annual patterns, holiday seasonality)
- **Short-term lags**: 1, 2, 3, 6 months
- **Rolling statistics**: Mean, std, min, max over 1, 3, 6, 12 months
- **Velocity ratios**: Recent vs historical (e.g., 3m vs 12m ago)
- **Trend features**: YoY growth, recent momentum

### 3. High-Cardinality Categorical Handling
- **Target Encoding (Bin Counting)**: For categorical features with many levels
  - Computes conditional probability of target given category
  - Smoothing and noise for regularization
  - Prevents leakage via strict train/val separation
- **TF-IDF (Default)**: For free-text transaction descriptions
  - Captures semantic meaning and spending intent (e.g., "Woolworths" vs "Zando")
  - Top 500 features with unigrams + bigrams
- **Feature Hashing**: Alternative for text descriptions
  - Memory-efficient, useful for very high-cardinality text
  - Fixed-length vector output (default: 1000 dimensions)

### 4. Time-Series Cross-Validation
- **Rolling Forecasting Origin**: Validation always follows training chronologically
- **Purging**: 3-month gap between train and validation to prevent leakage
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error)

### 5. Gradient Boosting Models
Supports three state-of-the-art implementations:
- **LightGBM**: Fast training, excellent performance (default)
- **XGBoost**: Robust, battle-tested
- **CatBoost**: Native categorical handling

## Project Structure

```
Medallion-Transaction-Forecast-Engine/
├── src/
│   ├── config.py                  # Configuration management
│   ├── data_preprocessing.py       # Temporal features, target transform
│   ├── categorical_encoder.py     # Target encoding, feature hashing
│   ├── validation.py                # Time-series CV with purging
│   └── model.py                     # Gradient boosting wrapper
├── train.py                         # Training script
├── predict.py                       # Prediction/submission script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## CRITICAL: Submission Format

**⚠️ WARNING**: The scoring platform expects predictions in `np.log1p` format (natural log of predictions plus one), NOT raw transaction counts.

The `predict.py` script automatically handles this - it outputs log-transformed predictions directly. Do NOT manually apply `expm1` to the submission file.

**Expected submission format**:
```python
# predict.py does this automatically:
submission['next_3m_txn_count'] = np.log1p(raw_predictions)  # ✓ CORRECT
# NOT: submission['next_3m_txn_count'] = raw_predictions     # ✗ WRONG
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with LightGBM (default)
python train.py \
    --train-path data/train.csv \
    --test-path data/test.csv \
    --model-type lightgbm \
    --n-splits 5 \
    --output-dir outputs

# Train with XGBoost
python train.py \
    --train-path data/train.csv \
    --model-type xgboost \
    --n-splits 5

# Train with CatBoost
python train.py \
    --train-path data/train.csv \
    --model-type catboost \
    --n-splits 5
```

### Prediction

```bash
python predict.py \
    --test-path data/test.csv \
    --model-dir models \
    --output-path submission.csv
```

## Configuration

Edit `src/config.py` to customize:

```python
# Data paths
DataConfig.train_path = "data/train.csv"
DataConfig.test_path = "data/test.csv"

# Temporal features
FeatureConfig.seasonal_lags = [12, 11, 13]  # Annual patterns
FeatureConfig.rolling_windows = [1, 3, 6, 12]  # Rolling statistics

# Model hyperparameters
ModelConfig.lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    # ...
}
```

## Feature Engineering Details

### Temporal Features

| Feature Type | Description | Example |
|-------------|-------------|---------|
| Lag features | Historical values at different time points | `txn_count_lag_12m` |
| Rolling stats | Moving window statistics | `txn_count_rolling_mean_3m` |
| Velocity | Recent performance vs historical | `velocity_3m_vs_12m` |
| Seasonality | Calendar-based indicators | `is_nov_jan`, `is_holiday_season` |
| Trend | Direction of change | `trend_3m`, `yoy_growth` |

### Target Encoding

For high-cardinality categorical features:

```python
# Formula: smoothed mean with regularization
encoding = (count * category_mean + smoothing * global_mean) / (count + smoothing)

# Add noise for overfitting prevention
encoding += noise * global_mean * N(0, 1)
```

## Validation Strategy

```
Fold 1: [0-6m train] [gap 3m] [9-12m val]
Fold 2: [0-9m train] [gap 3m] [12-15m val]
Fold 3: [0-12m train] [gap 3m] [15-18m val]
Fold 4: [0-15m train] [gap 3m] [18-21m val]
Fold 5: [0-18m train] [gap 3m] [21-24m val]
```

Key points:
- Training window expands over folds
- Validation always follows training chronologically
- 3-month purge gap prevents leakage
- Final model trains on full dataset

## Performance Optimization

### For RMSLE Metric:
1. **Target transformation**: `log1p(y)` before training
2. **Prediction inversion**: `expm1(pred)` after prediction
3. **Clip negatives**: Ensure non-negative final predictions

### For Seasonality Capture:
- Lag 12 captures year-over-year patterns
- November-January seasonality flags for holiday spending
- Velocity ratios detect recent acceleration/deceleration

### For Robustness:
- Tree-based models handle outliers naturally
- Rolling statistics smooth noise
- Regularization prevents overfitting

## Output Files

After training, the following are saved:

```
models/
├── model.pkl                      # Trained model
├── preprocessors.pkl              # Feature engineering pipeline
└── target_transformer.pkl          # Log transform parameters

outputs/
├── feature_importance.csv          # Feature importance rankings
├── oof_predictions.csv            # Out-of-fold predictions
└── training.log                    # Training log

submission.csv                      # Final predictions
```

## Advanced Usage

### Custom Feature Engineering

```python
from src.data_preprocessing import TemporalFeatureEngineer

engineer = TemporalFeatureEngineer(
    seasonal_lags=[12, 24],  # 1 and 2 year lags
    rolling_windows=[3, 6, 12],
    velocity_windows=[(3, 12), (6, 12)]
)
```

### Ensemble Models

```python
from src.model import ModelEnsemble

# Train multiple models
model1 = train_model(X, y, config_lgb)
model2 = train_model(X, y, config_xgb)

# Create ensemble
ensemble = ModelEnsemble([model1, model2], weights=[0.6, 0.4])
predictions = ensemble.predict(X_test)
```

## Troubleshooting

### Memory Issues
- Reduce `hash_vector_size` for feature hashing
- Use fewer `rolling_windows` or `seasonal_lags`
- Sample training data for initial experiments

### Overfitting
- Increase `min_samples` in target encoding
- Add regularization (`reg_alpha`, `reg_lambda`)
- Reduce `n_estimators` and use early stopping

### Slow Training
- Use LightGBM instead of XGBoost
- Reduce `n_splits` in cross-validation
- Use GPU if available (CatBoost/XGBoost)

## References

Based on principles from:
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Feature Engineering for Machine Learning" by Alice Zheng & Amanda Casari
- Kaggle competition best practices for time series forecasting

## License

MIT License
