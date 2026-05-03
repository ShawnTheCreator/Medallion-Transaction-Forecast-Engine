"""
High-cardinality categorical encoding using Target Encoding.
Prevents data leakage through strict temporal separation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding for high-cardinality categorical features.
    Uses bin counting (conditional probability of target given category).
    
    CRITICAL: To prevent data leakage, encoding values must be computed
    on training data only, then applied to validation/test data.
    """
    
    def __init__(self,
                 categorical_cols: List[str],
                 target_col: str,
                 min_samples: int = 30,
                 smoothing: float = 1.0,
                 noise: float = 0.01,
                 handle_unknown: str = 'global_mean'):
        """
        Args:
            categorical_cols: List of categorical column names to encode
            target_col: Name of target variable
            min_samples: Minimum samples for a category to use its own mean
            smoothing: Smoothing factor for regularization
            noise: Amount of Gaussian noise to add (prevents overfitting)
            handle_unknown: How to handle unseen categories ('global_mean', 'nan')
        """
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.min_samples = min_samples
        self.smoothing = smoothing
        self.noise = noise
        self.handle_unknown = handle_unknown
        
        # Storage for learned mappings
        self.encoding_maps_: Dict[str, Dict] = {}
        self.global_mean_: float = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TargetEncoder':
        """
        Fit the encoder on training data.
        
        IMPORTANT: X must contain both features AND the target column,
        OR y must be provided separately.
        """
        if y is not None:
            target_values = y.values if isinstance(y, pd.Series) else y
        else:
            if self.target_col not in X.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in data")
            target_values = X[self.target_col].values
        
        # Store global mean for regularization
        self.global_mean_ = np.mean(target_values)
        
        # Compute encodings for each categorical column
        for col in self.categorical_cols:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found in data, skipping")
                continue
            
            encoding_map = self._compute_encoding(
                X[col].values, 
                target_values
            )
            self.encoding_maps_[col] = encoding_map
            logger.info(f"Fitted target encoder for '{col}' with {len(encoding_map)} unique values")
        
        return self
    
    def _compute_encoding(self, categories: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Compute target encoding for a single categorical column.
        Uses smoothing to handle categories with few samples.
        """
        # Create DataFrame for easier aggregation
        temp_df = pd.DataFrame({
            'category': categories,
            'target': targets
        })
        
        # Compute category statistics
        category_stats = temp_df.groupby('category')['target'].agg(['mean', 'count'])
        
        # Apply smoothing: weighted average between category mean and global mean
        encoding_map = {}
        for cat, row in category_stats.iterrows():
            cat_mean = row['mean']
            cat_count = row['count']
            
            # Smoothing formula: weighted average
            # More weight to global mean when fewer samples
            weight = cat_count / (cat_count + self.smoothing)
            smoothed_mean = weight * cat_mean + (1 - weight) * self.global_mean_
            
            # Add noise for regularization (only during fit, not transform)
            if self.noise > 0:
                smoothed_mean += np.random.normal(0, self.noise * self.global_mean_)
            
            encoding_map[cat] = smoothed_mean
        
        return encoding_map
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using learned encodings.
        Returns a copy with encoded values and original columns dropped.
        """
        df = X.copy()
        
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            
            if col not in self.encoding_maps_:
                logger.warning(f"No encoding map for '{col}', skipping")
                continue
            
            # Apply encoding
            new_col_name = f'{col}_target_enc'
            
            if self.handle_unknown == 'global_mean':
                df[new_col_name] = df[col].map(self.encoding_maps_[col]).fillna(self.global_mean_)
            else:  # 'nan'
                df[new_col_name] = df[col].map(self.encoding_maps_[col])
            
            # Drop original categorical column
            df = df.drop(columns=[col])
            
            logger.info(f"Transformed column '{col}' -> '{new_col_name}'")
        
        return df
    
    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        return [f'{col}_target_enc' for col in self.categorical_cols if col in self.encoding_maps_]


class FeatureHasherEncoder(BaseEstimator, TransformerMixin):
    """
    Feature hashing for extremely high-cardinality text features.
    Memory-efficient alternative to one-hot encoding.
    """
    
    def __init__(self,
                 text_cols: List[str],
                 n_features: int = 1000,
                 alternate_sign: bool = True):
        """
        Args:
            text_cols: List of text column names to hash
            n_features: Number of hash buckets
            alternate_sign: Whether to use alternating signs for collision handling
        """
        self.text_cols = text_cols
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        
    def _hash_feature(self, value: str, seed: int = 0) -> Tuple[List[int], List[float]]:
        """
        Hash a single string value to feature indices and values.
        Uses signed hashing to reduce collision impact.
        """
        if pd.isna(value):
            return [], []
        
        # Simple hash function
        hash_val = hash(str(value) + str(seed)) % self.n_features
        
        if self.alternate_sign:
            # Determine sign based on another hash
            sign = 1 if hash(str(value) + "sign") % 2 == 0 else -1
        else:
            sign = 1
        
        return [hash_val], [sign]
    
    def fit(self, X, y=None):
        """No fitting required for feature hashing."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform text columns to hashed features.
        Returns sparse matrix as DataFrame.
        """
        from scipy import sparse
        
        df = X.copy()
        all_hashed_features = []
        
        for col in self.text_cols:
            if col not in df.columns:
                continue
            
            # Create sparse matrix for this column
            rows = []
            cols = []
            data = []
            
            for idx, value in enumerate(df[col].fillna('')):
                indices, values = self._hash_feature(value, seed=hash(col) % 1000)
                for i, v in zip(indices, values):
                    rows.append(idx)
                    cols.append(i)
                    data.append(v)
            
            # Create sparse matrix
            hashed = sparse.csr_matrix(
                (data, (rows, cols)), 
                shape=(len(df), self.n_features)
            )
            
            # Convert to dense for DataFrame
            hashed_df = pd.DataFrame(
                hashed.toarray(),
                columns=[f'{col}_hash_{i}' for i in range(self.n_features)],
                index=df.index
            )
            
            all_hashed_features.append(hashed_df)
            
            # Drop original column
            df = df.drop(columns=[col])
        
        # Concatenate all hashed features
        if all_hashed_features:
            df = pd.concat([df] + all_hashed_features, axis=1)
        
        return df


class TfidfEncoder(BaseEstimator, TransformerMixin):
    """
    TF-IDF encoding for text descriptions.
    Captures semantic meaning and spending intent better than simple hashing.
    """
    
    def __init__(self,
                 text_cols: List[str],
                 max_features: int = 500,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5,
                 max_df: float = 0.95):
        """
        Args:
            text_cols: List of text column names to encode
            max_features: Maximum number of features (top TF-IDF terms)
            ngram_range: N-gram range for tokenization (default: unigrams + bigrams)
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms (ignore overly common terms)
        """
        self.text_cols = text_cols
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizers_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit TF-IDF vectorizers on training data."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        for col in self.text_cols:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found in data, skipping TF-IDF")
                continue
            
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                dtype=np.float32
            )
            
            # Fit on non-null values
            texts = X[col].fillna('').astype(str)
            vectorizer.fit(texts)
            
            self.vectorizers_[col] = vectorizer
            logger.info(f"Fitted TF-IDF for '{col}' with {len(vectorizer.vocabulary_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text columns to TF-IDF features."""
        from scipy import sparse
        
        df = X.copy()
        all_tfidf_features = []
        
        for col, vectorizer in self.vectorizers_.items():
            if col not in df.columns:
                continue
            
            texts = df[col].fillna('').astype(str)
            
            # Transform to TF-IDF
            tfidf_matrix = vectorizer.transform(texts)
            
            # Convert to DataFrame
            feature_names = [f'{col}_tfidf_{feat}' for feat in vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame.sparse.from_spmatrix(
                tfidf_matrix,
                columns=feature_names,
                index=df.index
            )
            
            all_tfidf_features.append(tfidf_df)
            
            # Drop original column
            df = df.drop(columns=[col])
            
            logger.info(f"Transformed '{col}' to {len(feature_names)} TF-IDF features")
        
        # Concatenate all TF-IDF features
        if all_tfidf_features:
            df = pd.concat([df] + all_tfidf_features, axis=1)
        
        return df


class CategoricalEncoderPipeline:
    """
    Pipeline for encoding all categorical features.
    Combines target encoding and text encoding (hashing or TF-IDF) as appropriate.
    """
    
    def __init__(self,
                 high_cardinality_cols: List[str],
                 text_cols: List[str],
                 target_col: str,
                 text_encoding_method: str = 'tfidf',  # 'hashing' or 'tfidf'
                 hash_vector_size: int = 1000,
                 tfidf_max_features: int = 500,
                 target_encoding_min_samples: int = 30):
        """
        Args:
            high_cardinality_cols: Columns for target encoding
            text_cols: Text columns to encode
            target_col: Target variable name
            text_encoding_method: 'hashing' or 'tfidf' (default: tfidf for better semantic capture)
            hash_vector_size: Feature hashing dimensions (if method='hashing')
            tfidf_max_features: Max TF-IDF features (if method='tfidf')
            target_encoding_min_samples: Minimum samples for target encoding
        """
        self.high_cardinality_cols = high_cardinality_cols
        self.text_cols = text_cols
        self.target_col = target_col
        self.text_encoding_method = text_encoding_method
        self.hash_vector_size = hash_vector_size
        self.tfidf_max_features = tfidf_max_features
        self.target_encoding_min_samples = target_encoding_min_samples
        
        self.target_encoder = None
        self.text_encoder = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit all encoders on training data."""
        # Fit target encoder for high-cardinality categoricals
        if self.high_cardinality_cols:
            self.target_encoder = TargetEncoder(
                categorical_cols=self.high_cardinality_cols,
                target_col=self.target_col,
                min_samples=self.target_encoding_min_samples
            )
            self.target_encoder.fit(X, y)
        
        # Text encoding: choose between TF-IDF (default) or hashing
        if self.text_cols:
            if self.text_encoding_method == 'tfidf':
                logger.info(f"Using TF-IDF encoding for text columns: {self.text_cols}")
                self.text_encoder = TfidfEncoder(
                    text_cols=self.text_cols,
                    max_features=self.tfidf_max_features,
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95
                )
            else:
                logger.info(f"Using Feature Hashing for text columns: {self.text_cols}")
                self.text_encoder = FeatureHasherEncoder(
                    text_cols=self.text_cols,
                    n_features=self.hash_vector_size
                )
            self.text_encoder.fit(X, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform all categorical columns."""
        df = X.copy()
        
        if self.target_encoder:
            df = self.target_encoder.transform(df)
        
        if self.text_encoder:
            df = self.text_encoder.transform(df)
        
        return df


def create_categorical_encoders(config: Dict, text_encoding_method: str = 'tfidf') -> CategoricalEncoderPipeline:
    """
    Factory function to create categorical encoder pipeline from config.
    
    Args:
        config: Configuration dictionary
        text_encoding_method: 'tfidf' (recommended) or 'hashing'
    """
    data_cfg = config['data']
    feature_cfg = config['features']
    
    # Separate high-cardinality categoricals from text columns
    # For this challenge, descriptions go to TF-IDF (captures semantic meaning)
    # Other high-cardinality categoricals go to target encoding
    text_cols = [col for col in data_cfg.high_cardinality_cols 
                 if 'description' in col.lower() or 'desc' in col.lower()]
    target_encoding_cols = [col for col in data_cfg.high_cardinality_cols 
                            if col not in text_cols]
    
    logger.info(f"Target encoding columns: {target_encoding_cols}")
    logger.info(f"Text encoding columns: {text_cols} (method: {text_encoding_method})")
    
    return CategoricalEncoderPipeline(
        high_cardinality_cols=target_encoding_cols,
        text_cols=text_cols,
        target_col=data_cfg.target_col,
        text_encoding_method=text_encoding_method,
        hash_vector_size=feature_cfg.hash_vector_size,
        tfidf_max_features=500,  # Top 500 TF-IDF features
        target_encoding_min_samples=feature_cfg.target_encoding_min_samples
    )
