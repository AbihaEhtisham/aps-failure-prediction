"""
Data preprocessing module for APS Failure Prediction

This module handles missing value imputation, feature transformation,
and data cleaning for the Scania APS failure prediction project.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingValueHandler:
    """
    Advanced missing value handler with multiple strategies.
    
    Handles missing values based on their percentage:
    - High (>80%): Creates binary flag and drops original
    - Medium (20-80%): MICE imputation
    - Low (<20%): Median imputation
    - None: No action needed
    
    Attributes:
        high_threshold (float): Threshold for high missing percentage
        feature_groups (dict): Dictionary of feature groups by missing percentage
        imputers (dict): Dictionary of fitted imputers for each group
        binary_flags (dict): Dictionary of binary flag columns
    """
    
    def __init__(self, high_threshold: float = 0.8):
        """
        Initialize MissingValueHandler.
        
        Args:
            high_threshold: Threshold for high missing percentage (default: 0.8)
        """
        self.high_threshold = high_threshold
        self.feature_groups: Dict[str, List[str]] = {}
        self.imputers: Dict[str, object] = {}
        self.binary_flags: Dict[str, str] = {}
        
    def fit(self, X: pd.DataFrame) -> 'MissingValueHandler':
        """
        Fit missing value handler to data.
        
        Args:
            X: Input dataframe
            
        Returns:
            self: Fitted handler
        """
        logger.info("Fitting MissingValueHandler...")
        
        missing_pct = X.isna().sum() / len(X) * 100
        
        # Categorize features by missing percentage
        self.feature_groups['high'] = missing_pct[missing_pct >= self.high_threshold * 100].index.tolist()
        self.feature_groups['medium'] = missing_pct[
            (missing_pct >= 20) & (missing_pct < self.high_threshold * 100)
        ].index.tolist()
        self.feature_groups['low'] = missing_pct[
            (missing_pct > 0) & (missing_pct < 20)
        ].index.tolist()
        self.feature_groups['none'] = missing_pct[missing_pct == 0].index.tolist()
        
        # Create binary flags for high missing features
        for feature in self.feature_groups['high']:
            self.binary_flags[feature] = f"{feature}_missing_flag"
        
        # Fit medium missing imputer (MICE)
        if self.feature_groups['medium']:
            logger.info(f"Fitting MICE imputer for {len(self.feature_groups['medium'])} features")
            self.imputers['medium'] = IterativeImputer(
                estimator=None,
                max_iter=10,
                random_state=42,
                n_nearest_features=5
            )
            self.imputers['medium'].fit(X[self.feature_groups['medium']])
        
        # Fit low missing imputer (median)
        if self.feature_groups['low']:
            logger.info(f"Fitting median imputer for {len(self.feature_groups['low'])} features")
            self.imputers['low'] = SimpleImputer(strategy='median')
            self.imputers['low'].fit(X[self.feature_groups['low']])
        
        logger.info("MissingValueHandler fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with missing value handling.
        
        Args:
            X: Input dataframe
            
        Returns:
            X_transformed: Dataframe with handled missing values
        """
        logger.info("Transforming data with MissingValueHandler...")
        X_transformed = X.copy()
        
        # 1. Handle high missing: Create binary flags
        for feature in self.feature_groups['high']:
            flag_name = self.binary_flags[feature]
            X_transformed[flag_name] = X[feature].isna().astype(int)
            # Drop the original high-missing column
            X_transformed = X_transformed.drop(columns=[feature])
        
        # 2. Handle medium missing: MICE imputation
        if self.feature_groups['medium']:
            medium_imputed = self.imputers['medium'].transform(
                X_transformed[self.feature_groups['medium']]
            )
            X_transformed[self.feature_groups['medium']] = medium_imputed
        
        # 3. Handle low missing: Median imputation
        if self.feature_groups['low']:
            low_imputed = self.imputers['low'].transform(
                X_transformed[self.feature_groups['low']]
            )
            X_transformed[self.feature_groups['low']] = low_imputed
        
        logger.info(f"Transformation complete. Shape: {X_transformed.shape}")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class FeatureTransformer:
    """
    Feature transformation with skewness handling and scaling.
    
    Applies:
    - Yeo-Johnson transformation to skewed features
    - Robust scaling to all features (outlier-resistant)
    
    Attributes:
        scaler (RobustScaler): Fitted robust scaler
        transformer (PowerTransformer): Fitted power transformer
        features_to_transform (List[str]): Features with high skewness
    """
    
    def __init__(self):
        """Initialize FeatureTransformer."""
        self.scaler = RobustScaler()
        self.transformer = None
        self.features_to_transform: List[str] = []
        
    def fit(self, X: pd.DataFrame) -> 'FeatureTransformer':
        """
        Fit transformers to data.
        
        Args:
            X: Input dataframe
            
        Returns:
            self: Fitted transformer
        """
        logger.info("Fitting FeatureTransformer...")
        
        # Identify skewed features (skewness > 1 or < -1)
        skewness = X.skew()
        self.features_to_transform = skewness[
            (skewness > 1) | (skewness < -1)
        ].index.tolist()
        
        logger.info(f"Found {len(self.features_to_transform)} skewed features")
        
        # Apply PowerTransformer to skewed features
        if self.features_to_transform:
            self.transformer = PowerTransformer(method='yeo-johnson')
            self.transformer.fit(X[self.features_to_transform])
        
        # Fit scaler
        self.scaler.fit(X)
        
        logger.info("FeatureTransformer fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.
        
        Args:
            X: Input dataframe
            
        Returns:
            X_transformed: Transformed dataframe
        """
        logger.info("Transforming data with FeatureTransformer...")
        X_transformed = X.copy()
        
        # Transform skewed features
        if self.transformer and self.features_to_transform:
            X_transformed[self.features_to_transform] = self.transformer.transform(
                X_transformed[self.features_to_transform]
            )
        
        # Scale all features
        X_scaled = self.scaler.transform(X_transformed)
        X_transformed = pd.DataFrame(
            X_scaled,
            columns=X_transformed.columns,
            index=X_transformed.index
        )
        
        logger.info("Transformation complete")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for production.
    
    Combines all preprocessing steps:
    1. Missing value handling
    2. Feature engineering (histogram + counter)
    3. Feature transformation
    4. Feature selection
    
    Attributes:
        missing_handler: MissingValueHandler instance
        hist_engineer: HistogramFeatureEngineer instance
        counter_engineer: CounterFeatureEngineer instance
        transformer: FeatureTransformer instance
        selector: FeatureSelector instance
    """
    
    def __init__(self):
        """Initialize PreprocessingPipeline."""
        self.missing_handler = None
        self.hist_engineer = None
        self.counter_engineer = None
        self.transformer = None
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        """
        Fit all preprocessing components.
        
        Args:
            X: Input dataframe
            y: Target series (optional, for feature selection)
            
        Returns:
            self: Fitted pipeline
        """
        logger.info("Fitting PreprocessingPipeline...")
        
        # 1. Missing value handling
        self.missing_handler = MissingValueHandler(high_threshold=0.8)
        X_imputed = self.missing_handler.fit_transform(X)
        
        # 2. Histogram feature engineering
        from .feature_engineering import HistogramFeatureEngineer
        self.hist_engineer = HistogramFeatureEngineer(prefix='ay_')
        X_hist = self.hist_engineer.fit_transform(X_imputed)
        
        # 3. Counter feature engineering
        from .feature_engineering import CounterFeatureEngineer
        self.counter_engineer = CounterFeatureEngineer()
        X_counter = self.counter_engineer.fit_transform(X_hist)
        
        # 4. Feature transformation
        self.transformer = FeatureTransformer()
        X_transformed = self.transformer.fit_transform(X_counter)
        
        # 5. Feature selection
        from .feature_engineering import FeatureSelector
        self.selector = FeatureSelector(variance_threshold=0.01, k_features=100)
        if y is not None:
            self.selector.fit(X_transformed, y)
        else:
            self.selector.selected_features = X_transformed.columns.tolist()
        
        logger.info("PreprocessingPipeline fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input dataframe
            
        Returns:
            X_transformed: Transformed dataframe
        """
        logger.info("Transforming data with PreprocessingPipeline...")
        
        X_imputed = self.missing_handler.transform(X)
        X_hist = self.hist_engineer.transform(X_imputed)
        X_counter = self.counter_engineer.transform(X_hist)
        X_transformed = self.transformer.transform(X_counter)
        X_selected = self.selector.transform(X_transformed)
        
        logger.info(f"Transformation complete. Shape: {X_selected.shape}")
        return X_selected
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)