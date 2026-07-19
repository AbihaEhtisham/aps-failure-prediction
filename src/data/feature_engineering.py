"""
Feature engineering module for APS Failure Prediction.

This module provides classes for engineering features from:
- Histogram data (ay_000-ay_009)
- Counter data (ag_, az_, ba_, etc.)
- Feature selection
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HistogramFeatureEngineer:
    """
    Extract statistical features from histogram data.
    
    Histogram features (ay_000-ay_009) represent distributions.
    This class extracts:
    - Mean, variance, standard deviation
    - Skewness and kurtosis
    - Peak location (mode)
    - Entropy (distribution complexity)
    - IQR (spread)
    
    Attributes:
        prefix (str): Prefix for histogram features
        hist_features (List[str]): List of histogram feature names
    """
    
    def __init__(self, prefix: str = 'ay_'):
        """
        Initialize HistogramFeatureEngineer.
        
        Args:
            prefix: Prefix of histogram features (default: 'ay_')
        """
        self.prefix = prefix
        self.hist_features: List[str] = []
        
    def fit(self, X: pd.DataFrame) -> 'HistogramFeatureEngineer':
        """
        Identify histogram features.
        
        Args:
            X: Input dataframe
            
        Returns:
            self: Fitted engineer
        """
        self.hist_features = [col for col in X.columns if col.startswith(self.prefix)]
        logger.info(f"Found {len(self.hist_features)} histogram features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical features from histograms.
        
        Args:
            X: Input dataframe
            
        Returns:
            X_transformed: Dataframe with histogram-derived features
        """
        logger.info("Engineering histogram features...")
        X_transformed = X.copy()
        
        if not self.hist_features:
            logger.warning("No histogram features found")
            return X_transformed
        
        # Convert hist features to numpy array for efficient computation
        hist_data = X[self.hist_features].values
        
        # Create statistical features
        X_transformed[f'{self.prefix}mean'] = np.nanmean(hist_data, axis=1)
        X_transformed[f'{self.prefix}variance'] = np.nanvar(hist_data, axis=1)
        X_transformed[f'{self.prefix}std'] = np.nanstd(hist_data, axis=1)
        X_transformed[f'{self.prefix}skew'] = pd.DataFrame(hist_data).skew(axis=1)
        X_transformed[f'{self.prefix}kurtosis'] = pd.DataFrame(hist_data).kurtosis(axis=1)
        
        # Find peak location (mode)
        with np.errstate(invalid='ignore'):
            peak_indices = np.nanargmax(hist_data, axis=1)
            peak_indices = np.where(np.isnan(peak_indices), -1, peak_indices)
        X_transformed[f'{self.prefix}peak_index'] = peak_indices
        
        # Calculate concentration (entropy-based)
        hist_probs = hist_data / (np.nansum(hist_data, axis=1, keepdims=True) + 1e-10)
        entropy = -np.nansum(hist_probs * np.log(hist_probs + 1e-10), axis=1)
        X_transformed[f'{self.prefix}entropy'] = entropy
        
        # Calculate spread (interquartile range of distribution)
        hist_quantiles = np.percentile(hist_data, [25, 75], axis=1)
        X_transformed[f'{self.prefix}iqr'] = hist_quantiles[1] - hist_quantiles[0]
        
        # Calculate peak-to-mean ratio
        peak_values = np.nanmax(hist_data, axis=1)
        mean_values = np.nanmean(hist_data, axis=1)
        X_transformed[f'{self.prefix}peak_ratio'] = peak_values / (mean_values + 1e-10)
        
        # Drop original histogram features to reduce dimensionality
        X_transformed = X_transformed.drop(columns=self.hist_features)
        
        logger.info(f"Added {X_transformed.shape[1]} histogram-derived features")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class CounterFeatureEngineer:
    """
    Engineer features from counter data.
    
    Counter features represent event counts over time.
    This class creates:
    - Cumulative sums
    - Rate of change (differences)
    - Max/min ratios
    - Growth rates
    
    Attributes:
        counter_groups (List[str]): Prefixes for counter feature groups
        groups (Dict[str, List[str]]): Dictionary of feature groups
    """
    
    def __init__(self, counter_groups: List[str] = None):
        """
        Initialize CounterFeatureEngineer.
        
        Args:
            counter_groups: List of prefixes for counter features
        """
        if counter_groups is None:
            self.counter_groups = ['ag_', 'az_', 'ba_', 'cs_', 'ee_', 'cn_']
        else:
            self.counter_groups = counter_groups
        self.groups: Dict[str, List[str]] = {}
        
    def fit(self, X: pd.DataFrame) -> 'CounterFeatureEngineer':
        """
        Identify counter feature groups.
        
        Args:
            X: Input dataframe
            
        Returns:
            self: Fitted engineer
        """
        for prefix in self.counter_groups:
            self.groups[prefix] = [col for col in X.columns if col.startswith(prefix)]
        
        total_counters = sum(len(features) for features in self.groups.values())
        logger.info(f"Found {total_counters} counter features across {len(self.groups)} groups")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create rate-of-change and cumulative features from counters.
        
        Args:
            X: Input dataframe
            
        Returns:
            X_transformed: Dataframe with counter-derived features
        """
        logger.info("Engineering counter features...")
        X_transformed = X.copy()
        
        for prefix, features in self.groups.items():
            if len(features) > 1:
                # Create cumulative sum
                X_transformed[f'{prefix}cumsum'] = X[features].sum(axis=1)
                
                # Create rate of change (difference between consecutive)
                for i in range(len(features) - 1):
                    diff = X[features[i+1]] - X[features[i]]
                    # Counters should only increase, clip negative values
                    X_transformed[f'{prefix}diff_{i}'] = diff.clip(lower=0)
                
                # Create max/min ratios
                max_vals = X[features].max(axis=1)
                min_vals = X[features].min(axis=1)
                X_transformed[f'{prefix}max_min_ratio'] = max_vals / (min_vals + 1e-10)
                
                # Create mean
                X_transformed[f'{prefix}mean'] = X[features].mean(axis=1)
                
                # Create standard deviation
                X_transformed[f'{prefix}std'] = X[features].std(axis=1)
                
                # Drop original counter features
                X_transformed = X_transformed.drop(columns=features)
        
        logger.info(f"Added counter-derived features. New shape: {X_transformed.shape}")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class FeatureSelector:
    """
    Feature selection with multiple strategies.
    
    Selects features based on:
    1. Variance threshold (remove low-variance features)
    2. Mutual information (select most informative features)
    
    Attributes:
        variance_threshold (float): Threshold for variance filtering
        k_features (int): Number of features to select
        selected_features (List[str]): Selected feature names
        selector (SelectKBest): Fitted selector
    """
    
    def __init__(self, variance_threshold: float = 0.01, k_features: int = 100):
        """
        Initialize FeatureSelector.
        
        Args:
            variance_threshold: Threshold for variance filtering
            k_features: Number of features to select
        """
        self.variance_threshold = variance_threshold
        self.k_features = k_features
        self.selected_features: List[str] = []
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit feature selector.
        
        Args:
            X: Input dataframe
            y: Target series
            
        Returns:
            self: Fitted selector
        """
        logger.info("Fitting FeatureSelector...")
        
        # 1. Remove low variance features
        variance_filter = VarianceThreshold(threshold=self.variance_threshold)
        variance_filter.fit(X)
        high_variance_features = X.columns[variance_filter.get_support()].tolist()
        logger.info(f"Kept {len(high_variance_features)} features after variance filtering")
        
        # 2. Select top k features using mutual information
        k = min(self.k_features, len(high_variance_features))
        self.selector = SelectKBest(mutual_info_classif, k=k)
        self.selector.fit(X[high_variance_features], y)
        
        # Get selected features
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = [high_variance_features[i] for i in selected_indices]
        
        logger.info(f"Selected {len(self.selected_features)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with selected features.
        
        Args:
            X: Input dataframe
            
        Returns:
            X_transformed: Dataframe with selected features
        """
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)