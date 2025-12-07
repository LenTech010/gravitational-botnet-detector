"""
Feature Extraction Module

Extracts and preprocesses features from CSV telemetry data for botnet detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts behavioral features from network telemetry CSV data.
    
    Features include:
    - Connection patterns (frequency, duration)
    - Data volume metrics (bytes sent/received)
    - Temporal patterns (time-based features)
    - Protocol distributions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary with feature extraction parameters
        """
        self.config = config or {}
        self.feature_names = []
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load telemetry data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame containing telemetry data
        """
        logger.info(f"Loading telemetry data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from telemetry DataFrame.
        
        Args:
            df: Input DataFrame with telemetry data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info("Extracting features from telemetry data")
        
        features = []
        feature_names = []
        
        # Ensure we have required columns (flexible to different CSV formats)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            # If no numeric columns, try to convert some standard fields
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract statistical features for each numeric column
        for col in numeric_cols:
            feature_names.append(f"{col}_mean")
            feature_names.append(f"{col}_std")
            feature_names.append(f"{col}_max")
        
        # Compute features for each row (connection/event)
        for idx, row in df.iterrows():
            row_features = []
            for col in numeric_cols:
                value = row[col]
                # Mean (normalized by row index + 1)
                row_features.append(float(value) / (idx + 1))
                # Std approximation (deviation from running mean)
                row_features.append(abs(float(value) - float(df[col].iloc[:idx+1].mean())))
                # Max comparison
                row_features.append(float(value) / (float(df[col].max()) + 1e-10))
            
            features.append(row_features)
        
        feature_matrix = np.array(features, dtype=np.float32)
        
        # Handle NaN and Inf values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize features to [0, 1] range
        feature_matrix = self._normalize_features(feature_matrix)
        
        self.feature_names = feature_names
        logger.info(f"Extracted {feature_matrix.shape[1]} features for {feature_matrix.shape[0]} samples")
        
        return feature_matrix, feature_names
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range using min-max scaling.
        
        Args:
            features: Raw feature matrix
            
        Returns:
            Normalized feature matrix
        """
        # Min-max normalization
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (features - min_vals) / range_vals
        
        return normalized
    
    def compute_aggregate_features(self, df: pd.DataFrame, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Compute aggregate features for grouped data (e.g., per IP address).
        
        Args:
            df: Input DataFrame
            group_by: Column name to group by (e.g., 'src_ip', 'dst_ip')
            
        Returns:
            DataFrame with aggregate features
        """
        if group_by and group_by in df.columns:
            logger.info(f"Computing aggregate features grouped by {group_by}")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            agg_df = df.groupby(group_by)[numeric_cols].agg(['mean', 'std', 'sum', 'count'])
            return agg_df
        else:
            logger.warning(f"Group by column '{group_by}' not found, returning ungrouped stats")
            return df.describe()
