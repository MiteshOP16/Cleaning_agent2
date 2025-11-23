import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, NearMiss,
    EditedNearestNeighbours, CondensedNearestNeighbour,
    OneSidedSelection, ClusterCentroids, NeighbourhoodCleaningRule
)
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.validation import check_array, check_X_y


class DataBalancer:
    """Handles various data balancing techniques for imbalanced datasets"""
    
    def __init__(self):
        self.balancing_methods = {
            'Random Oversampling': self._random_oversampling,
            'SMOTE': self._smote,
            'Random Undersampling': self._random_undersampling,
            'Tomek Links': self._tomek_links,
            'NearMiss-1': self._nearmiss_1,
            'NearMiss-2': self._nearmiss_2,
            'NearMiss-3': self._nearmiss_3,
            'ENN': self._enn,
            'CNN': self._cnn,
            'OSS': self._oss,
            'Cluster Centroids': self._cluster_centroids,
            'NCR': self._ncr,
            'SMOTE + Tomek Links': self._smote_tomek,
            'SMOTE + ENN': self._smote_enn,
        }
    
    def get_available_methods(self) -> Dict[str, List[str]]:
        """Return categorized list of available balancing methods"""
        return {
            'Oversampling': ['Random Oversampling', 'SMOTE'],
            'Undersampling': [
                'Random Undersampling', 'Tomek Links', 
                'NearMiss-1', 'NearMiss-2', 'NearMiss-3',
                'ENN', 'CNN', 'OSS', 
                'Cluster Centroids', 'NCR'
            ],
            'Hybrid': ['SMOTE + Tomek Links', 'SMOTE + ENN'],
            'Advanced': ['GAN Oversampling', 'VAE Oversampling', 'Cost-Sensitive Learning']
        }
    
    def validate_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, cleaning_history: Dict = None) -> Dict[str, Any]:
        """Validate data before balancing"""
        errors = []
        warnings = []
        
        if df is None or df.empty:
            errors.append("Dataset is empty or not loaded")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        if cleaning_history is None or len(cleaning_history) == 0:
            errors.append("Data has not been cleaned. Please use the Cleaning Wizard to clean your data before balancing.")
        
        if not feature_cols:
            errors.append("No feature columns selected")
        
        if not target_col:
            errors.append("No target column selected")
        
        if target_col and target_col not in df.columns:
            errors.append(f"Target column '{target_col}' not found in dataset")
        
        for col in feature_cols:
            if col not in df.columns:
                errors.append(f"Feature column '{col}' not found in dataset")
        
        if target_col and target_col in df.columns:
            if df[target_col].isnull().any():
                errors.append(f"Target column '{target_col}' contains missing values. Please clean this column using the Cleaning Wizard first.")
            
            unique_vals = df[target_col].nunique()
            if unique_vals < 2:
                errors.append(f"Target column must have at least 2 classes (found {unique_vals})")
            elif unique_vals > 10:
                warnings.append(f"Target column has {unique_vals} classes. Balancing works best with fewer classes.")
        
        categorical_features = []
        for col in feature_cols:
            if col in df.columns:
                if df[col].isnull().any():
                    errors.append(f"Feature column '{col}' contains missing values. Please clean this column using the Cleaning Wizard first.")
                
                if not pd.api.types.is_numeric_dtype(df[col]):
                    categorical_features.append(col)
        
        if categorical_features:
            errors.append(f"Feature columns {categorical_features} are not numeric. Balancing requires numeric features. Please encode categorical variables using the Column Analysis page before balancing, or select only numeric columns.")
        
        if len(df) < 10:
            errors.append("Dataset has fewer than 10 rows. Need more data for balancing.")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_class_distribution(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """Get the distribution of classes in the target column"""
        return df[target_col].value_counts().sort_index()
    
    def balance_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str, 
        method: str,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Apply balancing method to the data"""
        try:
            X = df[feature_cols].values
            y = df[target_col].values
            
            original_dist = self.get_class_distribution(df, target_col)
            
            if method not in self.balancing_methods:
                if method in ['GAN Oversampling', 'VAE Oversampling', 'Cost-Sensitive Learning']:
                    return {
                        'success': False,
                        'error': f"{method} is not yet implemented. This advanced method requires additional dependencies.",
                        'original_distribution': original_dist
                    }
                return {
                    'success': False,
                    'error': f"Unknown balancing method: {method}",
                    'original_distribution': original_dist
                }
            
            balancer_func = self.balancing_methods[method]
            X_balanced, y_balanced = balancer_func(X, y, random_state)
            
            balanced_df = pd.DataFrame(X_balanced, columns=feature_cols)
            balanced_df[target_col] = y_balanced
            
            balanced_dist = pd.Series(y_balanced).value_counts().sort_index()
            
            return {
                'success': True,
                'balanced_data': balanced_df,
                'original_distribution': original_dist,
                'balanced_distribution': balanced_dist,
                'method': method,
                'original_size': len(df),
                'balanced_size': len(balanced_df)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error during balancing: {str(e)}",
                'original_distribution': self.get_class_distribution(df, target_col)
            }
    
    def _random_oversampling(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Random Oversampling"""
        sampler = RandomOverSampler(random_state=random_state)
        return sampler.fit_resample(X, y)
    
    def _smote(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE (Synthetic Minority Over-sampling Technique)"""
        sampler = SMOTE(random_state=random_state, k_neighbors=min(5, len(np.unique(y)) - 1))
        return sampler.fit_resample(X, y)
    
    def _random_undersampling(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Random Undersampling"""
        sampler = RandomUnderSampler(random_state=random_state)
        return sampler.fit_resample(X, y)
    
    def _tomek_links(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tomek Links"""
        sampler = TomekLinks()
        return sampler.fit_resample(X, y)
    
    def _nearmiss_1(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """NearMiss-1"""
        sampler = NearMiss(version=1)
        return sampler.fit_resample(X, y)
    
    def _nearmiss_2(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """NearMiss-2"""
        sampler = NearMiss(version=2)
        return sampler.fit_resample(X, y)
    
    def _nearmiss_3(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """NearMiss-3"""
        sampler = NearMiss(version=3)
        return sampler.fit_resample(X, y)
    
    def _enn(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Edited Nearest Neighbours"""
        sampler = EditedNearestNeighbours()
        return sampler.fit_resample(X, y)
    
    def _cnn(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Condensed Nearest Neighbour"""
        sampler = CondensedNearestNeighbour(random_state=random_state)
        return sampler.fit_resample(X, y)
    
    def _oss(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """One-Sided Selection"""
        sampler = OneSidedSelection(random_state=random_state)
        return sampler.fit_resample(X, y)
    
    def _cluster_centroids(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster Centroids"""
        sampler = ClusterCentroids(random_state=random_state)
        return sampler.fit_resample(X, y)
    
    def _ncr(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Neighbourhood Cleaning Rule"""
        sampler = NeighbourhoodCleaningRule()
        return sampler.fit_resample(X, y)
    
    def _smote_tomek(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE + Tomek Links (Hybrid)"""
        sampler = SMOTETomek(random_state=random_state)
        return sampler.fit_resample(X, y)
    
    def _smote_enn(self, X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE + ENN (Hybrid)"""
        sampler = SMOTEENN(random_state=random_state)
        return sampler.fit_resample(X, y)
