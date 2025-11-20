import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .base_model import BaseClassifier
from collections import Counter


class CollaborativeFilteringClassifier(BaseClassifier):

    def __init__(self, n_components=50, n_neighbors=20, random_state=42):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.pca = None
        self.scaler = None
        self.nn_model = None
        self.X_train_encoded = None
        self.y_train = None
        self.feature_names = None
        
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False):

        feature_cols = [
            'manufacturer', 'region', 'year', 'age', 'odometer', 'use_mileage',
            'condition', 'cylinders', 'fuel', 'transmission', 'drive', 
            'type', 'paint_color', 'state', 'brand_category', 
            'model_simple', 'year_band'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].copy()
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().any():
                if is_training:
                    fill_value = X[col].median()
                    self.numeric_fill_values[col] = fill_value
                else:
                    fill_value = self.numeric_fill_values.get(col, 0)
                X[col] = X[col].fillna(fill_value)
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('unknown')
        
        return X, available_cols
    
    def _build_preprocessor(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', ohe, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def setup(self, X: pd.DataFrame, y: pd.Series):
        X_processed, feature_cols = self._prepare_features(X, is_training=True)
        self.feature_names = feature_cols
        
        self.preprocessor = self._build_preprocessor(X_processed)
        X_encoded = self.preprocessor.fit_transform(X_processed)
        
        # Apply PCA 
        self.pca = PCA(n_components=min(self.n_components, X_encoded.shape[1], X_encoded.shape[0] - 1), 
                       random_state=self.random_state)
        X_reduced = self.pca.fit_transform(X_encoded)
        
        # Store encoded features and labels
        self.X_train_encoded = X_reduced
        self.y_train = y.values
        
        self.classes_ = sorted(list(set(self.y_train)))
        
        self.nn_model = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X_reduced)), 
                                         metric='cosine', 
                                         n_jobs=-1)
        self.nn_model.fit(X_reduced)
        
        print(f"  PCA: Reduced from {X_encoded.shape[1]} to {X_reduced.shape[1]} dimensions")
        print(f"  Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:

        X_processed, _ = self._prepare_features(X, is_training=False)
        X_encoded = self.preprocessor.transform(X_processed)
        X_reduced = self.pca.transform(X_encoded)
        
        # Find nearest neighbors for each test sample
        distances, indices = self.nn_model.kneighbors(X_reduced)
        
        predictions = []
        for i, (neighbor_indices, neighbor_distances) in enumerate(zip(indices, distances)):
            # Get price bins of neighbors
            neighbor_bins = self.y_train[neighbor_indices]
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            weights = 1.0 / (neighbor_distances + epsilon)
            
            bin_weights = {}
            for bin_val, weight in zip(neighbor_bins, weights):
                bin_str = str(bin_val)
                bin_weights[bin_str] = bin_weights.get(bin_str, 0) + weight
            
            if bin_weights:
                most_common = max(bin_weights.items(), key=lambda x: x[1])[0]
            else:
                neighbor_bins_str = [str(bin_val) for bin_val in neighbor_bins]
                most_common = Counter(neighbor_bins_str).most_common(1)[0][0]
            
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_processed, _ = self._prepare_features(X, is_training=False)
        X_encoded = self.preprocessor.transform(X_processed)
        X_reduced = self.pca.transform(X_encoded)
        
        distances, indices = self.nn_model.kneighbors(X_reduced)
        
        unique_classes = sorted(list(set(self.y_train)))
        n_classes = len(unique_classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        probabilities = []
        for neighbor_indices, neighbor_distances in zip(indices, distances):
            neighbor_bins = self.y_train[neighbor_indices]
            
            epsilon = 1e-6
            weights = 1.0 / (neighbor_distances + epsilon)
            
            prob = np.zeros(n_classes)
            total_weight = 0
            
            for bin_val, weight in zip(neighbor_bins, weights):
                bin_str = str(bin_val)
                if bin_str in class_to_idx:
                    idx = class_to_idx[bin_str]
                    prob[idx] += weight
                    total_weight += weight
            
            if total_weight > 0:
                prob = prob / total_weight
            else:
                prob = np.ones(n_classes) / n_classes
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def __repr__(self):
        return f"CollaborativeFilteringClassifier(n_components={self.n_components}, n_neighbors={self.n_neighbors})"

