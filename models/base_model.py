import pandas as pd
from typing import Protocol
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Classifier(Protocol):
    def setup(self, X: pd.DataFrame, y: pd.Series): ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...



class BaseClassifier:
    
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        self.numeric_fill_values = {}  # Store median values for numeric columns
        self.label_encoder = None  # For encoding class labels



    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False):
        feature_cols = [
            'manufacturer', 'region', 'year', 'age', 'odometer', 'use_mileage',
            'condition', 'cylinders', 'fuel', 'transmission', 'drive', 
            'type', 'paint_color', 'state', 'brand_category', 
            'model_simple', 'year_band', 'miles_per_year'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].copy()
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().any():
                if is_training:
                    fill_value = X[col].median()
                    self.numeric_fill_values[col] = fill_value
                else:
                    # Use stored fill value from training
                    fill_value = self.numeric_fill_values.get(col, 0)
                X[col] = X[col].fillna(fill_value)
        
        # For categorical columns, fill with 'unknown'
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
        X_transformed = self.preprocessor.fit_transform(X_processed)
        
        self.model.fit(X_transformed, y)
        return self
    


    def predict(self, X: pd.DataFrame) -> np.ndarray:

        X_processed, _ = self._prepare_features(X, is_training=False)
        X_transformed = self.preprocessor.transform(X_processed)
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_processed, _ = self._prepare_features(X, is_training=False)
        X_transformed = self.preprocessor.transform(X_processed)
        
        return self.model.predict_proba(X_transformed)
