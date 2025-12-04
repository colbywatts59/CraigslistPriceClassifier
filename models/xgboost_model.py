from .base_model import BaseClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import feature_engineering
from xgboost import XGBClassifier


class XGBoostClassifier(BaseClassifier):
    
    def __init__(self, n_estimators=500, max_depth=8, learning_rate=0.05, 
                 subsample=0.8, colsample_bytree=0.8, random_state=42, 
                 n_jobs=-1, verbosity=1):
        super().__init__()
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=verbosity,
            objective='multi:softprob'
        )
        self.label_encoder = LabelEncoder()
        self.train_stats = {}
        
        self.feature_cols = [
            # Categoricals (Low Cardinality)
            'manufacturer', 'region', 'type', 'drive', 'fuel', 'transmission', 
            'brand_category', 'paint_color', 'model_simple', 'state'
            
            # Numeric
            'year', 'age', 'odometer', 'miles_per_year',
            'condition_numeric', 'cylinders_numeric',
            
            # Binary Flags (High Signal)
            'is_new', 'is_classic', 
            'is_truck', 'is_suv', 'is_luxury', 'is_economy', 'is_exotic',
            'is_electric', 'is_hybrid', 'is_diesel',
            'is_low_mileage', 'is_high_mileage', 'is_4wd'
        ]
    
    def setup(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y.values if hasattr(y, 'values') else y)
        self.train_stats = {}
        self.numeric_fill_values = {}
        super().setup(X, pd.Series(y_encoded))
        return self
        
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False):
        # 1. Feature Engineering
        df_engineered, _ = feature_engineering.create_engineered_features(
            df, 
            is_training=is_training, 
            train_stats=self.train_stats
        )
        
        # 2. Select Features
        available_cols = [col for col in self.feature_cols if col in df_engineered.columns]
        X = df_engineered[available_cols].copy()
        
      
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
    
    def predict(self, X):
        y_pred_encoded = super().predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return np.array(y_pred)
    
    def predict_proba(self, X):
        return super().predict_proba(X)
    
    def __repr__(self):
        return f"XGBoostClassifier(n_estimators={self.model.n_estimators}, max_depth={self.model.max_depth})"
