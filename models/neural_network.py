from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from .base_model import BaseClassifier
import numpy as np
import pandas as pd

class NeuralNetworkClassifier(BaseClassifier):
 
    def __init__(self, hidden_layer_sizes=(100, 50), activation='relu', 
                 solver='adam', alpha=0.0001, learning_rate='constant',
                 learning_rate_init=0.001, max_iter=500, random_state=42,
                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=10):
        super().__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha, 
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            verbose=True 
        )
        self.label_encoder = LabelEncoder()

        self.feature_cols = [
            # Categoricals (Low Cardinality)
            'manufacturer', 'region', 'type', 'drive', 'fuel', 'transmission', 
            'brand_category', 'state', 'model_simple',
            
            # Numeric
            'year', 'age', 'odometer', 'miles_per_year',
            'condition_numeric', 'cylinders_numeric',
            
            # Binary Flags (High Signal for Similarity)
            'is_new', 'is_classic', 
            'is_truck', 'is_suv', 'is_luxury', 'is_economy', 'is_exotic',
            'is_electric', 'is_diesel', 'is_4wd'
        ]
    
    def setup(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y.values if hasattr(y, 'values') else y)
        super().setup(X, pd.Series(y_encoded))
        return self
    
    def predict(self, X):
        y_pred_encoded = super().predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return np.array(y_pred)
    
    def predict_proba(self, X):
        return super().predict_proba(X)
    
    def __repr__(self):
        return f"NeuralNetworkClassifier(hidden_layers={self.model.hidden_layer_sizes}, activation={self.model.activation})"

