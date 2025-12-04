from .base_model import BaseClassifier
from .neural_network import NeuralNetworkClassifier
from .xgboost_model import XGBoostClassifier
from .collaborative_filtering import CollaborativeFilteringClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class EnsembleClassifier(BaseClassifier):
    
    def __init__(self):
        super().__init__()
        self.nn = NeuralNetworkClassifier(hidden_layer_sizes=(200, 100), max_iter=300)
        self.xgb = XGBoostClassifier(n_estimators=500, max_depth=8, learning_rate=0.05)
        self.cf = CollaborativeFilteringClassifier(n_components=0.95, n_neighbors=50)
        
        self.models = [
            ('neural_network', self.nn),
            ('xgboost', self.xgb),
            ('collab_filtering', self.cf)
        ]
        
        self.weights = {
            'neural_network': 0.4,
            'xgboost': 0.4,
            'collab_filtering': 0.2
        }
        
        self.label_encoder = LabelEncoder()
        
    def setup(self, X: pd.DataFrame, y: pd.Series):
        y_encoded = self.label_encoder.fit_transform(y.values if hasattr(y, 'values') else y)
        y_series = pd.Series(y_encoded)
        
        print("\nTraining Ensemble Components")
        for name, model in self.models:
            print(f"Training {name}...")
            model.setup(X, y)
            
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        
        for name, model in self.models:
            p = model.predict_proba(X)
            
            weighted_p = p * self.weights[name]
            probas.append(weighted_p)
            
        avg_proba = np.sum(probas, axis=0)
        
        y_pred_indices = np.argmax(avg_proba, axis=1)
        
        y_pred = self.label_encoder.inverse_transform(y_pred_indices)
        return np.array(y_pred)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for name, model in self.models:
            p = model.predict_proba(X)
            weighted_p = p * self.weights[name]
            probas.append(weighted_p)
        
        return np.sum(probas, axis=0)

    def __repr__(self):
        return "EnsembleClassifier(NN + XGB + CF)"

