import pandas as pd
import numpy as np
from .base_model import BaseClassifier

class BaselineClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.manufacturer_median_bins = {}  
        self.global_median_bin = None
        self.classes_ = None 
    
    def setup(self, X: pd.DataFrame, y: pd.Series):

        df = X.copy()
        df['price_bin'] = y
        
        df['price_bin'] = df['price_bin'].astype(str)
        y_str = y.astype(str)
        
        # Calculate median bin for each manufacturer
        if 'manufacturer' in df.columns:
            def get_mode(x):
                mode_vals = x.mode()
                if len(mode_vals) > 0:
                    return mode_vals.iloc[0]
                else:
                    return x.iloc[0]
            
            manufacturer_modes = df.groupby('manufacturer')['price_bin'].agg(get_mode)
            self.manufacturer_median_bins = manufacturer_modes.to_dict()
        else:
            # If manufacturer column doesn't exist, use global median
            self.manufacturer_median_bins = {}
        
        mode_vals = y_str.mode()
        self.global_median_bin = mode_vals.iloc[0] if len(mode_vals) > 0 else y_str.iloc[0]
        
        # Store unique classes in sorted order for predict_proba
        all_classes = list(self.manufacturer_median_bins.values()) + [self.global_median_bin]
        self.classes_ = sorted(list(set(all_classes)))
        
        return self
    

    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        
        if 'manufacturer' in X.columns:
            for manufacturer in X['manufacturer']:
                # Look up manufacturer, use global median as fallback
                manufacturer_str = str(manufacturer) if pd.notna(manufacturer) else None
                if manufacturer_str and manufacturer_str in self.manufacturer_median_bins:
                    predictions.append(self.manufacturer_median_bins[manufacturer_str])
                else:
                    predictions.append(self.global_median_bin)
        else:
            # If no manufacturer column, predict global median for all
            predictions = [self.global_median_bin] * len(X)
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:

        if not hasattr(self, 'classes_'):
            all_classes = list(self.manufacturer_median_bins.values()) + [self.global_median_bin]
            self.classes_ = sorted(list(set(all_classes)))
        
        n_classes = len(self.classes_)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        probabilities = []
        manufacturers = X['manufacturer'] if 'manufacturer' in X.columns else [None] * len(X)
        
        for manufacturer in manufacturers:
            prob = np.zeros(n_classes)
            manufacturer_str = str(manufacturer) if pd.notna(manufacturer) else None
            
            if manufacturer_str and manufacturer_str in self.manufacturer_median_bins:
                predicted_bin = self.manufacturer_median_bins[manufacturer_str]
                idx = class_to_idx.get(predicted_bin, 0)
                prob[idx] = 0.9  
                prob += 0.1 / n_classes
            else:
                idx = class_to_idx.get(self.global_median_bin, 0)
                prob[idx] = 0.7
                prob += 0.3 / n_classes
            
            probabilities.append(prob)
        
        return np.array(probabilities)
    
    def __repr__(self):
        return "BaselineClassifier"

