from sklearn.ensemble import RandomForestClassifier as RFClassifier
from .base_model import BaseClassifier

class RandomForestClassifier(BaseClassifier):
    
    def __init__(self, n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42):
        super().__init__()
        self.model = RFClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  
        )
    
    def __repr__(self):
        return f"RandomForestClassifier(n_estimators={self.model.n_estimators}, max_depth={self.model.max_depth})"
