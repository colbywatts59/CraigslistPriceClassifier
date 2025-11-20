from sklearn.ensemble import GradientBoostingClassifier as GBClassifier
from .base_model import BaseClassifier

class GradientBoostingClassifier(BaseClassifier):
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=0):
        super().__init__()
        self.model = GBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose  # Set to 1 to see progress
        )
    
    def __repr__(self):
        return f"GradientBoostingClassifier(n_estimators={self.model.n_estimators}, max_depth={self.model.max_depth}, learning_rate={self.model.learning_rate})"
