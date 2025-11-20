from sklearn.linear_model import LogisticRegression
from .base_model import BaseClassifier

class LinearClassifier(BaseClassifier):
    
    def __init__(self, max_iter=1000, random_state=42):
        super().__init__()
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state, n_jobs=-1)
    
    def __repr__(self):
        return "LinearClassifier"
