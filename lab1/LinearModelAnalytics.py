import numpy as np

class Lin_reg_analytic:
    
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    
    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.weights