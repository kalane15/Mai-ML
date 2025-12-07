import numpy as np


class CustomRidge:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, learning_rate=0.01):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.coef_ = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)

        for i in range(self.max_iter):
            predictions = X.dot(self.coef_)
            error = predictions - y
            gradient = (2 / n_samples) * X.T.dot(error) + 2 * self.alpha * self.coef_
            self.coef_ -= self.learning_rate * gradient

        return self

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.coef_)

    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'learning_rate': self.learning_rate
        }