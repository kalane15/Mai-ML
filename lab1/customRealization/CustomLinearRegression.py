import numpy as np


class CustomLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape
        X = np.array(X)
        y = np.array(y)
        X_b = np.c_[np.ones((samples, 1)), X]
        res = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.b = res[0]
        self.w = res[1:]

    def predict(self, x_test):
        return np.dot(x_test, self.w) + self.b

    def get_params(self, deep=True):
        return {}
