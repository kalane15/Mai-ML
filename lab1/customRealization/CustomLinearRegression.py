import numpy as np


class CustomLinearRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape

        X = np.array(X)
        y = np.array(y)
        self.w = np.zeros(features)
        self.b = 0
        while True:
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y

            dw = (1 / samples) * np.dot(X.T, error)
            db = (1 / samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b

    def get_params(self, deep=True):
        return {'lr': self.lr, 'iters': self.iters}
