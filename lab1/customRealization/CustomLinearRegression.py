import numpy as np


class CustomLinearRegression:
    def __init__(self, lr=0.0001, iters=10000000, tol=1e-4):
        self.lr = lr
        self.iters = iters
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape
        X = np.array(X)
        y = np.array(y)
        self.w = np.zeros(features)
        self.b = 0
        prev_loss = float('inf')

        for _ in range(self.iters):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y

            dw = (2 / samples) * np.dot(X.T, error)
            db = (2 / samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            current_loss = np.mean(error ** 2)

            if np.abs(prev_loss - current_loss) < self.tol:
                break

            prev_loss = current_loss

    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b

    def get_params(self, deep=True):
        return {'lr': self.lr, 'iters': self.iters, 'tol': self.tol}

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (ss_res / ss_tot)
