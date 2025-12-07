import numpy as np


class CustomSGDLinearRegression:
    def __init__(self, lr=0.0001, batch_size=1, iters=1000, tol=1e-4):
        self.lr = lr
        self.iters = iters
        self.tol = tol
        self.w = None
        self.b = None
        self.batch_size = batch_size

    def fit(self, X, y):
        samples, features = X.shape
        X = np.array(X)
        y = np.array(y)

        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iters):
            indices = np.arange(samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                y_pred = np.dot(X_batch, self.w) + self.b
                error = y_pred - y_batch

                dw = (2 / len(X_batch)) * np.dot(X_batch.T, error)
                db = (2 / len(X_batch)) * np.sum(error)

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b

    def get_params(self, deep=True):
        return {'lr': self.lr, 'iters': self.iters, 'tol': self.tol}
