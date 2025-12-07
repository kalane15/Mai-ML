import numpy as np


class CustomLinearRegressionGD:
    def __init__(self, lr=0.01, iters=1000, tol=1e-4):
        self.lr = lr
        self.iters = iters
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        print("called")
        samples, features = X.shape
        X = np.array(X)
        y = np.array(y)

        self.w = np.zeros(features)
        self.b = 0
        prev_loss = float('inf')

        for epoch in range(self.iters):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y

            dw = (2 / samples) * np.dot(X.T, error)
            db = (2 / samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db
            # print(self.w)
            loss = np.mean(error ** 2)
            if np.abs(prev_loss - loss) < self.tol:
                print(f"Converged at epoch {epoch}.")
                return
            prev_loss = loss

    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b

    def get_params(self, deep=True):
        return {'lr': self.lr, 'iters': self.iters, 'tol': self.tol}
