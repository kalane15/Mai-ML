import numpy as np

class MyLinearRegressionWithRegularization:
    def __init__(self, alpha=0.1, lr=0.01, regularization_type='L2', p=2, l1_ratio=0.5, iterations=1000):
        self.alpha = alpha
        self.regularization_type = regularization_type
        self.p = p
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.iterations):
            preds = X @ self.weights + self.bias
            errors = preds - y

            dw = (2 / m) * (X.T @ errors)
            db = (2 / m) * np.sum(errors)

            if self.regularization_type == 'L1':
                dw += self.alpha * np.sign(self.weights)
            elif self.regularization_type == 'L2':
                dw += 2 * self.alpha * self.weights
            elif self.regularization_type == 'Lp':
                dw += self.alpha * self.p * np.sign(self.weights) * np.abs(self.weights) ** (self.p - 1)
            elif self.regularization_type == 'L1+L2':
                dw += self.alpha * (self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * 2 * self.weights)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return X @ self.weights + self.bias

    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'lr': self.lr,
            'regularization_type': self.regularization_type,
            'p': self.p,
            'l1_ratio': self.l1_ratio,
            'iterations': self.iterations
        }

