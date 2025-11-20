import numpy as np
from enum import Enum


class ModelLearnWay(Enum):
    GD = 0
    SGD = 1
    ANALYTICAL = 2


class MyLinearModel:
    @staticmethod
    def mse(y_true, y_pred):
        loss = 0
        for i in range(len(y_true)):
            y = y_true[i]
            pred = y_pred[i]
            loss += (y - pred) ** 2
        loss /= len(y_true)
        return loss

    def __init__(self, lr=0.01, iters=2000, b=0):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = b

    def fit(self, x_train, y_train, way=ModelLearnWay.GD):
        if way is ModelLearnWay.GD:
            self.fit_gd(x_train, y_train)
        elif way is ModelLearnWay.SGD:
            self.fit_sgd(x_train, y_train)
        elif way is ModelLearnWay.ANALYTICAL:
            self.fit_analytical(x_train, y_train)

    def fit_gd(self, x_train, y_train):
        samples, features = x_train.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iters):
            preds = np.dot(x_train, self.w) + self.b

            dw = (1 / samples) * np.dot(x_train.T, (preds - y_train))
            db = (1 / samples) * np.sum(preds - y_train)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def fit_sgd(self, x_train, y_train):
        samples, features = x_train.shape
        self.w = np.zeros(features)
        self.b = 0

        rng = np.random.default_rng(seed=42)

        for _ in range(self.iters):
            indices = rng.choice(np.arange(x_train.shape[0]), size=1, replace=False)
            X_batch = x_train[indices]
            y_batch = y_train[indices]

            preds = np.dot(X_batch, self.w) + self.b

            dw = np.dot(X_batch.T, (preds - y_batch)) / 1
            db = np.sum(preds - y_batch) / 1

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def fit_analytical(self, x_train, y_train):
        X_b = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
        self.w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train

    def predict(self, x_test):
        preds = np.dot(x_test, self.w) + self.b
        return preds
