import numpy as np
import pandas as pd
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

    def __init__(self, lr=0.01, iters=200, b=0, way=ModelLearnWay.GD):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = b
        self.way = way

    def fit(self, x_train, y_train):
        if self.way is ModelLearnWay.GD:
            self.fit_gd(x_train, y_train)
        elif self.way is ModelLearnWay.SGD:
            self.fit_sgd(x_train, y_train)
        elif self.way is ModelLearnWay.ANALYTICAL:
            self.fit_analytical(x_train, y_train)

    def fit_gd(self, x_train, y_train):
        samples, features = x_train.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iters):
            preds = np.dot(x_train, self.w) + self.b
            print(_)
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


class MyCrossValidation:
    @staticmethod
    def k_fold_cross_validation(model, x, y, k=5):
        samples = len(x)

        indices = np.arange(samples)
        np.random.shuffle(indices)

        fold_size = samples // k
        mse_scores = []

        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = np.mean((y_test - y_pred) ** 2)
            mse_scores.append(mse)

        avg_mse = np.mean(mse_scores)
        return avg_mse


class DataProcessor:
    @staticmethod
    def leave_one_out_cross_validation(model, x, y):
        return MyCrossValidation.k_fold_cross_validation(model, x, y, len(x))

    @staticmethod
    def df_to_matrix_numeric(df, target_column=None):
        """
        Assumes that all columns in df are numeric features
        or categorical features encoded appropriately
        """
        df_copy = df.copy()

        if target_column:
            y = df_copy[target_column].values
            df_copy = df_copy.drop(columns=[target_column])
        else:
            y = None

        x_numeric = df_copy.values
        x = np.hstack([np.ones((x_numeric.shape[0], 1)), x_numeric])

        return x, y

    @staticmethod
    def apply_one_hot_encoding(df, encoding_column):
        one_hot = pd.get_dummies(df[encoding_column], prefix=encoding_column)

        df_encoded = df.drop(columns=[encoding_column])
        df_encoded = pd.concat([df_encoded, one_hot], axis=1)

        return df_encoded

    @staticmethod
    def apply_one_hot_encoding_all(df):
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns

        for column in non_numeric_columns:
            df = DataProcessor.apply_one_hot_encoding(df, column)
        return df
