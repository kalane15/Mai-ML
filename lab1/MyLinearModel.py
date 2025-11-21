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

    def __init__(self, lr=0.01, iters=20000, b=0, way=ModelLearnWay.GD, l1=0, l2=0):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = b
        self.way = way
        self.lambda1_reg = l1
        self.lambda2_reg = l2

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

        print(np.isnan(x_train).any())
        print(np.isnan(y_train).any())

        for _ in range(self.iters):
            preds = np.dot(x_train, self.w) + self.b

            dw = (1 / samples) * np.dot(x_train.T, (preds - y_train)) + self.lambda1_reg * np.sign(self.w)
            db = (1 / samples) * np.sum(preds - y_train)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def fit_sgd(self, x_train, y_train, batch_size=1):
        samples, features = x_train.shape

        print(np.isnan(x_train).any())
        print(np.isnan(y_train).any())

        self.w = np.zeros(features)
        self.b = 0

        rng = np.random.default_rng(seed=42)

        for _ in range(self.iters):
            indices = rng.choice(np.arange(samples), size=batch_size, replace=False)

            x_batch = x_train[indices]
            y_batch = y_train[indices]

            preds = np.dot(x_batch, self.w) + self.b

            dw = np.dot(x_batch.T, (preds - y_batch)) / batch_size
            db = np.sum(preds - y_batch) / batch_size

            self.w -= self.lr * dw
            self.b -= self.lr * db

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

    @staticmethod
    def reshape_to(df1, df2):
        columns_df1 = df1.columns
        missing_columns = [col for col in columns_df1 if col not in df2.columns]
        print(missing_columns)
        missing_df = pd.DataFrame(0, index=df2.index, columns=missing_columns)
        df2 = pd.concat([df2, missing_df], axis=1)
        df2 = df2[columns_df1]

        return df2

    @staticmethod
    def target_encoding_inplace(df, target_column, data=None):
        new_data = {}
        for cat_column in df.select_dtypes(include=['category', 'object']):
            if data is None:
                encoding_map = df.groupby(cat_column)[target_column].mean()
                new_data[cat_column] = encoding_map
            else:
                try:
                    encoding_map = data[cat_column]
                except KeyError:
                    continue
            df[cat_column] = df[cat_column].map(encoding_map)

        return df, new_data

    @staticmethod
    def replace_nans_with_mean_in_all_columns(df):
        df_filled = df.apply(lambda col: col.fillna(col.mean()), axis=0)
        return df_filled

    @staticmethod
    def normalize_z_score(df, mean, std):
        std_values = std
        df = df.loc[:, std_values != 0]
        df = (df - mean) / std
        return df
