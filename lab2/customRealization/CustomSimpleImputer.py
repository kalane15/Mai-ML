import numpy as np
import pandas as pd


class CustomSimpleImputer:
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.fill_value = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)

        if self.strategy == "median":
            self.fill_value = X_df.median(axis=0)
        elif self.strategy == "most_frequent":
            self.fill_value = X_df.mode(axis=0).iloc[0]
        elif self.strategy == "mean":
            self.fill_value = X_df.mean(axis=0)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_filled = X_df.fillna(self.fill_value)
        return X_filled.values

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"strategy": self.strategy}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self