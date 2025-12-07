import numpy as np


class CustomRobustScaler:
    def __init__(self, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True):
        self.quantile_range = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        q_min, q_max = self.quantile_range
        X = np.asarray(X)

        Q1 = np.percentile(X, q_min, axis=0)
        Q3 = np.percentile(X, q_max, axis=0)

        self.center_ = np.median(X, axis=0) if self.with_centering else np.zeros(X.shape[1])
        self.scale_ = (Q3 - Q1) if self.with_scaling else np.ones(X.shape[1])

        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.center_ is None or self.scale_ is None:
            raise ValueError("The scaler is not fitted yet.")
        X_centered = X - self.center_ if self.with_centering else X
        return X_centered / self.scale_ if self.with_scaling else X_centered

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {
            "quantile_range": self.quantile_range,
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
