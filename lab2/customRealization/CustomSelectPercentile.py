import numpy as np
from scipy.stats import f


def f_regression(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    X = X - X.mean(axis=0)

    denom = np.sqrt((X ** 2).sum(axis=0) * (y ** 2).sum())
    denom[denom == 0] = np.inf
    corr = (X * y[:, None]).sum(axis=0) / denom
    corr = np.clip(corr, -1, 1)

    n = y.size
    F = (corr ** 2 / (1 - corr ** 2)) * (n - 2)
    F[np.isinf(F)] = np.finfo(float).max

    p_values = f.sf(F, 1, n - 2)
    return F, p_values


class CustomSelectPercentile:
    def __init__(self, score_func=f_regression, percentile=25):
        self.score_func = score_func
        self.percentile = percentile
        self.selected_indices = None

    def fit(self, X, y):
        scores, _ = self.score_func(X, y)
        threshold = np.percentile(scores, 100 - self.percentile)
        self.selected_indices = np.where(scores >= threshold)[0]
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"score_func": self.score_func, "percentile": self.percentile}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
