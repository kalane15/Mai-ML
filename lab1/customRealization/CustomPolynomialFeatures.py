import numpy as np


class CustomPolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X):
        return self

    def transform(self, X):
        from itertools import combinations_with_replacement

        n_samples, n_features = X.shape
        output = [np.ones((n_samples, 1))] if self.include_bias else []
        output.append(X)
        for degree in range(2, self.degree + 1):
            for combo in combinations_with_replacement(range(n_features), degree):
                output.append(np.prod(X[:, combo], axis=1).reshape(-1, 1))
        return np.hstack(output)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"degree": self.degree, "include_bias": self.include_bias}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
