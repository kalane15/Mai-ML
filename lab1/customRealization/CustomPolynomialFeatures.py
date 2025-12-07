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
        output = [X]
        for degree in range(2, self.degree + 1):
            for combo in combinations_with_replacement(range(n_features), degree):
                output.append(np.prod(X[:, combo], axis=1).reshape(-1, 1))
        result = np.hstack(output)
        if not self.include_bias:
            result = result[:, 1:]
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
