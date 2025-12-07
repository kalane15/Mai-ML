import numpy as np
from scipy.stats import pearsonr


def f_regression(X, y):
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    p_values = np.zeros(n_features)

    for i in range(n_features):
        correlation, p_value = pearsonr(X[:, i], y)

        scores[i] = correlation ** 2 / (1 - correlation ** 2)
        p_values[i] = p_value

    return scores, p_values


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
