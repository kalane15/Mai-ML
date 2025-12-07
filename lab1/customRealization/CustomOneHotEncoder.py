import numpy as np


class CustomOneHotEncoder:
    def __init__(self):
        self.categories_ = None
        self.n_features_out_ = 0

    def fit(self, X):
        self.categories_ = [np.unique(X[:, col]) for col in range(X.shape[1])]
        return self

    def transform(self, X):
        encoded = np.zeros((X.shape[0], sum(len(cat) for cat in self.categories_)))
        start_idx = 0

        for col in range(X.shape[1]):
            col_data = X[:, col]
            for i, category in enumerate(self.categories_[col]):
                col_encoded = (col_data == category)
                encoded[:, start_idx + i] = col_encoded
            start_idx += len(self.categories_[col])

        return encoded

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

