import numpy as np


class CustomRobustScaler:
    def __init__(self):
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        self.center_ = (Q1 + Q3) / 2
        self.scale_ = Q3 - Q1

        self.scale_[self.scale_ == 0] = 1

        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
