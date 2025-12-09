import numpy as np


class CustomRidgeAnalytical:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w = None
        self.b = None
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]

        I = np.eye(n + 1)
        I[0, 0] = 0

        XtX = X_b.T.dot(X_b)
        ridge_matrix = XtX + self.alpha * I
        res = np.linalg.pinv(ridge_matrix).dot(X_b.T).dot(y)

        self.b = res[0]
        self.w = res[1:]
        self.intercept_ = self.b
        self.coef_ = self.w
        self.n_features_in_ = n
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)

