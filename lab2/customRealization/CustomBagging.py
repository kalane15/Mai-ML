import numpy as np
from sklearn.base import clone


class CustomBagging:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.models = []

        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            model = clone(self.base_model)
            model.fit(X_boot, y_boot)
            self.models.append(model)

        return self

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        final = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=preds
        )
        return final
