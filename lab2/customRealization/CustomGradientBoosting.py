from copy import deepcopy

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class CustomGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 loss='squared', subsample=1.0, random_state=None,
                 base_model=DecisionTreeRegressor(max_depth=3, min_samples_split=2)):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.initial = None
        self.rng = np.random.RandomState(random_state)
        self.base_model = base_model

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        self.trees = []

        if self.loss == 'squared':
            self.initial = np.mean(y)
            F = np.full(n_samples, self.initial, dtype=float)

            for m in range(self.n_estimators):
                residuals = y - F

                if self.subsample < 1.0:
                    idx = self.rng.choice(n_samples, int(self.subsample * n_samples), replace=False)
                else:
                    idx = np.arange(n_samples)

                tree = deepcopy(self.base_model)
                tree.fit(X[idx], residuals[idx])
                update = tree.predict(X)
                F = F + self.learning_rate * update

                self.trees.append(tree)

        elif self.loss == 'logistic':
            p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
            self.initial = np.log(p / (1 - p))
            F = np.full(n_samples, self.initial, dtype=float)

            for m in range(self.n_estimators):
                prob = self._sigmoid(F)
                residuals = y - prob

                if self.subsample < 1.0:
                    idx = self.rng.choice(n_samples, int(self.subsample * n_samples), replace=False)
                else:
                    idx = np.arange(n_samples)

                tree = deepcopy(self.base_model)
                tree.fit(X[idx], residuals[idx])
                update = tree.predict(X)
                F = F + self.learning_rate * update

                self.trees.append(tree)
        else:
            raise ValueError("loss must be 'squared' or 'logistic'")

        return self

    def predict_inner(self, X):
        X = np.asarray(X)
        if self.initial is None:
            raise ValueError("Model is not fitted")
        F = np.full(X.shape[0], self.initial, dtype=float)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def predict(self, X):
        if self.loss == 'squared':
            return self.predict_inner(X)
        else:
            logits = self.predict_inner(X)
            probs = self._sigmoid(logits)
            return (probs >= 0.5).astype(int)

    def predict_log(self, X):
        logits = self.predict_inner(X)
        probs = self._sigmoid(logits)
        return np.vstack([1 - probs, probs]).T
