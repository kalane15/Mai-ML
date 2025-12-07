import numpy as np


class CustomPipeline:
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}

    def fit(self, X, y=None):
        for name, step in self.steps:
            step.fit(X, y)
            X = step.transform(X)
        return X

    def transform(self, X):
        for name, step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        for name, step in self.steps:
            step.fit(X, y)
            X = step.transform(X)
        return X

    def predict(self, X):
        last_step_name, last_step = self.steps[-1]
        return last_step.predict(X)


