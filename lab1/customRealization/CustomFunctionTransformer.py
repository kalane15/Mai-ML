class CustomFunctionTransformer:
    def __init__(self, func):
        self.func = func

    def fit(self, X):
        return self

    def transform(self, X):
        return self.func(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)