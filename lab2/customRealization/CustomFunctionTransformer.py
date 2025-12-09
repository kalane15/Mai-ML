class CustomFunctionTransformer:
    def __init__(self, func):
        self.func = func

    def fit(self, X):
        return self

    def transform(self, X):
        return self.func(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"func": self.func}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self