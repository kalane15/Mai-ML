import numpy as np


class CustomOneHotEncoder:
    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_out_ = 0

    def fit(self, X):
        X = np.asarray(X)
        self.categories_ = [np.unique(X[:, col]) for col in range(X.shape[1])]
        self.n_features_out_ = sum(len(cat) for cat in self.categories_)
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise RuntimeError("CustomOneHotEncoder is not fitted yet.")

        X = np.asarray(X)
        encoded = np.zeros((X.shape[0], self.n_features_out_), dtype=float)
        start_idx = 0

        for col in range(X.shape[1]):
            col_data = X[:, col]
            cats = self.categories_[col]
            cat_to_idx = {cat: i for i, cat in enumerate(cats)}
            for row_idx, value in enumerate(col_data):
                if value in cat_to_idx:
                    encoded[row_idx, start_idx + cat_to_idx[value]] = 1.0
                else:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category {value} in column {col}")
                    elif self.handle_unknown == "ignore":
                        # leave zeros for unknown category (sklearn behavior)
                        continue
                    else:
                        raise ValueError("handle_unknown must be 'ignore' or 'error'")
            start_idx += len(cats)

        return encoded

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"handle_unknown": self.handle_unknown}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

