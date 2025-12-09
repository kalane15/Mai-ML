import numpy as np

from CustomSimpleImputer import CustomSimpleImputer
from CustomFunctionTransformer import CustomFunctionTransformer
from CustomRobustScaler import CustomRobustScaler
from CustomPolynomialFeatures import CustomPolynomialFeatures
from CustomOneHotEncoder import CustomOneHotEncoder
from CustomRidgeAnalytical import CustomRidgeAnalytical
from CustomSelectPercentile import CustomSelectPercentile


def signed_log1p(data: np.ndarray) -> np.ndarray:
    return np.sign(data) * (np.log1p(np.abs(data)))


class CustomPipelineModel:
    def __init__(self, alpha=5.0, percentile=25, regressor=None):
        self.alpha = alpha
        self.percentile = percentile
        self.numeric_imputer = None
        self.numeric_func_transformer = None
        self.numeric_scaler = None
        self.numeric_poly = None
        self.categorical_imputer = None
        self.encoder = None
        self.feature_selector = None
        self.regressor = None
        self.is_fitted_ = False
        self.regressor = regressor

    def _init_components(self):
        self.numeric_imputer = CustomSimpleImputer(strategy="median")
        self.numeric_func_transformer = CustomFunctionTransformer(signed_log1p)
        self.numeric_scaler = CustomRobustScaler()
        self.numeric_poly = CustomPolynomialFeatures(degree=3, include_bias=False)
        self.categorical_imputer = CustomSimpleImputer(strategy="most_frequent")
        self.encoder = CustomOneHotEncoder(handle_unknown="ignore")
        self.feature_selector = CustomSelectPercentile(percentile=self.percentile)
        if self.regressor is None:
            self.regressor = CustomRidgeAnalytical(alpha=self.alpha)

    def pipeline(self, df, fit: bool = False):
        self._init_components()
        numeric_features = df.select_dtypes(include=[np.number]).to_numpy()
        categorical_features = df.select_dtypes(exclude=[np.number]).to_numpy()

        # Numeric pipeline
        if fit:
            numeric_features = self.numeric_imputer.fit_transform(numeric_features)
            numeric_features = self.numeric_func_transformer.fit_transform(numeric_features)
            numeric_features = self.numeric_scaler.fit_transform(numeric_features)
            numeric_features = self.numeric_poly.fit_transform(numeric_features)
        else:
            numeric_features = self.numeric_imputer.transform(numeric_features)
            numeric_features = self.numeric_func_transformer.transform(numeric_features)
            numeric_features = self.numeric_scaler.transform(numeric_features)
            numeric_features = self.numeric_poly.transform(numeric_features)

        # Categorical pipeline
        if categorical_features.size == 0:
            categorical_encoded = np.zeros((numeric_features.shape[0], 0))
        else:
            if fit:
                categorical_features = self.categorical_imputer.fit_transform(categorical_features)
                categorical_encoded = self.encoder.fit_transform(categorical_features)
            else:
                categorical_features = self.categorical_imputer.transform(categorical_features)
                categorical_encoded = self.encoder.transform(categorical_features)

        combined = np.hstack((numeric_features, categorical_encoded))
        return combined

    def fit(self, X, y):
        X_proc = self.pipeline(X, fit=True)
        X_sel = self.feature_selector.fit_transform(X_proc, y)
        self.regressor.fit(X_sel, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("CustomPipelineModel is not fitted yet.")
        X_proc = self._pipeline(X, fit=False)
        X_sel = self.feature_selector.transform(X_proc)
        return self.regressor.predict(X_sel)

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "percentile": self.percentile}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
