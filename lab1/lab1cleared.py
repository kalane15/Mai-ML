import numpy as np
import pandas as pd
from customRealization.CustomSimpleImputer import CustomSimpleImputer
from customRealization.CustomFunctionTransformer import CustomFunctionTransformer
from customRealization.CustomRobustScaler import CustomRobustScaler
from customRealization.CustomPolynomialFeatures import CustomPolynomialFeatures
from customRealization.CustomOneHotEncoder import CustomOneHotEncoder
from customRealization.CustomRidgeAnalytical import CustomRidgeAnalytical
from customRealization.CustomSelectPercentile import CustomSelectPercentile


class CustomPipelineModel:
    def __init__(self, alpha=5.0, percentile=25):
        self.alpha = alpha
        self.percentile = percentile
        # Will be instantiated in fit
        self.numeric_imputer = None
        self.numeric_func_transformer = None
        self.numeric_scaler = None
        self.numeric_poly = None
        self.categorical_imputer = None
        self.encoder = None
        self.feature_selector = None
        self.regressor = None
        self.is_fitted_ = False

    def _init_components(self):
        self.numeric_imputer = CustomSimpleImputer(strategy="median")
        self.numeric_func_transformer = CustomFunctionTransformer(signed_log1p)
        self.numeric_scaler = CustomRobustScaler()
        self.numeric_poly = CustomPolynomialFeatures(degree=3, include_bias=False)
        self.categorical_imputer = CustomSimpleImputer(strategy="most_frequent")
        self.encoder = CustomOneHotEncoder(handle_unknown="ignore")
        self.feature_selector = CustomSelectPercentile(percentile=self.percentile)
        self.regressor = CustomRidgeAnalytical(alpha=self.alpha)

    def _pipeline(self, df, fit: bool = False):
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
        self._init_components()
        X_proc = self._pipeline(X, fit=True)
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


def make_bins(df, col_name, num_bins=8):
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(df[f'{col_name}'], percentiles)
    bin_edges = np.unique(bin_edges)
    labels = [f"{col_name}{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(len(bin_edges) - 1)]
    df[f"{col_name}Binned"] = pd.cut(
        df[f"{col_name}"], bins=bin_edges, labels=labels, include_lowest=True
    ).astype(str)

    return df


def modify_features(df):
    numerical_columns = df.select_dtypes(include=['number'])
    df[numerical_columns.columns] = numerical_columns.fillna(numerical_columns.median())
    df = df.drop("AnnualIncome", axis=1)
    df = df.drop("Age", axis=1)
    df = df.drop("MonthlyLoanPayment", axis=1)
    df = df.drop('ApplicationDate', axis=1)
    df['MonthlyIncomeToLoanAmountRatio'] = df['MonthlyIncome'] / df['LoanAmount']
    df['LoanToValueRatio'] = df['LoanAmount'] / df['TotalAssets']
    df['NetWorthToLoanAmountRatio'] = df['NetWorth'] / df['LoanAmount']
    return df


def signed_log1p(data: np.ndarray) -> np.ndarray:
    return np.sign(data) * (np.log1p(np.abs(data)))


df_train = pd.read_csv("train.csv")

df_train = df_train[df_train['RiskScore'] > 0]
df_train = df_train[df_train['RiskScore'] < 100]

df_train = modify_features(df_train)
df_train_no_target = df_train.drop(columns='RiskScore')
y_train = df_train['RiskScore'].to_numpy()

final_model = CustomPipelineModel(alpha=5.0, percentile=25)
final_model.fit(df_train_no_target, y_train)

df_test = pd.read_csv("test.csv")
df_test = df_test.drop(columns="ID")
df_test = modify_features(df_test)

test_predictions = final_model.predict(df_test)
print("Test prediction stats:", test_predictions)

df_preds = pd.DataFrame({
    'ID': range(0, len(test_predictions)),
    'RiskScore': test_predictions
})
df_preds.to_csv("res.csv", index=False)
