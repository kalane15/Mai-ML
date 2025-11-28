from __future__ import annotations

import inspect

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler, \
    KBinsDiscretizer

from MyLinearModel import *
import pandas as pd
import numpy as np


def signed_log1p(data: np.ndarray) -> np.ndarray:
    """Apply log1p to magnitudes while preserving sign for numeric stability."""
    return np.sign(data) * np.log1p(np.abs(data))


FEATURE_SELECTION_PERCENTILE = 25
df = pd.read_csv("train.csv")

df = df[df['RiskScore'] > 20]
df = df[df['RiskScore'] < 80]
df = df.drop("AnnualIncome", axis=1)
df = df.drop("Age", axis=1)
df = df.drop("TotalAssets", axis=1)
df = df.drop("LoanAmount", axis=1)
df = df.drop("InterestRate", axis=1)
df = df.drop('ApplicationDate', axis=1)

X = df.drop(columns='RiskScore')
y = df['RiskScore']

X_test = pd.read_csv("test.csv")

numeric_features = make_column_selector(dtype_include=np.number)
categorical_features = make_column_selector(dtype_exclude=np.number)

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("signed_log", FunctionTransformer(signed_log1p, validate=False)),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("poly_scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("binning", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("feature_select", SelectPercentile(score_func=f_regression, percentile=FEATURE_SELECTION_PERCENTILE)),
        ("regressor", LinearRegression()),
    ]
)

pipeline.fit(X, y)
preds = pipeline.predict(X_test)
print(preds)

df_preds = pd.DataFrame({
    'ID': range(0, len(preds)),
    'RiskScore': preds
})

cv = KFold(n_splits=5, shuffle=True, random_state=1488)
mse_scores = -cross_val_score(
    pipeline, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
)

mean_mse = mse_scores.mean()
std_mse = mse_scores.std()
print(f"Cross-validated MSE: {mean_mse:.4f} ± {std_mse:.4f}")
df_preds.to_csv("res.csv", index=False)
