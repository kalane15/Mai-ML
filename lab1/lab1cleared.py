import numpy as np

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

from customRealization.CustomSimpleImputer import CustomSimpleImputer
from customRealization.CustomFunctionTransformer import CustomFunctionTransformer
from customRealization.CustomRobustScaler import CustomRobustScaler
from customRealization.CustomPolynomialFeatures import CustomPolynomialFeatures
from customRealization.CustomOneHotEncoder import CustomOneHotEncoder
from customRealization.CustomLinearRegression import CustomLinearRegression
from customRealization.CustomSelectPercentile import CustomSelectPercentile


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

    df = make_bins(df, 'CreditScore')
    return df


def signed_log1p(data: np.ndarray) -> np.ndarray:
    return np.sign(data) * (np.log1p(np.abs(data)))


encoder = CustomOneHotEncoder()


def pipeline(df):
    numeric_features = df.select_dtypes(include=[np.number]).to_numpy()
    categorical_features = df.select_dtypes(exclude=[np.number]).to_numpy()

    numeric_features = CustomSimpleImputer(strategy='median').fit_transform(numeric_features)
    numeric_features = CustomFunctionTransformer(signed_log1p).fit_transform(numeric_features)
    numeric_features = CustomRobustScaler().fit_transform(numeric_features)
    numeric_features = CustomPolynomialFeatures(degree=3).fit_transform(numeric_features)

    categorical_features = CustomSimpleImputer(strategy="most_frequent").fit_transform(categorical_features)

    if encoder.categories_ is None:
        categorical_features = encoder.fit_transform(categorical_features)
    else:
        categorical_features = encoder.transform(categorical_features)
    combined = np.hstack((numeric_features, categorical_features))

    return combined


df = pd.read_csv("train.csv")

df = df[df['RiskScore'] > 0]
df = df[df['RiskScore'] < 100]

df = modify_features(df)
df_no_target = df.drop(columns='RiskScore')

X = pipeline(df_no_target)
y = df['RiskScore'].to_numpy()
X = CustomSelectPercentile().fit_transform(X, y)

cv = KFold(n_splits=15, shuffle=True, random_state=188)
mse_scores = -cross_val_score(
    CustomLinearRegression(), X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
)
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()
print(f"Cross-validated MSE: {mean_mse:.4f} ± {std_mse:.4f} = {mean_mse - std_mse}")
