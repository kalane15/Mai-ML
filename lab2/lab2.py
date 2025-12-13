import numpy as np
import pandas as pd
from customRealization.CustomPipeline import CustomPipelineModel
import lightgbm as lgb

from customRealization.CustomSelectPercentile import CustomSelectPercentile


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


pipe = CustomPipelineModel()
selector = CustomSelectPercentile(percentile=25)


def get_train_data():
    df_train = pd.read_csv("train_c.csv")

    df_train = modify_features(df_train)
    df_train_no_target = df_train.drop(columns='LoanApproved')
    df_target = df_train['LoanApproved']
    y_train = df_target.to_numpy()

    x_train = pipe.pipeline(df_train_no_target, fit=True)
    x_train = selector.fit_transform(x_train, y_train)

    return x_train, y_train


def get_test_data():
    df = pd.read_csv("test_c.csv")
    df = df.drop(columns="ID")
    df = modify_features(df)
    df = pipe.pipeline(df, fit=True)

    df = selector.transform(df)
    return df


X, y = get_train_data()
params = {
    'max_depth': 7,
    'learning_rate': 0.07581910324041137,
    'n_estimators': 50,
    'subsample': 0.9126331830833376,
    'alpha': 5.753379354378411
}
X_test = get_test_data()
model = lgb.LGBMClassifier(**params)

model.fit(X, y)

test_predictions = model.predict(X_test)
print("Test predictions:", test_predictions)

df_preds = pd.DataFrame({
    'ID': range(0, len(test_predictions)),
    'LoanApproved': test_predictions
})
df_preds.to_csv("res.csv", index=False)
