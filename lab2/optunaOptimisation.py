import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from customRealization.CustomPipeline import CustomPipelineModel


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


def get_train_data():
    df_train = pd.read_csv("train_c.csv")

    df_train = modify_features(df_train)
    df_train_no_target = df_train.drop(columns='LoanApproved')
    df_target = df_train['LoanApproved']
    y_train = df_target.to_numpy()

    x_train = CustomPipelineModel().pipeline(df_train_no_target, fit=True)

    return x_train, y_train


X, y = get_train_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    try:
        param = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 0.1, 10),
            'alpha': trial.suggest_float('alpha', 0.1, 10)
        }

        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse
    except Exception as e:
        print(f"Error occurred during trial: {e}")
        return 1e6


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
#
print("Best parameters:", study.best_params)
