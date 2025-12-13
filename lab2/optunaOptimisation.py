import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import TargetEncoder

import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from customRealization.CustomPipeline import CustomPipelineModel
from customRealization.CustomGradientBoosting import CustomGradientBoosting
from sklearn.metrics import roc_auc_score

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
    df = pipe.pipeline(df)

    df = selector.transform(df)
    return df


X, y = get_train_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=148)


def objective(trial):
    try:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'random_state': 148,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        m = roc_auc_score(y_test, y_pred_proba)
        return m
    except Exception as e:
        print(f"Error occurred during trial: {e}")
        return 1e6


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
