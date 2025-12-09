import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from customRealization.CustomPipeline import CustomPipelineModel


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


def get_train_data():
    df_train = pd.read_csv("train_c.csv")

    df_train = modify_features(df_train)
    df_train_no_target = df_train.drop(columns='LoanApproved')
    df_target = df_train['LoanApproved']
    y_train = TargetEncoder(df_target)

    x_train = CustomPipelineModel().pipeline(df_train_no_target)

    return x_train, y_train
