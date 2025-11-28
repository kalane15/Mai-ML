from sklearn.model_selection import KFold, cross_val_score

from MyLinearModel import *
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
# df = df.select_dtypes(include=['number'])

df = df[df['RiskScore'] > 20]
df = df[df['RiskScore'] < 80]
df = df.drop("AnnualIncome", axis=1)
df = df.drop("Age", axis=1)
df = df.drop("TotalAssets", axis=1)
df = df.drop("LoanAmount", axis=1)
df = df.drop("InterestRate", axis=1)
df = df.drop('ApplicationDate', axis=1)
df, encode_data = DataProcessor.target_encoding_inplace(df, 'RiskScore')

df = DataProcessor.replace_nans_with_mean_in_all_columns(df)
y = df['RiskScore']
df_without_risk = df.drop(columns=['RiskScore'])
df_normalized = DataProcessor.normalize_z_score(df_without_risk, df_without_risk.mean(), df_without_risk.std())
df_normalized['RiskScore'] = y

x, y = DataProcessor.df_to_matrix_numeric(df_normalized, 'RiskScore')

model = MyLinearModel(iters=20000)
model.fit(x, y)
print(MyCrossValidation.k_fold_cross_validation(model, x, y, 5))


df_test = pd.read_csv("test.csv")
df_test = df_test.drop('ID', axis=1)
df_test, _ = DataProcessor.target_encoding_inplace(df_test, 'RiskScore', encode_data)
df_test = DataProcessor.reshape_to(df.loc[:, df.columns != 'RiskScore'], df_test)

df_test = DataProcessor.normalize_z_score(df_test, df_without_risk.mean(), df_without_risk.std())

print(model.w)
x_test, _ = DataProcessor.df_to_matrix_numeric(df_test)
preds = model.predict(x_test)
df_preds = pd.DataFrame({
    'ID': range(0, len(preds)),
    'RiskScore': preds
})


mean_mse = mse_scores.mean()
std_mse = mse_scores.std()
print(f"Cross-validated MSE: {mean_mse:.4f} ± {std_mse:.4f}")

df_preds.to_csv('res.csv', index=False)
