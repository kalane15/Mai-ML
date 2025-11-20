from MyLinearModel import *
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

df = DataProcessor.apply_one_hot_encoding_all(df)
df = DataProcessor.replace_nans_with_mean_in_all_columns(df)
df = df.astype(int)

x, y = DataProcessor.df_to_matrix_numeric(df, 'RiskScore')
model = MyLinearModel()

model.fit(x, y)

df_test = pd.read_csv("test.csv")

df_test = DataProcessor.apply_one_hot_encoding_all(df_test)

df_test = DataProcessor.add_missing_columns(df.loc[:, df.columns != 'RiskScore'], df_test)
x, _ = DataProcessor.df_to_matrix_numeric(df_test)

print(model.w)

preds = model.predict(x)
print(preds)
