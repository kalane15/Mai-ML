from MyLinearModel import *
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df = df.select_dtypes(include=['number'])
df = (df - df.mean()) / df.std()

# df = DataProcessor.apply_one_hot_encoding_all(df)
df = DataProcessor.replace_nans_with_mean_in_all_columns(df)
df = df.astype(int)
print(df[df['RiskScore'] > 1000].shape)