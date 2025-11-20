from MyLinearModel import *
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

print(df.shape)

df = DataProcessor.apply_one_hot_encoding_all(df)
print(df.shape)

x, y = DataProcessor.df_to_matrix_numeric(df, 'RiskScore')
model = MyLinearModel(way=ModelLearnWay.SGD)


model.fit(x, y)

df_test = pd.read_csv("test.csv")

x = DataProcessor.apply_one_hot_encoding_all(df_test)

x = DataProcessor.df_to_matrix_numeric(x)

model.predict(x)
