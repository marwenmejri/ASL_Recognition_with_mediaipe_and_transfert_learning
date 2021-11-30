import pandas as pd

df = pd.read_csv("coords.csv")

# print(df.columns)
print(df.shape)
#
df.drop(df.index[:], 0, inplace=True)
#
# print(df.shape)
#
# print(df)
# print(df.isnull().sum().sum())
#
df.to_csv("coords.csv")

