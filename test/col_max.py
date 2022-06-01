import pandas as pd


d = {"hoge": [100, 200, 300],
    "huga": [1, 2, 3]
}
df = pd.DataFrame(d)
print(df)
print(df["hoge"].max())
print(df["huga"].max())
df["hoge"] = df["hoge"] / df["hoge"].max()
print(df)