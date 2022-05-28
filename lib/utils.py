from contextlib import contextmanager
import time
import pandas as pd


@contextmanager
def timer(title):
    t0 = time.time()  # __enter__()に対応
    yield  # with timer(title)で始めた処理がreturnの代わりにyieldで値を返す
    print("{} - done in {:.0f}s".format(title, time.time() - t0))  # __exit__()に対応


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
