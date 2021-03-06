from contextlib import contextmanager
import time
import pandas as pd


def display_categorical_features(df, thresh=10):
    """データを最初にここに通して質的変数の確認を行う"""
    unique_count = df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
    for idx in unique_count.index:
        if unique_count[idx] <= thresh:
            print(f"{idx}\t{unique_count[idx]}\t{df[idx].unique()}")
        else:
            print(f"{idx}\t{unique_count[idx]}")


@contextmanager
def timer(title):
    t0 = time.time()  # __enter__()に対応
    yield  # with timer(title)で始めた処理がreturnの代わりにyieldで値を返す
    print("{} - done in {:.0f}s".format(title, time.time() - t0))  # __exit__()に対応


def one_hot_encoder(df, nan_as_category=True):
    """質的変数をダミー変数に変換"""
    original_columns = list(df.columns)
    # dtypeがobject(i.e.str)の系列のカラム名をリストアップ
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    # pd.get_dummiesでダミー変数に変換
    # 変換するカラムにはcategorical_columnsを指定
    # pd.get_dummiesではpandas.DataFrameを指定すると[元の列名_カテゴリー名]に変換される
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
