import pandas as pd
import numpy as np
from lib.utils import one_hot_encoder
import gc


dir_ref = "./csv-data"


def application_train_test(num_rows=None, nan_as_category=False):
    """データの読み込みと結合"""
    # pd.read_csvでnrowsは最初の数行だけを読み込む場合に指定
    df = pd.read_csv(dir_ref + '/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv(dir_ref + '/application_test.csv', nrows=num_rows)
    print(f"Train samples: {len(df)}, test samples: {len(test_df)}")
    # concatによりtest_dfがTARGETがNaNでdfの下に結合される
    df = pd.concat([df, test_df]).reset_index()
    print(f"Combined samples: {len(df)}")

    """trainデータ内でCODE_GENDERがXNAになっているのが4例あるので除去"""
    # 4例の除去に気付くのは難しそう。実行時エラーから気付くのが普通？
    # あるいは各データの値をユニークで求めて気付くのか
    df = df[df['CODE_GENDER'] != 'XNA']
    print(f"Remove XNA: {len(df)}")

    """2値の質的特徴を変換"""
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df
