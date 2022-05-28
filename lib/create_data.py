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
        # pd.factorizeは質的変数を数値に変換してくれる
        # CODE_GENDER: M/F, FLAG_OWN_CAR: Y/N, FLAG_REALITY: Y/Nなので
        # uniquesには変換後の数値が変換前に何だったかを表す質的変数が入っている
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    """3値以上の質的変数をダミー変数化"""
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    """DAYS_EMPLOYEDで365.243はNaNを示す"""
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    """新たな特徴量の生成"""
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    """メモリの解放"""
    # delは名前空間から変数を削除するが必ずしもメモリから変数をクリアするわけではない
    # delステートメントを使用して変数を削除した後、
    # gc.collect()メソッドを使用して変数をメモリからクリアできる
    del test_df
    gc.collect()
    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    """bureau.csvとbureau_balance.csvを加工"""

    """まずデータを読み込む"""
    bureau = pd.read_csv(dir_ref + '/bureau.csv', nrows=num_rows)
    bb = pd.read_csv(dir_ref + '/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    """bureauに関するデータを結合する"""
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                              for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(
        ['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(
        ['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(
        ['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg
