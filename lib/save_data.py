import re


def df2csv(df):
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    print(df.columns)

    df.drop(columns=['index'], inplace=True)

    train_df = df[df['TARGET'].notnull()]
    train_df.drop(columns=['SK_ID_CURR'], inplace=True)

    test_df = df[df['TARGET'].isnull()]
    SK_ID_CURR = test_df['SK_ID_CURR']
    test_df.drop(columns=['SK_ID_CURR', 'TARGET'], inplace=True)

    print(train_df['TARGET'].dtypes)
    train_df['TARGET'] = train_df['TARGET'].astype(int)
    print(train_df['TARGET'].dtypes)

    train_df.to_csv("test-train.csv", index=False)
    SK_ID_CURR.to_csv("SK_ID_CURR.csv", index=False)
    test_df.to_csv("test-test.csv", index=False)
