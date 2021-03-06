import gc
import re
from lib.connect_dataset import application_train_test, bureau_and_balance, previous_applications
from lib.connect_dataset import pos_cash, installments_payments, credit_card_balance
from lib.models import kfold_lightgbm, display_importances
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dir_ref = "./csv-data"


def main():
    df = application_train_test(None)

    bureau = bureau_and_balance(None)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()

    prev = previous_applications(None)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()

    pos = pos_cash(None)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()

    ins = installments_payments(None)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()

    cc = credit_card_balance(None)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()

    # df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # train_df = df[df['TARGET'].notnull()]
    # test_df = df[df['TARGET'].isnull()]

    # print(train_df['TARGET'].dtypes)
    # train_df['TARGET'] = train_df['TARGET'].astype(int)
    # print(train_df['TARGET'].dtypes)

    # train_df.to_csv("simple-dataset-train.csv", index=False)
    # test_df.to_csv("simple-dataset-test.csv", index=False)

    feature_importance = kfold_lightgbm(df, num_folds=10, submission_file_name=submission_file_name, stratified=False)
    display_importances(feature_importance)


if __name__ == "__main__":
    submission_file_name = "./submission/rule-based-features.csv"
    main()
