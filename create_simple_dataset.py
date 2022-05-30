import gc
import re
from lib.connect_dataset import application_train_test, bureau_and_balance, previous_applications
from lib.connect_dataset import pos_cash, installments_payments, credit_card_balance
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

    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    train_df.to_csv("simple-dataset-train.csv")
    test_df.to_csv("simple-dataset-test.csv")


if __name__ == "__main__":
    main()
