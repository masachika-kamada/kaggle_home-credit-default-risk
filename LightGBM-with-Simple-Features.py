import numpy as np
import pandas as pd
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from lib.utils import timer, one_hot_encoder
from lib.create_data import application_train_test, bureau_and_balance

dir_ref = "./csv-data"


def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)

    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        # print("Previous applications df shape:", prev.shape)
        # df = df.join(prev, how='left', on='SK_ID_CURR')
        # del prev
        # gc.collect()
    # with timer("Process POS-CASH balance"):
    #     pos = pos_cash(num_rows)
    #     print("Pos-cash balance df shape:", pos.shape)
    #     df = df.join(pos, how='left', on='SK_ID_CURR')
    #     del pos
    #     gc.collect()
    # with timer("Process installments payments"):
    #     ins = installments_payments(num_rows)
    #     print("Installments payments df shape:", ins.shape)
    #     df = df.join(ins, how='left', on='SK_ID_CURR')
    #     del ins
    #     gc.collect()
    # with timer("Process credit card balance"):
    #     cc = credit_card_balance(num_rows)
    #     print("Credit card balance df shape:", cc.shape)
    #     df = df.join(cc, how='left', on='SK_ID_CURR')
    #     del cc
    #     gc.collect()

    # with timer("Run LightGBM with kfold"):
    #     feat_importance = kfold_lightgbm(
    #         df, num_folds=10, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission.csv"
    with timer("Full model run"):
        main()
