import argparse
import datetime
import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import xgboost as xgb

from parameters import xgb_params
from utils import create_dtrain_dtest
from utils import create_features_and_labels
from utils import evalerror
from utils import fair_objective
from utils import load_data


# Argument 'model_name' is used as stacked predictions of out-of-fold predictions
parser = argparse.ArgumentParser()
parser.add_argument('model_name', metavar='model_name', type=str)
args = parser.parse_args()
model_name = args.model_name

train_data = './data/train_preprocessed.csv'
test_data = './data/test_preprocessed.csv'

# XGBoost parameters
num_boost_round = 10000
early_stopping_rounds = 10
verbose_eval = 100
# KFold parameter
n_splits = 5

shift = 200

out_of_fold_preds_list = []
df_stacked_out_of_fold_preds = pd.DataFrame()
stacked_preds_csv = './result/stacked_preds_{}.csv'.format(model_name)

df_preds_tmp = pd.DataFrame()
temporal_preds_csv = './result/temporal_preds_xgb.csv'


if __name__ == '__main__':
    train, test = load_data(train_data, test_data)
    X_train, y_train, X_test = create_features_and_labels(train, test)

    dtest = xgb.DMatrix(X_test)

    kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    for i, (train_index, test_index) in enumerate(kf.split(X_train), start=1):
        print('Start Fold {}'.format(i))
        t0_fold = time.time()

        X_train_prime, X_test_prime = X_train[train_index], X_train[test_index]
        y_train_prime, y_test_prime = y_train[train_index], y_train[test_index]

        dtrain_prime = xgb.DMatrix(X_train_prime, label=y_train_prime)
        dtest_prime = xgb.DMatrix(X_test_prime, label=y_test_prime)

        watch_list = [(dtrain_prime, 'train_prime'), (dtest_prime, 'test_prime')]

        gbdt = \
        xgb.train(xgb_params,
                  dtrain_prime,
                  num_boost_round=num_boost_round,
                  early_stopping_rounds=early_stopping_rounds,
                  evals=watch_list,
                  verbose_eval=verbose_eval,
                  feval=evalerror,
                  obj=fair_objective,
                  )

        # Out of fold prediction
        out_of_fold_preds = \
        gbdt.predict(dtest_prime, ntree_limit=gbdt.best_ntree_limit)
        out_of_fold_preds = np.exp(out_of_fold_preds) - shift
        # Hold this prediction and save them later
        out_of_fold_preds_list.append(out_of_fold_preds)

        # temporal prediction
        preds_tmp = gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)
        preds_tmp = np.exp(preds_tmp) - shift
        df_preds_tmp['fold_{}'.format(i)] = preds_tmp

        print('End Fold {} in {} s'.format(i, time.time() - t0_fold))

    # Save the stacked out-of-fold predictions
    df_stacked_out_of_fold_preds[model_name] = np.concatenate(out_of_fold_preds_list)
    df_stacked_out_of_fold_preds\
    .to_csv(stacked_preds_csv, index=False)
    # Save the temporal predictions
    df_preds_tmp.to_csv(temporal_preds_csv)
