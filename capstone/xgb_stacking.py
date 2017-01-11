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


train_data = './data/train_preprocessed.csv'
test_data = './data/test_preprocessed.csv'

# XGBoost parameters
num_boost_round = 577
#early_stopping_rounds = 10 early_stopping_rounds parameter causes an error in train
#verbose_eval = 100
# KFold parameter
n_splits = 5

shift = 200

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
        dtest_prime = xgb.DMatrix(X_test_prime)

        gbdt = \
        xgb.train(xgb_params,
                  dtrain_prime,
                  num_boost_round=num_boost_round,
                  #early_stopping_rounds=early_stopping_rounds,
                  #verbose_eval=verbose_eval,
                  feval=evalerror,
                  obj=fair_objective,
                  )

        # Out of fold prediction
        out_of_fold_preds = np.exp(gbdt.predict(dtest_prime)) - shift
        # Save the out-of-fold prediction
        with open('./result/preds_out_of_fold_{}.npy'.format(i), 'wb') as f:
            np.save(f, out_of_fold_preds)

        # temporal prediction
        preds_tmp = np.exp(gbdt.predict(dtest)) - shift
        df_preds_tmp['fold_{}'.format(i)] = preds_tmp

        print('End Fold {} in {} s'.format(i, time.time() - t0_fold))
        import pdb; pdb.set_trace()

    # Save the temporal predictions
    df_preds_tmp.to_csv(temporal_preds_csv)
