import datetime
import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

import xgboost as xgb

from parameters import xgb_params


RANDOM_STATE = 2016
shift = 0


def create_train_test_joined():
    train = pd.read_csv('./data/train_lexicographical.csv')
    test = pd.read_csv('./data/test_lexicographical.csv')
    test['loss'] = np.nan

    return train, test


def fair_objective(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con**2 / (np.abs(x) + con)**2
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


def create_dtrain_dtest(train, test):
    y = np.log(train['loss'] + shift)
    X = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['loss', 'id'], 1)

    dtrain = xgb.DMatrix(X, label=y)
    dtest = xgb.DMatrix(X_test)

    return dtrain, dtest


if __name__ == '__main__':
    dt0 = datetime.datetime.today()
    print('Start at ', dt0)

    train, test = create_train_test_joined()
    dtrain, dtest = create_dtrain_dtest(train, test)

    print('Start Cross Validation')
    t0 = time.time()
    cv = \
    xgb.cv(xgb_params,
           dtrain,
           num_boost_round=10000,
           nfold=5,
           stratified=False,
           early_stopping_rounds=100,
           verbose_eval=100,
           show_stdv=True,
           feval=evalerror,
           maximize=False,
           obj=fair_objective)
    print('CV in ', time.time() - t0)
    cv.to_csv('./CSV/xgb_CV_result.csv')
    print('End Cross Validation')

    cv_mean = cv.iloc[-1, 0]
    cv_std = cv.iloc[-1, 1]
    print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

    best_num_rounds = cv.shape[0]
    print('Best # of rounds', best_num_rounds)

    dt1 = datetime.datetime.today()
    print('End at ', dt1)
    print(dt1 - dt0)
