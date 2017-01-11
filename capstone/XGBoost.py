import datetime
import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

import xgboost as xgb

from parameters import xgb_params
from utils import load_data
from utils import create_dtrain_dtest
from utils import evalerror
from utils import fair_objective


train_data = './data/train_preprocessed.csv'
test_data = './data/test_preprocessed.csv'

RANDOM_STATE = 0
shift = 200

# parameters for XGB Cross Validation
num_boost_round = 10000
early_stopping_rounds = 10
verbose_eval = 100
nfold = 5


if __name__ == '__main__':
    dt0 = datetime.datetime.today()
    print('Start at ', dt0)

    train, test = load_data(train_data, test_data)
    dtrain, dtest = create_dtrain_dtest(train, test)

    print('Start Cross Validation')
    t0 = time.time()
    cv = \
    xgb.cv(xgb_params,
           dtrain,
           num_boost_round=num_boost_round,
           nfold=nfold,
           stratified=False,
           early_stopping_rounds=early_stopping_rounds,
           verbose_eval=verbose_eval,
           show_stdv=True,
           feval=evalerror,
           maximize=False,
           obj=fair_objective)
    print('CV in ', time.time() - t0)
    cv.to_csv('./result/xgb_CV_result.csv')
    print('End Cross Validation')

    cv_mean = cv.iloc[-1, 0]
    cv_std = cv.iloc[-1, 1]
    print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

    best_num_rounds = cv.shape[0]
    print('Best # of rounds', best_num_rounds)

    dt1 = datetime.datetime.today()
    print('End at ', dt1)
    print(dt1 - dt0)


    # Train for test
    print('Start training')
    t0 = time.time()
    gbdt = \
    xgb.train(xgb_params,
              dtrain,
              num_boost_round=num_boost_round,
              early_stopping_rounds=early_stopping_rounds,
              feval=evalerror,
              obj=fair_objective,
              )
    print('Trained in ', time.time() - t0)

    # Prediction for test data
    prediction = np.exp(gbdt.predict(dtest)) - shift

    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['loss'] = prediction
    submission.to_csv('./result/submission_xgb.csv', index=False)
