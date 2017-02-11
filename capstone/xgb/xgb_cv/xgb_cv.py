# coding: UTF-8
"""Hyper parameter tuning with 5-Fold Cross Validation"""
import argparse
import datetime
import os
import sys
import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import xgboost as xgb

sys.path.append('../')
from parameters import xgb_params
from utils import create_dtrain_dtest
from utils import create_features_and_labels
from utils import evalerror
from utils import fair_objective
from utils import load_data


# Data preparation
data_dir = '../../data/'
train_data = data_dir + 'train_preprocessed.csv'
test_data = data_dir + 'test_preprocessed.csv'
train, test = load_data(train_data, test_data)
X_train, y_train, X_test = create_features_and_labels(train, test)

# parameters for XGB Cross Validation
num_boost_round = 10000
early_stopping_rounds = 10
verbose_eval = 100
nfold = 5

shift = 200

# prepare a directory to save results
result_directory = './result/'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)


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
    cv.to_csv(result_directory + 'cross_validation.csv')
    print('End Cross Validation')

    cv_mean = cv.iloc[-1, 0]
    cv_std = cv.iloc[-1, 1]
    print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

    best_num_rounds = cv.shape[0]
    print('Best # of rounds', best_num_rounds)

    dt1 = datetime.datetime.today()
    print('End at ', dt1)
    print(dt1 - dt0)
