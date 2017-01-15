# coding: UTF-8
import argparse
import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from utils import load_data
from utils import create_features_and_labels

from neural_network import nn_4_layer
from neural_network_utils import fit_batch_generator
from neural_network_utils import predict_batch_generator


# parameters
shift = 200
# cross validation param
n_splits = 10
# NN param
batch_size = 128
nb_epoch = 50

# Argument 'model_name' is used as stacked predictions of out-of-fold predictions
parser = argparse.ArgumentParser()
parser.add_argument('model_name', metavar='model_name', type=str)
args = parser.parse_args()
model_name = args.model_name

# Data preparation
train_data = './data/train_preprocessed.csv'
test_data = './data/test_preprocessed.csv'
df_train, df_test = load_data(train_data, test_data)
X_train, y_train, X_test = create_features_and_labels(df_train, df_test)

# result holder
df_val_score = pd.DataFrame()


if __name__ == '__main__':
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    for i, (train_index, test_index) in enumerate(kf.split(X_train), start=1):
        print('Start Fold {}'.format(i))
        t0_fold = time.time()

        X_train_prime, X_val = X_train[train_index], X_train[test_index]
        y_train_prime, y_val = y_train[train_index], y_train[test_index]

        # create model
        input_dim = X_train_prime.shape[1]
        model = nn_4_layer(input_dim)

        # model training
        history = model.fit_generator(
            generator=fit_batch_generator(X_train_prime, y_train_prime, batch_size),
            nb_epoch=nb_epoch,
            samples_per_epoch=X_train_prime.shape[0],
            verbose=1)

        # evaluate validation loss
        preds = \
            np.exp(
                model.predict_generator(
                    generator=predict_batch_generator(X_val, batch_size),
                    val_samples=X_val.shape[0]).reshape(-1)
            ) - shift
        trues = np.exp(y_val) - shift
        df_val_score.loc[i, 'val_loss'] = mean_absolute_error(trues, preds)

        print('End Fold {} in {} s'.format(i, time.time() - t0_fold))

    # save the validation scores
    csv_file_name = './result/cross_validation_neural_net.csv'.format(model_name)
    df_val_score.to_csv(csv_file_name, index=False)
