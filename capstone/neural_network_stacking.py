# coding: UTF-8
import argparse
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from utils import load_data
from utils import create_features_and_labels

from neural_network import create_model
from neural_network_utils import fit_batch_generator
from neural_network_utils import predict_batch_generator


shift = 200

n_splits = 5

# NN params
batch_size = 128
nb_epoch = 10

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

df_stacked_out_of_fold_preds = pd.DataFrame()
df_stacked_out_of_fold_preds['id'] = df_train['id']
stacked_preds_csv = './result/stacked_preds_{}.csv'.format(model_name)

df_temporal_preds = pd.DataFrame()
df_temporal_preds['id'] = df_test['id']
temporal_preds_csv = './result/temporal_preds_{}.csv'.format(model_name)

# placeholder
out_of_fold_preds = np.zeros(X_train.shape[0])
temporal_preds = np.zeros(X_test.shape[0])


if __name__ == '__main__':

    kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    # Stacking
    for i, (train_index, test_index) in enumerate(kf.split(X_train), start=1):
        print('Start Fold {}'.format(i))
        t0_fold = time.time()

        X_train_prime, X_test_prime = X_train[train_index], X_train[test_index]
        y_train_prime, y_test_prime = y_train[train_index], y_train[test_index]

        # create model
        input_dim = X_train_prime.shape[1]
        model = create_model(input_dim)

        # model training
        history = model.fit_generator(
            generator=fit_batch_generator(X_train_prime, y_train_prime, batch_size),
            nb_epoch=nb_epoch,
            samples_per_epoch=X_train_prime.shape[0],
            verbose=1,
            validation_data=(X_test_prime, y_test_prime))

        # Save the history
        df_history = pd.DataFrame(history.history)
        df_history.to_csv('./result/{}_fold_{}_history.csv'.format(model_name, str(i)), index=False)

        # Out of fold prediction
        out_of_fold_preds[test_index] = \
            np.exp(
                model.predict_generator(
                    generator=predict_batch_generator(X_test_prime, batch_size),
                    val_samples=X_test_prime.shape[0]).reshape(-1)
            ) - shift

        # temporal prediction
        temporal_preds = \
            np.exp(
                model.predict_generator(
                    generator=predict_batch_generator(X_test, batch_size),
                    val_samples=X_test.shape[0]).reshape(-1)
            ) - shift
        df_temporal_preds['fold_{}'.format(i)] = temporal_preds

        print('End Fold {} in {} s'.format(i, time.time() - t0_fold))

    # Save the stacked out-of-fold predictions
    df_stacked_out_of_fold_preds[model_name] = out_of_fold_preds
    df_stacked_out_of_fold_preds.to_csv(stacked_preds_csv, index=False)
    # Save the temporal predictions
    df_temporal_preds.to_csv(temporal_preds_csv)

    # Save the model architecture
    with open('./models/{}.json'.format(model_name), 'w') as f:
        f.write(model.to_json())
    # Save the model weights
    model.save_weights('./models/{}_weights.h5'.format(model_name))
