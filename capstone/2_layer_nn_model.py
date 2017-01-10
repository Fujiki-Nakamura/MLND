import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense


model_name = './models/2_layer_benchmark'
model_weights_name = model_name + '_weights'
history_path = model_name + '_history.csv'

batch_size = 128
nb_epochs = 10

# load the data
train = pd.read_csv('./data/train_preprocessed.csv')
test = pd.read_csv('./data/test_preprocessed.csv')
X_train_origin = train.drop(['id', 'loss'], axis=1).values
y_train_origin = train['loss'].values
X_train, y_train, X_val, y_val = \
train_test_split(X_train_origin, y_train_origin, test_size=0.25, random_state=0)

input_dim = X_train.shape[1]


def get_model():
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    return model


def main():
    model = get_model()
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size, nb_epoch=nb_epochs,
        validation_data=(X_val, y_val), verbose=1)
    # Debug
    for l in model.layers:
        print(l.name, l.input_shape, l.output_shape, l.activation)
    # Save the model
    model.save(model_name)
    model.save_weights(model_weights_name)

    # Save the model's training history
    df_history = pd.DataFrame(history.history)
    df_history.to_csv(history_path, index=False)


if __name__ == '__main__':
    main()

