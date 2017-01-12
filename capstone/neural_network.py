# coding: UTF-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

from neural_network_utils import modified_mean_absolute_error


def create_model(input_dim):
    model = Sequential()

    model.add(Dense(256, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(1))

    # TODO: want loss to be modified_mean_absolute_error
    model.compile(loss='mae', optimizer='adam')

    return model
