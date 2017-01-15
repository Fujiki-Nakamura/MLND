# coding: UTF-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

from neural_network_utils import modified_mean_absolute_error


def create_model(input_dim):
    return nn_4_layer(input_dim)


def nn_2_layer(input_dim):
    model = Sequential()

    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')

    return model


def nn_4_layer(input_dim):
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

    model.compile(loss='mae', optimizer='adam')

    return model


def keras_starter(input_dim):
    # NOTE: model from
    # https://www.kaggle.com/mtinti/allstate-claims-severity/keras-starter-with-bagging-1111-84364
    model = Sequential()

    model.add(Dense(400, input_dim=input_dim, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(50, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1, init='he_normal'))

    # TODO: want loss to be modified_mean_absolute_error
    model.compile(loss='mae', optimizer='adam')

    return model
