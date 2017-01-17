# coding: UTF-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization


def create_model(input_dim):
    model = Sequential()

    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))

    # output layer
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')

    return model
