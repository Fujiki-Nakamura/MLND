# coding; UTF-8
from keras import backend as K


def fit_batch_generator(X, y, batch_size):
    nb_samples = X.shape[0]

    while True:
        for start_i in range(0, nb_samples, batch_size):
            end_i = start_i + batch_size
            X_batch = X[start_i:end_i]
            y_batch = y[start_i:end_i]
            yield X_batch, y_batch


def predict_batch_generator(X, batch_size):
    nb_samples = X.shape[0]

    while True:
        for start_i in range(0, nb_samples, batch_size):
            end_i = start_i + batch_size
            X_batch = X[start_i:end_i]
            yield X_batch


# TODO: Perhaps something wrong
def modified_mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_pred) - K.exp(y_true)), axis=-1)
