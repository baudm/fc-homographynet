#!/usr/bin/env python

from keras import backend as K
from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective


def mean_corner_error(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, 4, 2))
    y_pred = K.reshape(y_pred, (-1, 4, 2))
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True)), axis=1)


def image_consistency(y_true, y_pred):
    dssim = DSSIMObjective()
    return mean_squared_error(y_true, y_pred) + dssim(y_true, y_pred)
