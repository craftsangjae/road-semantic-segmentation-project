"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf


def sparse_categorical_crossentropy_with_ignore(ignore_value=255):
    """
    Sparse Categorical CrossEntropy without specific label(ignore value)

    :param ignore_value: the label to exclude from loss calculation
    :return:
    """
    def sparse_categorical_cross_entropy(y_true, y_pred):
        y_true = Reshape((-1,))(y_true)

        num_classes = y_pred.shape[-1]
        y_pred = Reshape((-1, num_classes))(y_pred)

        ignore_mask = tf.where(tf.equal(y_true, ignore_value),
                               tf.zeros_like(y_true, tf.float32),
                               tf.ones_like(y_true, tf.float32))
        y_true = y_true * ignore_mask

        loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        loss = loss * ignore_mask
        return K.mean(loss, axis=1)
    return sparse_categorical_cross_entropy


def binary_crossentropy_with_ignore(ignore_value=255):
    """
    Binary CrossEntropy without specific label(ignore value)

    :param ignore_value: the label to exclude from loss calculation
    :return:
    """
    def binary_cross_entropy(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, 1))
        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.reshape(y_pred, (-1, 1))

        ignore_mask = tf.where(tf.equal(y_true, ignore_value),
                               tf.zeros_like(y_true, tf.float32),
                               tf.ones_like(y_true, tf.float32))
        y_true = y_true * ignore_mask

        loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        loss = loss * ignore_mask
        return K.mean(loss, axis=1)
    return binary_cross_entropy


get_custom_objects().update({
    "sparse_categorical_cross_entropy":sparse_categorical_crossentropy_with_ignore(ignore_value=255),
    "binary_cross_entropy":binary_crossentropy_with_ignore(ignore_value=255)
})