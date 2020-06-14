"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras import backend as K
import tensorflow as tf

def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values (sparse output)
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred, axis=(1,2))
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true,axis=(1,2)) + K.sum(y_pred, axis=(1,2)) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return tf.where(union>0.,intersection / union, tf.ones_like(union))


def build_iou_for(label: int, name: str = None):
    """
    Build an Intersection over Union (IoU) metric for a label.
    Args:
        label: the label to build the IoU metric for
        name: an optional name for debugging the built method
    Returns:
        a keras_code metric to evaluate IoU for the given label

    Note:
        label and name support list inputs for multiple labels
    """
    # handle recursive inputs (e.g. a list of labels and names)
    if isinstance(label, list):
        if isinstance(name, list):
            return [build_iou_for(l, n) for (l, n) in zip(label, name)]
        return [build_iou_for(l) for l in label]

    # build the method for returning the IoU of the given label
    def label_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score for {0}.
        Args:
            y_true: the expected y values (sparse output)
            y_pred: the predicted y values as a one-hot or softmax output
        Returns:
            the scalar IoU value for the given label ({0})
        """.format(label)
        return iou(y_true, y_pred, label)

    # if no name is provided, us the label
    if name is None:
        name = label
    # change the name of the method for debugging
    label_iou.__name__ = 'iou_{}'.format(name)

    return label_iou


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values (sparse output)
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # iterate over labels to calculate IoU for
    iou_list = [iou(y_true, y_pred, label) for label in range(num_labels)]
    total_iou = tf.squeeze(tf.add_n(iou_list),axis=-1)

    # divide total IoU by number of labels to get mean IoU
    mean_iou =  total_iou / num_labels
    return K.mean(mean_iou)


def binary_iou(thr=0.5):
    """
    Build an Intersection over Union (IoU) metric for binary label.
    Args:
        thr : the threshold for binary classification
    Returns:
        a keras_code metric to evaluate IoU for the given label

    Note:
        label and name support list inputs for multiple labels
    """
    def _iou(y_true, y_pred):
        # extract the label values using the argmax operator then
        # calculate equality of the predictions and truths to the label
        y_true = K.cast(y_true > thr, K.floatx())
        y_pred = K.cast(y_pred > thr, K.floatx())
        y_true = K.squeeze(y_true, axis=-1)
        y_pred = K.squeeze(y_pred, axis=-1)
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred, axis=(1, 2))
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true, axis=(1, 2)) + K.sum(y_pred, axis=(1, 2)) - intersection
        # avoid divide by zero - if the union is zero, return 1
        # otherwise, return the intersection over union
        iou = tf.where(union > 0., intersection / union, tf.ones_like(union))
        return K.mean(iou)
    return _iou
