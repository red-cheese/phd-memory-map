"""
Taken from https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model.
"""


from keras import backend as K


def _precision(y_true_cat, y_pred_cat):
    true_positives = K.sum(K.round(K.clip(y_true_cat * y_pred_cat, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_cat, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def _recall(y_true_cat, y_pred_cat):
    true_positives = K.sum(K.round(K.clip(y_true_cat * y_pred_cat, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_cat, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def binary_recall(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=1), K.floatx())
    y_pred = K.cast(K.argmax(y_pred, axis=1), K.floatx())
    return _recall(y_true, y_pred)


def binary_precision(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=1), K.floatx())
    y_pred = K.cast(K.argmax(y_pred, axis=1), K.floatx())
    return _precision(y_true, y_pred)


def binary_f1(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=1), K.floatx())
    y_pred = K.cast(K.argmax(y_pred, axis=1), K.floatx())

    precision = _precision(y_true, y_pred)
    recall = _recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
