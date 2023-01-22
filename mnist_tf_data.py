

import tensorflow as tf
mnist = tf.keras.datasets.mnist


def get_class_data(label, train=True):
    print('Get data for class', label)

    # The data is cached in ~/.keras/datasets.
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if train:
        print('Full training set size:', len(y_train))
        X_train = X_train[y_train == label, :]
        print('Class', label, 'training set size:', X_train.shape[0])
    else:
        print('Full testing set size:', len(y_test))
        X_test = X_test[y_test == label, :]
        print('Class', label, 'testing set size', X_test.shape[0])

    return X_train if train else X_test
