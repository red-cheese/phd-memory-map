

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from mnist17.constants import *


def _vstack(seq, shuffle=True):
    x = np.array([], dtype=seq[0][0].dtype).reshape((0, INPUT_DIM))
    y = np.array([], dtype=seq[0][1].dtype).reshape((0, 2))
    for _xs, _ys in seq:
        x = np.vstack((x, _xs))
        y = np.vstack((y, _ys))

    if shuffle:
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx, :]
        y = y[idx, :]

    return x, y


def basic_train_test(label0=1, label1=7, enforce_50_50=True):
    (mn_x_train, mn_y_train), (mn_x_test, mn_y_test) = mnist.load_data()
    mn_x_train = mn_x_train.reshape((-1, mn_x_train.shape[-1] * mn_x_train.shape[-2]))
    mn_x_test = mn_x_test.reshape((-1, mn_x_test.shape[-1] * mn_x_test.shape[-2]))

    idx0 = mn_y_train == label0
    idx0_test = mn_y_test == label0
    train_x_0, train_y_0 = mn_x_train[idx0, :], mn_y_train[idx0]
    test_x_0, test_y_0 = mn_x_test[idx0_test, :], mn_y_test[idx0_test]

    idx1 = mn_y_train == label1
    idx1_test = mn_y_test == label1
    train_x_1, train_y_1 = mn_x_train[idx1, :], mn_y_train[idx1]
    test_x_1, test_y_1 = mn_x_test[idx1_test, :], mn_y_test[idx1_test]

    train_y_0 *= 0
    test_y_0 *= 0
    train_y_1 = train_y_1 * 0 + 1
    test_y_1 = test_y_1 * 0 + 1
    train_y_0 = to_categorical(train_y_0, num_classes=2)
    test_y_0 = to_categorical(test_y_0, num_classes=2)
    train_y_1 = to_categorical(train_y_1, num_classes=2)
    test_y_1 = to_categorical(test_y_1, num_classes=2)

    train_set_size = (NUM_TRAIN_BATCHES // 2) * BATCH_SIZE  # Per class.
    test_set_size = (NUM_TEST_BATCHES // 2) * BATCH_SIZE  # Per class.

    train_x_0, train_y_0 = train_x_0[:train_set_size], train_y_0[:train_set_size]
    test_x_0, test_y_0 = test_x_0[:test_set_size], test_y_0[:test_set_size]
    train_x_1, train_y_1 = train_x_1[:train_set_size], train_y_1[:train_set_size]
    test_x_1, test_y_1 = test_x_1[:test_set_size], test_y_1[:test_set_size]

    if not enforce_50_50:
        train_x, train_y = _vstack([(train_x_0, train_y_0),
                                    (train_x_1, train_y_1)],
                                   shuffle=True)
    else:
        print('Arranging data to enforce 50-50 split in each training batch')
        batches = []
        for i in range(NUM_TRAIN_BATCHES):
            start_idx = i * (BATCH_SIZE // 2)
            end_idx = (i + 1) * (BATCH_SIZE // 2)
            # It's a tuple.
            batch = _vstack([(train_x_0[start_idx:end_idx], train_y_0[start_idx:end_idx]),
                             (train_x_1[start_idx:end_idx], train_y_1[start_idx:end_idx])],
                            shuffle=True)
            batches.append(batch)
        train_x, train_y = _vstack(batches, shuffle=False)

    assert train_set_size == len(train_x) // 2

    test_x, test_y = _vstack([(test_x_0, test_y_0),
                              (test_x_1, test_y_1)],
                             shuffle=True)

    # Helper array to track removed batches in the mmap-based defense.
    train_batch_id = np.zeros((NUM_TRAIN_BATCHES * BATCH_SIZE,), dtype=np.int32)
    for i in range(NUM_TRAIN_BATCHES):
        train_batch_id[i * BATCH_SIZE:((i + 1) * BATCH_SIZE)] = i

    return train_x / 255, train_y, test_x / 255, test_y, train_batch_id