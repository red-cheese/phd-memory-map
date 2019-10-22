"""
Sample from 2 Gaussians and classify them, track mmap.
"""

import numpy as np
from numpy.random import multivariate_normal

from ml import dense


INPUT_DIM = 256
BATCH_SIZE = 64

NUM_TRAIN_BATCHES = 80
NUM_TEST_BATCHES = 20


def _generate_data(mu, sigma, cls, n):
    x = multivariate_normal(mu, sigma, size=n)
    y = np.zeros((n, 2), dtype=np.int32)
    y[:, cls] = 1
    return x, y


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


def gaussian0():
    """
    Experiment 0: no poisoning, static IID data.
    """

    print('Experiment 0')

    train_set_size = (NUM_TRAIN_BATCHES // 2) * BATCH_SIZE
    test_set_size = (NUM_TEST_BATCHES // 2) * BATCH_SIZE

    mu_0 = np.full((INPUT_DIM,), 5., dtype=np.float64)
    sigma_0 = np.eye(INPUT_DIM, dtype=np.float64) * 0.1
    train_x_0, train_y_0 = _generate_data(mu_0, sigma_0, 0, train_set_size)
    test_x_0, test_y_0 = _generate_data(mu_0, sigma_0, 0, test_set_size)

    mu_1 = np.full((INPUT_DIM,), 2., dtype=np.float64)
    sigma_1 = np.eye(INPUT_DIM, dtype=np.float64) * 1.
    train_x_1, train_y_1 = _generate_data(mu_1, sigma_1, 1, train_set_size)
    test_x_1, test_y_1 = _generate_data(mu_1, sigma_1, 1, test_set_size)

    train_x, train_y = _vstack([(train_x_0, train_y_0),
                                (train_x_1, train_y_1)],
                               shuffle=True)
    test_x, test_y = _vstack([(test_x_0, test_y_0),
                              (test_x_1, test_y_1)],
                             shuffle=True)

    dense_nn = dense.DenseNN(name='gaussian0',
                             input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                             classes=[0, 1], batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y))

    print('=====')
    print('Evaluate against test data:')
    print(dense_nn.evaluate(test_x, test_y))

    print()
    print('Experiment 0 complete')


def main():
    gaussian0()


if __name__ == '__main__':
    main()
