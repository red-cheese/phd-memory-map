"""
Sample from 2 Gaussians and classify them, track mmap.
"""


import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from numpy.random import multivariate_normal

from ml import dense


INPUT_DIM = 256
BATCH_SIZE = 64
NUM_EPOCHS = 5

NUM_TRAIN_BATCHES = 80
NUM_TEST_BATCHES = 20

CYCOL = cycle('bgrcmk')  # For colours.


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


def _basic_train_test():
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

    return train_x, train_y, test_x, test_y


def _flip_labels(orig_labels, start_idx, end_idx, flip_proba):
    print('Flip labels in the interval [{}, {}) (batches {}-{}) with probability {}'
          .format(start_idx, end_idx, start_idx // BATCH_SIZE, end_idx // BATCH_SIZE, flip_proba))

    labels = np.copy(orig_labels)
    idx = np.random.choice(np.arange(start_idx, end_idx),
                           size=int(flip_proba * (end_idx - start_idx)),
                           replace=False)
    old_labels = labels[idx, :]
    labels[idx, :] = 1 - old_labels

    return labels


def _remove_batches(xs, ys, batches_to_remove):
    batches_to_remove_set = set(batches_to_remove)
    to_keep = [i for i in range(xs.shape[0])
               if i // BATCH_SIZE not in batches_to_remove_set]
    return xs[to_keep], ys[to_keep]


def _mmap_dist(ms1, ms2):
    """Returns distances between per-epoch memory maps."""
    return [np.linalg.norm(m1 - m2) for m1, m2 in zip(ms1, ms2)]


def _find_bad_batch(mmap):
    """
    Tries to find a bad (poisoned) batch of data by looking at the mmap.

    Computes average losses (mmap values) for each batch (across all training
    steps), then computed mean and std of these losses, and flags batches for
    which the losses are beyond 3 sigmas.

    This is a very crude way of finding "bad" batches, and it's temporary.
    """

    avg_losses = np.mean(mmap, axis=0)
    mean_, std_ = np.mean(avg_losses), np.std(avg_losses)
    bad_batches = [i for i, loss in enumerate(avg_losses)
                   if mean_ - 3 * std_ > loss or mean_ + 3 * std_ < loss]
    print('Identified bad batches:', bad_batches)
    return bad_batches


def gaussian0():
    """
    Experiment 0: no poisoning, static IID data.
    """

    print('Experiment 0')

    train_x, train_y, test_x, test_y = _basic_train_test()

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


def gaussian1():
    """
    Experiment 1: flip labels in several batches in the middle.
    """

    print('Experiment 1')

    train_x, train_y, test_x, test_y = _basic_train_test()

    # Flip some labels in the middle of the training set.
    # There are 80 batches in total, so flip labels in batches 40 to 50.
    train_y = _flip_labels(train_y, 40 * BATCH_SIZE, 50 * BATCH_SIZE, 1.)

    dense_nn = dense.DenseNN(name='gaussian1',
                             input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                             classes=[0, 1], batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y))

    print('=====')
    print('Evaluate against test data:')
    print(dense_nn.evaluate(test_x, test_y))

    print()
    print('Experiment 1 complete')


def gaussian2():
    """
    Experiment 2: flip labels in the middle, with varying flip probability,
    compute distances between memory maps with normal vs flipped labels.
    """

    print('Experiment 2')

    train_x, train_y, test_x, test_y = _basic_train_test()

    # First train the (base) model without flipping any labels.
    dense_nn = dense.DenseNN(name='gaussian2_base',
                             input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                             classes=[0, 1], batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=NUM_EPOCHS)
    base_mmaps = dense_nn.epoch_mmaps
    # Epoch -> list of distances.
    mmap_distances = dict([(epoch, []) for epoch in range(NUM_EPOCHS)])

    start_idx, end_idx = 40 * BATCH_SIZE, 50 * BATCH_SIZE
    flip_probas = np.around(np.arange(start=0., stop=1., step=0.05), decimals=2)
    # TODO Check that no labels are flipped when flip_proba = 0.0
    for flip_proba in flip_probas:
        new_train_y = _flip_labels(train_y, start_idx, end_idx, flip_proba)
        dense_nn = dense.DenseNN(name='gaussian2_flip={}'.format(flip_proba),
                                 input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                                 classes=[0, 1], batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        dense_nn.fit(train_x, new_train_y, validation_data=(test_x, test_y),
                     num_epochs=NUM_EPOCHS)
        mmaps = dense_nn.epoch_mmaps
        for epoch, distance in enumerate(_mmap_dist(base_mmaps, mmaps)):
            mmap_distances[epoch].append(distance)

    plt.title('Distances between flipped mmap and base mmap')
    plt.xlabel('Flip probability')
    plt.ylabel('Distance')
    for epoch in range(NUM_EPOCHS):
        epoch_distances = mmap_distances[epoch]
        plt.plot(flip_probas, epoch_distances, c=next(CYCOL),
                 label='Epoch {}'.format(epoch + 1))
    plt.legend()
    plt.savefig('mmap_dist.png', dpi=150)
    plt.gcf().clear()

    print()
    print('Experiment 2 complete')


def gaussian3():
    """
    Experiment 3: same as 2, but with an attempt to locate and remove poisoned
    batches, and recalibrate after removing each bad batch.
    """

    print('Experiment 3')

    train_x, train_y, test_x, test_y = _basic_train_test()

    # Poison several scattered batches with different flip probabilities.
    train_y = _flip_labels(train_y, 20 * BATCH_SIZE, 23 * BATCH_SIZE, 0.3)
    train_y = _flip_labels(train_y, 40 * BATCH_SIZE, 45 * BATCH_SIZE, 0.5)
    train_y = _flip_labels(train_y, 70 * BATCH_SIZE, 72 * BATCH_SIZE, 0.9)

    dense_nn = dense.DenseNN(name='gaussian3a',
                             input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                             classes=[0, 1], batch_size=BATCH_SIZE,
                             # Do we need to normalise?
                             mmap_normalise=False)

    # Run the first 5 epochs with poisoned data and obtain the last mmap.
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=5)

    last_mmap = dense_nn.epoch_mmaps[-1]
    bad_batch_indices = _find_bad_batch(last_mmap)

    # Remove bad batches and recalibrate until no more bad batches are found.
    while bad_batch_indices:
        train_x, train_y = _remove_batches(train_x, train_y, bad_batch_indices)

        print('Recalibrating...')
        # Recalibrate for 1 epoch and look at the last mmap again.
        dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                     num_epochs=1, continue_training=True)
        print('Recalibration done')

        last_mmap = dense_nn.epoch_mmaps[-1]
        bad_batch_indices = _find_bad_batch(last_mmap)

    # Final fitting to double check the mmap.
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=1, continue_training=True)

    print()
    print('Experiment 3 complete')


def main():
    gaussian3()


if __name__ == '__main__':
    main()
