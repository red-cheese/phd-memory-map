"""
Various shared constants and utils to clean up the experiment files.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from itertools import cycle
from numpy.random import multivariate_normal
from sklearn.linear_model import LogisticRegression

from constants import *


CYCOL = cycle('bgrcmk')  # For colouring plots.


# =============================================================================
# Data utils
# =============================================================================


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


def basic_train_test(mu_0, sigma_0, mu_1, sigma_1, plot_dir=None,
                     log_reg=False):
    """
    Samples from two given Gaussians. No flipping labels.
    """

    train_set_size = (NUM_TRAIN_BATCHES // 2) * BATCH_SIZE  # Per class.
    test_set_size = (NUM_TEST_BATCHES // 2) * BATCH_SIZE  # Per class.

    train_x_0, train_y_0 = _generate_data(mu_0, sigma_0, 0, train_set_size)
    test_x_0, test_y_0 = _generate_data(mu_0, sigma_0, 0, test_set_size)

    train_x_1, train_y_1 = _generate_data(mu_1, sigma_1, 1, train_set_size)
    test_x_1, test_y_1 = _generate_data(mu_1, sigma_1, 1, test_set_size)

    train_x, train_y = _vstack([(train_x_0, train_y_0),
                                (train_x_1, train_y_1)],
                               shuffle=True)
    test_x, test_y = _vstack([(test_x_0, test_y_0),
                              (test_x_1, test_y_1)],
                             shuffle=True)

    # Helper array to track removed batches in the mmap-based defense.
    train_batch_id = np.zeros((NUM_TRAIN_BATCHES * BATCH_SIZE,), dtype=np.int32)
    for i in range(NUM_TRAIN_BATCHES):
        train_batch_id[i * BATCH_SIZE:((i + 1) * BATCH_SIZE)] = i

    if plot_dir:
        # Plot first two components of the two Gaussians.
        # train_set_size is per class, hence sample 10% of all data.
        sample_idx = np.random.choice(np.arange(train_x.shape[0]), replace=False, size=int(0.2 * train_set_size))
        sample_idx = sorted(sample_idx)
        train_x_sample = train_x[sample_idx]
        train_y_sample = train_y[sample_idx]
        tmp = train_x_sample[train_y_sample[:, 0] == 1]
        plt.scatter(tmp[:, 0], tmp[:, 1], color='blue', s=1, label='Class 0')
        tmp = train_x_sample[train_y_sample[:, 1] == 1]
        plt.scatter(tmp[:, 0], tmp[:, 1], color='red', s=1, label='Class 1')
        plt.xlabel('Component 0')
        plt.ylabel('Component 1')
        plt.legend()
        plt.title('Data - first 2 components')
        plt.savefig('{}/data.png'.format(plot_dir), dpi=150)
        plt.gcf().clear()

        # Save the data for easier debugging.
        with open('{}/data.pkl'.format(plot_dir), 'wb') as f:
            pkl.dump({
                'train_x': train_x,
                'train_y': train_y,
                'test_x': test_x,
                'test_y': test_y,
                'train_batch_id': train_batch_id,
            }, f)

        if log_reg:
            train_y_labels = np.argmax(train_y, axis=1)
            # Watch out for any warnings.
            clf = LogisticRegression(random_state=0, verbose=1).fit(train_x, train_y_labels)
            score = clf.score(train_x, train_y_labels)
            with open('{}/log_reg.txt'.format(plot_dir), 'wt') as f:
                f.write('Logistic Regression score: {}'.format(score))


    return train_x, train_y, test_x, test_y, train_batch_id


# =============================================================================
# Memory map utils
# =============================================================================


def flip_labels(orig_labels, start_idx, end_idx, flip_proba, copy=True):
    print('Flip labels in the interval [{}, {}) (batches {}-{}) with probability {}'
          .format(start_idx, end_idx, start_idx // BATCH_SIZE, end_idx // BATCH_SIZE, flip_proba))

    labels = np.copy(orig_labels) if copy else orig_labels

    idx = np.random.choice(np.arange(start_idx, end_idx),
                           size=int(flip_proba * (end_idx - start_idx)),
                           replace=False)
    old_labels = labels[idx, :]
    labels[idx, :] = 1 - old_labels

    return labels


def remove_batches(xs, ys, batches_to_remove, train_batch_id):
    batches_to_remove_set = set(batches_to_remove)
    to_keep = [i for i in range(xs.shape[0])
               if i // BATCH_SIZE not in batches_to_remove_set]
    to_keep_set = set(to_keep)
    removed_batches = {batch_id for j, batch_id in enumerate(train_batch_id)
                       if j not in to_keep_set}
    print('Removed bad batches:', sorted(removed_batches))
    return xs[to_keep], ys[to_keep], train_batch_id[to_keep], removed_batches


def dist(x1, x2):
    return np.linalg.norm(x1 - x2) ** 2


def mmap_dist(ms1, ms2):
    """Returns distances between per-epoch memory maps."""
    return [np.linalg.norm(m1 - m2) for m1, m2 in zip(ms1, ms2)]


def find_bad_batch(mmap):
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
    print('(Warn: skewed indexing!) Identified bad batches:', bad_batches)
    return bad_batches
