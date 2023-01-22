"""
Various shared constants and utils to clean up the experiment files.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from itertools import cycle
from numpy.linalg import norm
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


def _plot_2d(train_x, train_y, train_batch_id, test_x, test_y, plot_dir,
             name='data', sample=0.1, extra_x=None, extra_y=None):
    # Plot first two components of the two Gaussians.
    sample_idx = np.random.choice(np.arange(train_x.shape[0]), replace=False, size=int(sample * len(train_x)))
    sample_idx = sorted(sample_idx)
    train_x_sample = train_x[sample_idx]
    train_y_sample = train_y[sample_idx]
    tmp = train_x_sample[train_y_sample[:, 0] == 1]
    plt.scatter(tmp[:, 0], tmp[:, 1], color='blue', s=1, label='Class 0')
    tmp = train_x_sample[train_y_sample[:, 1] == 1]
    plt.scatter(tmp[:, 0], tmp[:, 1], color='red', s=1, label='Class 1')

    if extra_x is not None and extra_y is not None:
        for i in range(len(extra_x)):
            tmp = extra_x[i][extra_y[i][:, 0] == 1]
            plt.scatter(tmp[:, 0], tmp[:, 1], color='blue', s=1, marker='x')
            tmp = extra_x[i][extra_y[i][:, 1] == 1]
            plt.scatter(tmp[:, 0], tmp[:, 1], color='red', s=1, marker='x')

    plt.xlabel('Component 0')
    plt.ylabel('Component 1')
    plt.legend()
    plt.title('Data - first 2 components')
    plt.savefig('{}/{}.png'.format(plot_dir, name), dpi=150)
    plt.gcf().clear()

    # Save the data for easier debugging.
    with open('{}/{}.pkl'.format(plot_dir, name), 'wb') as f:
        pkl.dump({
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y,
            'train_batch_id': train_batch_id,
        }, f)


def basic_train_test(mu_0, sigma_0, mu_1, sigma_1, plot_dir=None,
                     log_reg=False, enforce_50_50=True):
    """
    Samples from two given Gaussians. No flipping labels.
    """

    train_set_size = (NUM_TRAIN_BATCHES // 2) * BATCH_SIZE  # Per class.
    test_set_size = (NUM_TEST_BATCHES // 2) * BATCH_SIZE  # Per class.

    train_x_0, train_y_0 = _generate_data(mu_0, sigma_0, 0, train_set_size)
    test_x_0, test_y_0 = _generate_data(mu_0, sigma_0, 0, test_set_size)

    train_x_1, train_y_1 = _generate_data(mu_1, sigma_1, 1, train_set_size)
    test_x_1, test_y_1 = _generate_data(mu_1, sigma_1, 1, test_set_size)

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

    if plot_dir:
        _plot_2d(train_x, train_y, train_batch_id, test_x, test_y, plot_dir, name='data')

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


def flip_labels_center(train_x, train_y, start_idx, end_idx,
                       mu0, sigma0, mu1, sigma1,
                       flip_proba, plot_dir):
    print('Shift center labels in the interval [{}, {}) (batches {}-{})'
          .format(start_idx, end_idx, start_idx // BATCH_SIZE, end_idx // BATCH_SIZE))

    center = (mu0 + mu1) / 2
    std0 = sigma0[0, 0]
    std1 = sigma1[0, 0]
    std = np.eye(INPUT_DIM, dtype=np.float64) * min(std0, std1) / 2

    # Flip the class with the highest std.
    flip_class = 0 if std0 >= std1 else 1

    idx = np.random.choice(np.arange(start_idx, end_idx),
                           size=int(flip_proba * (end_idx - start_idx)),
                           replace=False)

    x, y = _generate_data(center, std, flip_class, len(idx) * 3)  # Sample enough data.
    to_replace = idx[np.argmax(train_y[idx], axis=1) == flip_class]
    train_x[to_replace] = x[:len(to_replace)]  # Labels stay the same.

    return to_replace


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
