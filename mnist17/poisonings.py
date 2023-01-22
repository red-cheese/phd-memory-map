import matplotlib.pyplot as plt
import numpy as np
import os

from numpy.random import normal

import utils_common
from mnist17.constants import *


def base(train_x, train_y):
    true_batch_labels = np.zeros(shape=(NUM_TRAIN_BATCHES,))
    return train_x, train_y, true_batch_labels


def flip_poisoning(train_x, train_y):
    flip_settings = [
        (0.1, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.25, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.5, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
    ]

    for flip_proba, poisoned_batch_settings in flip_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_y = utils_common.flip_labels(train_y, start_idx, end_idx, flip_proba, BATCH_SIZE, copy=False)

    true_batch_labels = np.zeros(shape=(NUM_TRAIN_BATCHES,))
    true_batch_labels[10:15] = 1
    true_batch_labels[66:71] = 1
    true_batch_labels[30:35] = 2
    true_batch_labels[46:51] = 2
    true_batch_labels[39:42] = 3

    return train_x, train_y, true_batch_labels


def noise_poisoning(train_x, train_y, plot_dir=None, plot=False):
    if plot:
        assert plot_dir is not None

    # [(Gaussian noise STD, [(start of batch, end of batch - exclusive)])]
    noise_settings = [
        (0.3, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.5, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.7, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
    ]
    for noise_std, poisoned_batch_settings in noise_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_x = _add_noise(train_x, start_idx, end_idx, noise_std, plot_dir, plot=plot)
    true_batch_labels = np.zeros(shape=(NUM_TRAIN_BATCHES,))
    true_batch_labels[10:15] = 1
    true_batch_labels[66:71] = 1
    true_batch_labels[30:35] = 2
    true_batch_labels[46:51] = 2
    true_batch_labels[39:42] = 3

    return train_x, train_y, true_batch_labels


def _add_noise(x, start_idx, end_idx, noise_std, iter_dir, plot=True):
    """
    Noise poisoning: add Gaussian noise.
    """

    noise_sample = normal(0., noise_std, size=x[start_idx:end_idx].shape)
    x[start_idx:end_idx] += noise_sample
    x = np.clip(x, a_min=0, a_max=1)

    if plot:
        f, ax = plt.subplots(2, 5, figsize=(10, 5))
        ax = ax.flatten()
        for i in range(10):
            idx = start_idx + i
            ax[i].imshow(x[idx].reshape(28, 28))
        plt.savefig(os.path.join(iter_dir, 'noise{}.png'.format(noise_std)), dpi=150)
        plt.gcf().clear()

    return x