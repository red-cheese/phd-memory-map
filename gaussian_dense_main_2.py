"""
Sample from 2 Gaussians and classify them, track mmap: part 2.
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

import utils
from constants import *
from ml import dense
from gaussian_params_2_classes import DISTRIB_PARAMS


def gaussian5a_batch_corr():
    """
    Plot loss sequences for batches.
    Compute and plot batch cross-correlation matrix.

    No data is poisoned.
    """

    parent_dir = 'gaussian5a_batch_corr'

    for exp_id, mu0, sigma0, mu1, sigma1 in DISTRIB_PARAMS:
        print()
        print('==============================')
        print()
        print('Experiment ID:', exp_id)
        experiment_dir = '{}/{}'.format(parent_dir, exp_id)
        os.makedirs(experiment_dir, exist_ok=True)

        train_x, train_y, test_x, test_y, _ = utils.basic_train_test(mu0, sigma0, mu1, sigma1,
                                                                     plot_dir=experiment_dir)

        dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                                 name=exp_id,
                                 input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                                 classes=(0, 1), batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        model_dir = dense_nn.model_dir
        print('Model dir:', model_dir)

        for epoch in range(NUM_EPOCHS):
            current_epoch = len(dense_nn.epoch_mmaps) + 1

            dense_nn.fit(train_x, train_y,
                         validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

            # Model is loaded during fit(). Start from scratch every time.
            assert len(dense_nn.epoch_mmaps) == epoch + 1

            mmap, _, _ = dense_nn.epoch_mmaps[-1]
            assert mmap.shape[0] == NUM_TRAIN_BATCHES

            # 1. Plot batch loss sequences (10 sequences in each plot).
            for i in range(0, mmap.shape[0], 10):  # 0, 10, 20, ...
                plt.title('Mini-batch losses: batches {}-{} - epoch {}'.format(i, i + 9, current_epoch))
                plt.xlabel('Training step')
                plt.ylabel('Loss value')
                for j in range(i, i + 10):
                    plt.plot(mmap[j, :], label='Batch {}'.format(j))
                plt.legend()
                plt.savefig('./{}/epoch{}_batch_losses_{}-{}.png'
                            .format(model_dir, current_epoch, i, i + 9),
                            dpi=150)
                plt.gcf().clear()

            # 2. Compute and plot cross-correlation matrix.
            corr_mx = np.zeros(shape=mmap.shape)
            for i in range(mmap.shape[0]):
                for j in range(i, mmap.shape[0]):  # Add self-correlation for reference.
                    losses_i = mmap[i, :]
                    losses_j = mmap[j, :]
                    corr = np.corrcoef(losses_i, losses_j)[0, 1]
                    corr_mx[i, j] = corr
            mask = np.zeros(shape=mmap.shape, dtype=np.bool)
            mask[np.tril_indices_from(mask, -1)] = True  # Include self-corr.
            tick_labels = [str(k) if k % 10 == 0 or k == mmap.shape[0] - 1 else None
                           for k in range(mmap.shape[0])]
            ax = sns.heatmap(corr_mx, xticklabels=tick_labels, yticklabels=tick_labels, mask=mask)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            plt.title('Batch cross-correlation matrix - epoch {}'.format(current_epoch))
            plt.savefig('./{}/epoch{}_batch_xcorr'.format(model_dir, current_epoch),
                        dpi=150)
            plt.gcf().clear()


def main():
    gaussian5a_batch_corr()


if __name__ == '__main__':
    main()
