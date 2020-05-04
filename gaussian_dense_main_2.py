"""
Sample from 2 Gaussians and classify them, track mmap: part 2.
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from numpy.linalg import matrix_rank

import utils
from constants import *
from ml import dense
from gaussian_params_2_classes import DISTRIB_PARAMS


def _plot_cond_and_eig(eig_by_epoch, cond_by_epoch, name, short_name, moder_dir):
    epoch_ticks = [t for t in range(len(cond_by_epoch))]
    epoch_labels = [l + 1 for l in range(len(cond_by_epoch))]

    plt.title('{} - condition number by epoch'.format(name))
    plt.xlabel('Epoch')
    plt.ylabel('Condition number')
    plt.xticks(ticks=epoch_ticks, labels=epoch_labels)
    plt.plot(cond_by_epoch)
    plt.savefig('./{}/cond_{}.png'.format(moder_dir, short_name), dpi=150)
    plt.gcf().clear()

    plt.title('{} - eigenvalues by epoch'.format(name))
    plt.xlabel('Epoch')
    plt.ylabel('Eigenvalues')
    plt.boxplot(eig_by_epoch)
    plt.savefig('./{}/eig_{}.png'.format(moder_dir, short_name), dpi=150)
    plt.gcf().clear()


def _plot_square_matrix(mx,
                        name, short_name,
                        cond,
                        model_dir, current_epoch,
                        xlabel=None, ylabel=None,
                        xlabel_top=False):
    assert len(mx.shape) == 2
    assert mx.shape[0] == mx.shape[1]

    tick_labels = [str(k) if k % 10 == 0 or k == mx.shape[0] - 1 else None for k in range(mx.shape[0])]
    ax = sns.heatmap(mx, xticklabels=tick_labels, yticklabels=tick_labels)

    if xlabel_top:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    plt.title('{} - epoch {} - cond {}'.format(name, current_epoch, cond))
    plt.savefig('./{}/epoch{}_{}'.format(model_dir, current_epoch, short_name),
                dpi=150)
    plt.gcf().clear()


def _compute_and_plot_x_corr(mmap, model_dir, current_epoch):
    # First compute the matrix and plot it.
    corr_mx = np.zeros(shape=mmap.shape)

    # Compute the diagonal and upper triangle.
    for i in range(mmap.shape[0]):
        for j in range(i, mmap.shape[0]):  # Add self-correlation for reference.
            corr_mx[i, j] = np.corrcoef(mmap[i, :], mmap[j, :])[0, 1]

    # Fill in lower triangle.
    for i in range(mmap.shape[0]):
        for j in range(0, i):
            corr_mx[i, j] = corr_mx[j, i]

    # Absolute numbers of eigenvalues and condition number.
    eig = [np.absolute(v) for v in np.linalg.eig(corr_mx)[0]]
    cond = np.linalg.cond(corr_mx)

    # Plot matrix as a heatmap.
    _plot_square_matrix(corr_mx,
                        name='Batch cross-correlation matrix', short_name='batch_xcorr',
                        cond=cond, model_dir=model_dir, current_epoch=current_epoch, xlabel_top=True)

    return eig, cond


def _compute_and_plot_auto_corr(mmap, model_dir, current_epoch):
    auto_corr_mx = np.zeros(shape=mmap.shape)
    auto_corr_mx[:, 0] = 1.

    for t in range(1, mmap.shape[0]):
        for i in range(mmap.shape[0]):
            auto_corr_mx[i, t] = np.corrcoef(mmap[i, :-t], mmap[i, t:])[0][1]
    auto_corr_mx[:, -1] = 1.
    # After a few practice runs, it looks like some auto-correlation values might be nan.
    # This is a hack to be able to compute eigenvalues later.
    auto_corr_mx[np.isnan(auto_corr_mx)] = 2.

    # Absolute numbers of eigenvalues and condition number.
    eig = [np.absolute(v) for v in np.linalg.eig(auto_corr_mx)[0]]
    cond = np.linalg.cond(auto_corr_mx)

    # Plot matrix as a heatmap.
    _plot_square_matrix(auto_corr_mx,
                        name='Batch auto-correlation matrix', short_name='batch_autocorr',
                        cond=cond, model_dir=model_dir, current_epoch=current_epoch, xlabel='Lag', ylabel='Batch')

    return eig, cond


def _gaussian5_exp(experiment_dir, train_x, train_y, test_x, test_y):
    exp_id = experiment_dir.split('/')[-1]

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=exp_id,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(0, 1), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    # Absolute eigenvalues and condition number by epoch.
    mmap_cond, mmap_eig = [], []
    batch_xcorr_cond, batch_xcorr_eig = [], []
    batch_autocorr_cond, batch_autocond_eig = [], []

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

        # 2. Cross-correlation matrix.
        xcorr_eig, xcorr_cond = _compute_and_plot_x_corr(mmap, model_dir, current_epoch)
        batch_xcorr_eig.append(xcorr_eig)
        batch_xcorr_cond.append(xcorr_cond)

        # 3. Compute autocorrelation matrix.
        autocorr_eig, autocorr_cond = _compute_and_plot_auto_corr(mmap, model_dir, current_epoch)
        batch_autocond_eig.append(autocorr_eig)
        batch_autocorr_cond.append(autocorr_cond)

        # 4. Mmap eigenvalues and cond number.
        mmap_eig.append([np.absolute(v) for v in np.linalg.eig(mmap)[0]])
        mmap_cond.append(np.linalg.cond(mmap))

    # Plot eigenvalues and condition numbers for all matrices.
    _plot_cond_and_eig(batch_xcorr_eig, batch_xcorr_cond, 'Batch cross-correlation matrix', 'batch_xcorr', model_dir)
    _plot_cond_and_eig(batch_autocond_eig, batch_autocorr_cond, 'Batch auto-correlation matrix', 'batch_autocorr', model_dir)
    _plot_cond_and_eig(mmap_eig, mmap_cond, 'Mmap', 'mmap', model_dir)


def gaussian5a_batch_corr():
    """
    Plot loss sequences for batches.
    Compute and plot batch cross-correlation matrix.

    No data is poisoned.
    """

    parent_dir = 'gaussian5a_batch_corr_new'

    for exp_id, mu0, sigma0, mu1, sigma1 in DISTRIB_PARAMS:
        print()
        print('==============================')
        print()
        print('Experiment ID:', exp_id)
        experiment_dir = '{}/{}'.format(parent_dir, exp_id)
        os.makedirs(experiment_dir, exist_ok=True)
        train_x, train_y, test_x, test_y, _ = utils.basic_train_test(mu0, sigma0, mu1, sigma1,
                                                                     plot_dir=experiment_dir,
                                                                     log_reg=True)
        _gaussian5_exp(experiment_dir, train_x, train_y, test_x, test_y)


def gaussian5b_batch_corr():
    """
    Plot loss sequences for batches.
    Compute and plot batch cross-correlation matrix.

    The following batches are poisoned with different flip probabilities:
         * FP = 0.1  : batches 10-14, 66-70 (10 batches).
         * FP = 0.25 : batches 30-34, 46-50 (10 batches).
         * FP = 0.5  : batches 39-41 (3 batches).
    Total poisoned batches: 23.
    Total training batches: 80.
    """

    parent_dir = 'gaussian5b_batch_corr_new'

    for exp_id, mu0, sigma0, mu1, sigma1 in DISTRIB_PARAMS:
        print()
        print('==============================')
        print()
        print('Experiment ID:', exp_id)
        experiment_dir = '{}/{}'.format(parent_dir, exp_id)
        os.makedirs(experiment_dir, exist_ok=True)

        train_x, train_y, test_x, test_y, _ = utils.basic_train_test(mu0, sigma0, mu1, sigma1,
                                                                     plot_dir=experiment_dir,
                                                                     log_reg=True)

        # [(flip probability, [(start of batch, end of batch - exclusive)])]
        flip_settings = [
            (0.1,  [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
            (0.25, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
            (0.5,  [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
        ]
        for flip_proba, poisoned_batch_settings in flip_settings:
            for start_idx, end_idx in poisoned_batch_settings:
                train_y = utils.flip_labels(train_y, start_idx, end_idx, flip_proba, copy=False)

        _gaussian5_exp(experiment_dir, train_x, train_y, test_x, test_y)


def gaussian6a_rank():
    parent_dir = 'gaussian6a_rank'

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

        ranks = []
        for epoch in range(NUM_EPOCHS):
            dense_nn.fit(train_x, train_y,
                         validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

            # Model is loaded during fit(). Start from scratch every time.
            assert len(dense_nn.epoch_mmaps) == epoch + 1

            mmap, _, _ = dense_nn.epoch_mmaps[-1]
            assert mmap.shape[0] == NUM_TRAIN_BATCHES

            corr_mx = _compute_and_plot_x_corr(mmap, fill_lower=True)
            r = matrix_rank(corr_mx, hermitian=True)
            ranks.append(r)

        plt.title('Rank of cross-correlation matrix')
        plt.xlabel('Epoch')
        plt.ylabel('Rank value')
        plt.plot(np.arange(NUM_EPOCHS) + 1, ranks)
        plt.savefig('./{}/x_corr_rank.png'.format(model_dir), dpi=150)
        plt.gcf().clear()


def gaussian6b_rank():
    parent_dir = 'gaussian6b_rank'

    for exp_id, mu0, sigma0, mu1, sigma1 in DISTRIB_PARAMS:
        print()
        print('==============================')
        print()
        print('Experiment ID:', exp_id)
        experiment_dir = '{}/{}'.format(parent_dir, exp_id)
        os.makedirs(experiment_dir, exist_ok=True)

        train_x, train_y, test_x, test_y, _ = utils.basic_train_test(mu0, sigma0, mu1, sigma1,
                                                                     plot_dir=experiment_dir)
        # [(flip probability, [(start of batch, end of batch - exclusive)])]
        flip_settings = [
            (0.1, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
            (0.25, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
            (0.5, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
        ]
        for flip_proba, poisoned_batch_settings in flip_settings:
            for start_idx, end_idx in poisoned_batch_settings:
                train_y = utils.flip_labels(train_y, start_idx, end_idx, flip_proba, copy=False)

        dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                                 name=exp_id,
                                 input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                                 classes=(0, 1), batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        model_dir = dense_nn.model_dir
        print('Model dir:', model_dir)

        ranks = []
        for epoch in range(NUM_EPOCHS):
            dense_nn.fit(train_x, train_y,
                         validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

            # Model is loaded during fit(). Start from scratch every time.
            assert len(dense_nn.epoch_mmaps) == epoch + 1

            mmap, _, _ = dense_nn.epoch_mmaps[-1]
            assert mmap.shape[0] == NUM_TRAIN_BATCHES

            corr_mx = _compute_and_plot_x_corr(mmap, fill_lower=True)
            r = matrix_rank(corr_mx, hermitian=True)
            ranks.append(r)

        plt.title('Rank of cross-correlation matrix')
        plt.xlabel('Epoch')
        plt.ylabel('Rank value')
        plt.plot(np.arange(NUM_EPOCHS) + 1, ranks)
        plt.savefig('./{}/x_corr_rank.png'.format(model_dir), dpi=150)
        plt.gcf().clear()


def main():
    gaussian5a_batch_corr()
    gaussian5b_batch_corr()
    pass


if __name__ == '__main__':
    main()
