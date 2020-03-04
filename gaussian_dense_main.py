"""
Sample from 2 Gaussians and classify them, track mmap.
"""


import numpy as np
import os
import seaborn as sns
from keras import metrics as keras_metrics

from scipy.stats import kstest, norm, normaltest, probplot, shapiro

from gaussian_params_2_classes import DISTRIB_PARAMS
from ml import dense
from ml import metrics as metrics_
from utils import *

NUM_EPOCHS = 5


def gaussian0():
    """
    Experiment 0: no poisoning, static IID data.
    """

    print('Experiment 0')

    train_x, train_y, test_x, test_y = basic_train_test()

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


def gaussian0a_loss_distrib():
    """
    Experiment 0a: monitor distributions of mmap values (across all batches)
    at each training step, for several different pairs of Gaussians.

    No poisoning, static IID data.

    Inspired by
    https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    """

    parent_dir = 'gaussian0a_loss_distrib'
    num_epochs = 10

    # List of experiment specs: [ Exp 1: (mu0, sigma0, mu1, sigma1), ... ]
    distrib_params = [
        (
            # G1: (5.0, 0.1 * I), G2: (2.0, I) - very well separated Gaussians
            'exp0',
            np.full((INPUT_DIM,), 5., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64) * 0.1,
            np.full((INPUT_DIM,), 2., dtype=np.float64), np.eye(INPUT_DIM, dtype=np.float64)
        ),
    ]

    for exp_id, mu0, sigma0, mu1, sigma1 in distrib_params:
        print()
        print('==============================')
        print()
        print('Experiment ID:', exp_id)
        experiment_dir = '{}/{}'.format(parent_dir, exp_id)
        os.makedirs(experiment_dir, exist_ok=True)

        train_x, train_y, test_x, test_y, _ = basic_train_test(mu_0=mu0, sigma_0=sigma0, mu_1=mu1, sigma_1=sigma1,
                                                               plot_dir=experiment_dir)

        dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                                 name=exp_id,
                                 input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                                 classes=[0, 1], batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        model_dir = dense_nn.model_dir
        epochs_done = len(dense_nn.epoch_mmaps)
        print('Model dir:', model_dir)
        print('Epochs already done:', epochs_done)

        for epoch in range(num_epochs):
            current_epoch = epochs_done + epoch + 1

            dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y), num_epochs=1,
                         continue_training=True)
            mmap = dense_nn.epoch_mmaps[-1]

            # Plot 1: Sample means and stds of losses for each training step.
            mmap_means = np.mean(mmap, axis=0)
            mmap_stds = np.std(mmap, axis=0)
            sns.regplot(np.arange(len(mmap_means)), mmap_means, color='green', label='Sample mean', scatter_kws={'s': 2})
            sns.regplot(np.arange(len(mmap_stds)), mmap_stds, color='orange', label='Sample std', scatter_kws={'s': 2})
            plt.xlabel('Training step')
            plt.xlabel('Value')
            plt.legend()
            plt.title('Sample mean and std of losses per training step - epoch {}'.format(current_epoch))
            plt.savefig('./{}/epoch{}_means_stds.png'.format(model_dir, current_epoch), dpi=150)
            plt.gcf().clear()

            # Plot 2: Histograms of losses for each 10th training step, plotted all together.
            for i in range(5, NUM_TRAIN_BATCHES, 10):
                step_losses = mmap[:, i]  # Losses for all batches on step i.
                plt.hist(step_losses, bins=15, density=False, alpha=0.5, label='after step {}'.format(i),
                         edgecolor='black', linewidth=0.5)
            plt.xlabel('Loss value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Losses per training step - epoch {}'.format(current_epoch))
            plt.savefig('./{}/epoch{}_hist.png'.format(model_dir, current_epoch), dpi=150)
            plt.gcf().clear()

            # Plot 3: Test batch losses per training step for normality.
            assert mmap.shape[0] == mmap.shape[1] == NUM_TRAIN_BATCHES  # As no data gets removed.
            W_stats, p_sw_values = [], []  # Shapiro-Wilk
            D_stats, p_ks_values = [], []  # KS
            K2_stats, p_k2_values = [], []  # K^2
            step_losses = None
            for i in range(mmap.shape[1]):
                step_losses = mmap[:, i]  # Sample.

                # Shapiro-Wilk.
                W, p_sw = shapiro(step_losses)
                W_stats.append(W)
                p_sw_values.append(p_sw)

                # Kolmogorov-Smirnov.
                # Need to fit first.
                loc, scale = norm.fit(step_losses)
                dist = norm(loc, scale)
                D, p_ks = kstest(step_losses, dist.cdf, N=len(step_losses))
                D_stats.append(D)
                p_ks_values.append(p_ks)

                # K^2.
                K2, p_k2 = normaltest(step_losses)
                K2_stats.append(K2)
                p_k2_values.append(p_k2)

            # QQ-plot on the last step losses.
            probplot(step_losses, plot=plt)
            plt.title('QQ plot for last step losses - epoch {}'.format(current_epoch))
            plt.savefig('./{}/epoch{}_qq.png'.format(model_dir, current_epoch), dpi=150)
            plt.gcf().clear()

            sns.regplot(np.arange(len(W_stats)), np.log(W_stats), color='green', label='Shapiro-Wilk test statistic (log)', scatter_kws={'s': 2})
            sns.regplot(np.arange(len(p_sw_values)), p_sw_values, color='orange', label='p-value', scatter_kws={'s': 2})
            plt.xlabel('Training step')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Shapiro-Wilk test statistic and p-value per training step - epoch {}'.format(current_epoch))
            plt.savefig('./{}/epoch{}_shapiro.png'.format(model_dir, current_epoch), dpi=150)
            plt.gcf().clear()

            sns.regplot(np.arange(len(D_stats)), np.log(D_stats), color='green', label='Kolmogorov-Smirnov test statistic (log)', scatter_kws={'s': 2})
            sns.regplot(np.arange(len(p_ks_values)), p_ks_values, color='orange', label='p-value', scatter_kws={'s': 2})
            plt.xlabel('Training step')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Kolmogorov-Smirnov test statistic and p-value per training step - epoch {}'.format(current_epoch))
            plt.savefig('./{}/epoch{}_ks.png'.format(model_dir, current_epoch), dpi=150)
            plt.gcf().clear()

            sns.regplot(np.arange(len(K2_stats)), np.log(K2_stats), color='green', label='K^2 test statistic (log)', scatter_kws={'s': 2})
            sns.regplot(np.arange(len(p_k2_values)), p_k2_values, color='orange', label='p-value', scatter_kws={'s': 2})
            plt.xlabel('Training step')
            plt.ylabel('Value')
            plt.legend()
            plt.title("D'Agostino K^2 test statistic and p-value per training step - epoch {}".format(current_epoch))
            plt.savefig('./{}/epoch{}_dagostino_k2.png'.format(model_dir, current_epoch), dpi=150)
            plt.gcf().clear()


def gaussian0b_relative_positions():
    """
    Experiment 0b: see what happens to anomaly detection process if there is
    no poisoning.
    """

    parent_dir = 'gaussian0b_relative_positions'
    num_epochs = 10

    for exp_id, mu0, sigma0, mu1, sigma1 in DISTRIB_PARAMS:
        print()
        print('==============================')
        print()
        print('Experiment ID:', exp_id)
        experiment_dir = '{}/{}'.format(parent_dir, exp_id)
        os.makedirs(experiment_dir, exist_ok=True)

        # This will plot samples from the training set.
        train_x, train_y, test_x, test_y, _ = basic_train_test(mu_0=mu0, sigma_0=sigma0, mu_1=mu1, sigma_1=sigma1,
                                                               plot_dir=experiment_dir)

def gaussian1():
    """
    Experiment 1: flip labels in several batches in the middle.
    """

    print('Experiment 1')

    train_x, train_y, test_x, test_y, _ = basic_train_test()

    # Flip some labels in the middle of the training set.
    # There are 80 batches in total, so flip labels in batches 40 to 50.
    train_y = flip_labels(train_y, 40 * BATCH_SIZE, 50 * BATCH_SIZE, 1.)

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

    train_x, train_y, test_x, test_y, _ = basic_train_test()

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
        new_train_y = flip_labels(train_y, start_idx, end_idx, flip_proba)
        dense_nn = dense.DenseNN(name='gaussian2_flip={}'.format(flip_proba),
                                 input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                                 classes=[0, 1], batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        dense_nn.fit(train_x, new_train_y, validation_data=(test_x, test_y),
                     num_epochs=NUM_EPOCHS)
        mmaps = dense_nn.epoch_mmaps
        for epoch, distance in enumerate(mmap_dist(base_mmaps, mmaps)):
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

    train_x, train_y, test_x, test_y, _ = basic_train_test()

    # Poison several scattered batches with different flip probabilities.
    train_y = flip_labels(train_y, 20 * BATCH_SIZE, 23 * BATCH_SIZE, 0.3)
    train_y = flip_labels(train_y, 40 * BATCH_SIZE, 45 * BATCH_SIZE, 0.5)
    train_y = flip_labels(train_y, 70 * BATCH_SIZE, 72 * BATCH_SIZE, 0.9)

    dense_nn = dense.DenseNN(name='gaussian3a',
                             input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                             classes=[0, 1], batch_size=BATCH_SIZE,
                             # Do we need to normalise?
                             mmap_normalise=False)

    # Run the first 5 epochs with poisoned data and obtain the last mmap.
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=5)

    last_mmap = dense_nn.epoch_mmaps[-1]
    bad_batch_indices = find_bad_batch(last_mmap)

    # Remove bad batches and recalibrate until no more bad batches are found.
    while bad_batch_indices:
        train_x, train_y = remove_batches(train_x, train_y, bad_batch_indices)

        print('Recalibrating...')
        # Recalibrate for 1 epoch and look at the last mmap again.
        dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                     num_epochs=1, continue_training=True)
        print('Recalibration done')

        last_mmap = dense_nn.epoch_mmaps[-1]
        bad_batch_indices = find_bad_batch(last_mmap)

    # Final fitting to double check the mmap.
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=1, continue_training=True)

    print()
    print('Experiment 3 complete')


def _run_model_with_sphere(dense_nn, train_x, train_y, test_x, test_y,
                           train_batch_id, limit_flip_proba):
    """
    Sphere defense is a data sanitisation defense.

    See https://papers.nips.cc/paper/6943-certified-defenses-for-data-poisoning-attacks.pdf.
    """

    # All removed batches (original indexing).
    removed_batch_idx = []

    # Centroids of the two clusters (the data are already poisoned).
    class0 = train_y[:, 0] == 1
    mu0 = np.mean(train_x[class0], axis=0)
    class1 = train_y[:, 1] == 1
    mu1 = np.mean(train_x[class1], axis=0)
    print('Centroids: class 0 = {}, class 1 = {}'.format(mu0, mu1))

    # The allowed radius is 1/2 of the distance between the two centroids.
    r = 0.5 * dist(mu0, mu1)

    for i in range(NUM_TRAIN_BATCHES):
        batch_x = train_x[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)]
        batch_y = train_y[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)]
        batch_x_0 = batch_x[np.argmax(batch_y, axis=1) == 0]
        batch_x_1 = batch_x[np.argmax(batch_y, axis=1) == 1]
        num_poisoned = len([x for x in batch_x_0 if dist(x, mu0) >= r])
        num_poisoned += len([x for x in batch_x_1 if dist(x, mu1) >= r])
        if num_poisoned >= BATCH_SIZE * limit_flip_proba:
            removed_batch_idx.append(i)

    # Remove the identified batches.
    train_x, train_y, _, _ = remove_batches(train_x, train_y, removed_batch_idx, train_batch_id)

    # Fit the model.
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=NUM_EPOCHS)

    return dense_nn, removed_batch_idx


def _run_model_with_mmap(dense_nn, train_x, train_y, test_x, test_y, train_batch_id,
                         num_epochs=NUM_EPOCHS):
    """
    Helper method. Applies the mmap method to clean the data. Returns the final
    trained model and the indices of removed batches.

    Basically this is what happens in Experiment 3 (see above).
    """

    # All removed batches (original indexing).
    removed_batch_idx = []

    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y), num_epochs=num_epochs)

    last_mmap = dense_nn.epoch_mmaps[-1]
    bad_batch_idx = find_bad_batch(last_mmap)

    # Remove bad batches and recalibrate until no more bad batches are found.
    while bad_batch_idx:
        train_x, train_y, train_batch_id, removed_batches = remove_batches(train_x, train_y, bad_batch_idx, train_batch_id)
        removed_batch_idx.extend(removed_batches)

        print('Recalibrating...')
        # Recalibrate for 1 epoch and look at the last mmap again.
        dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                     num_epochs=1, continue_training=True)
        print('Recalibration done')

        last_mmap = dense_nn.epoch_mmaps[-1]
        bad_batch_idx = find_bad_batch(last_mmap)

    # Final fitting to double check the mmap.
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=1, continue_training=True)

    return dense_nn, removed_batch_idx


def _save_metrics(metrics_dict, metrics):
    acc, prec, rec, f1 = metrics
    metrics_dict['accuracy'].append(acc)
    metrics_dict['precision'].append(prec)
    metrics_dict['recall'].append(rec)
    metrics_dict['f1'].append(f1)


def _save_removal_metrics(metrics_dict, true_poisoned_batches, removed_batches):
    # Prepare one-hot encoded results.
    y_true = np.zeros((NUM_TRAIN_BATCHES, 2))
    y_true[:, 0] = 1.
    y_true[sorted(true_poisoned_batches), 0] = 0.
    y_true[sorted(true_poisoned_batches), 1] = 1.

    y_pred = np.zeros((NUM_TRAIN_BATCHES, 2))
    y_pred[:, 0] = 1.
    y_pred[sorted(removed_batches), 0] = 0.
    y_pred[sorted(removed_batches), 1] = 1.

    acc = keras_metrics.categorical_accuracy(y_true, y_pred)
    prec = metrics_.binary_precision(y_true, y_pred)
    rec = metrics_.binary_recall(y_true, y_pred)
    f1 = metrics_.binary_f1(y_true, y_pred)
    _save_metrics(metrics_dict, (acc, prec, rec, f1))

    print('All bad batches:', sorted(true_poisoned_batches))
    print('All removed batches:', sorted(removed_batches))


def _print_metrics(name, metrics):
    print('{}:\tacc {}\tprecision {}\trecall {}\tf1 {}'.format(
        name,
        np.mean(np.asarray(metrics['accuracy'])),
        np.mean(np.asarray(metrics['precision'])),
        np.mean(np.asarray(metrics['recall'])),
        np.mean(np.asarray(metrics['f1'])),
    ))


def gaussian4():
    """
    Experiment 4: assess quality of mmaps and other basic defenses, computing
    MC estimates.
    """

    print('Experiment 4')

    num_mc_runs = 5

    # =========================================================================
    # Hyperparameters which are likely to affect the results greatly.
    # =========================================================================
    flip_proba = 0.5  # In each poisoned batch, 50% labels are flipped.
    batch_poison_proba = 0.2  # 10% (0.5 * 0.2) of the training data are poisoned.
    num_poisoned_batches = int(batch_poison_proba * NUM_TRAIN_BATCHES)
    # =========================================================================

    # Model (final classifier) quality.
    no_defense_1_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}  # Baseline 1.
    sphere_2_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}  # Baseline 2.
    # slab_3_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}  # Baseline 3.
    mmap_4_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    # Removal quality of batches: are we removing good batches or bad batches.
    sphere_2_removal_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    mmap_4_removal_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for mc_idx in range(num_mc_runs):
        print('MC run {} starting...'.format(mc_idx))

        train_x, train_y, test_x, test_y, train_batch_id, _ = basic_train_test()

        # Randomly select which batches to poison.
        poisoned_batch_idx = np.random.choice(np.arange(NUM_TRAIN_BATCHES),
                                              size=num_poisoned_batches,
                                              replace=False)
        # Perform poisoning.
        for batch_i in poisoned_batch_idx:
            start, end = batch_i * BATCH_SIZE, (batch_i + 1) * BATCH_SIZE
            train_y = flip_labels(train_y, start, end, flip_proba)

        # Baseline 1.
        # Model trained for 5 epochs, no defense against poisoning.
        print('Baseline 1 - NN with no defense against poisoning')
        no_defense_1 = dense.DenseNN(name='gaussian4a_1_{}'.format(mc_idx),
                                 input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                                 classes=[0, 1], batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        no_defense_1.fit(train_x, train_y, validation_data=(test_x, test_y),
                     num_epochs=NUM_EPOCHS)
        loss, *metrics = no_defense_1.evaluate(test_x, test_y)
        _save_metrics(no_defense_1_metrics, metrics)
        print()

        # Baseline 2.
        # Remove outliers with sphere defense, then run a dense model.
        print('Baseline 2 - sphere defence')
        sphere_2 = dense.DenseNN(name='gaussian4a_2_{}'.format(mc_idx),
                                 input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                                 classes=[0, 1], batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        sphere_2, sphere_2_removed_batches = _run_model_with_sphere(sphere_2, train_x, train_y, test_x, test_y, train_batch_id, flip_proba)
        loss, *metrics = sphere_2.evaluate(test_x, test_y)
        _save_metrics(sphere_2_metrics, metrics)
        _save_removal_metrics(sphere_2_removal_metrics, poisoned_batch_idx, sphere_2_removed_batches)
        print()

        # Baseline 3.
        # Remove outliers with slab defense, then run a dense model.
        # TODO

        # Method 4.
        print('Main method - NN with mmap-based defense')
        mmap_4 = dense.DenseNN(name='gaussian4a_4_{}'.format(mc_idx),
                                 input_dim=INPUT_DIM, h1_dim=64, h2_dim=32,
                                 classes=[0, 1], batch_size=BATCH_SIZE,
                                 mmap_normalise=False)
        mmap_4, removed_batches = _run_model_with_mmap(mmap_4, train_x, train_y, test_x, test_y, train_batch_id)
        loss, *metrics = mmap_4.evaluate(test_x, test_y)  # TODO Accuracy might not be best, reimplement!
        _save_metrics(mmap_4_metrics, metrics)
        _save_removal_metrics(mmap_4_removal_metrics, poisoned_batch_idx, removed_batches)
        print()

        print('MC run {} complete'.format(mc_idx))
        print()

    print('Comparison results:')
    _print_metrics('Baseline 1', no_defense_1_metrics)
    _print_metrics('Sphere 2', sphere_2_metrics)
    _print_metrics('Mmap', mmap_4_metrics)
    print()
    _print_metrics('Sphere quality of removing batches', sphere_2_removal_metrics)
    _print_metrics('Mmap quality of removing batches', mmap_4_removal_metrics)

    print()
    print('Experiment 4 complete')


def main():
    gaussian0b_relative_positions()


if __name__ == '__main__':
    main()
