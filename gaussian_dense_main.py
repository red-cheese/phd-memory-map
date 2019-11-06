"""
Sample from 2 Gaussians and classify them, track mmap.
"""


import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from keras import metrics as keras_metrics
from numpy.random import multivariate_normal

from ml import dense
from ml import metrics as metrics_


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
    train_set_size = (NUM_TRAIN_BATCHES // 2) * BATCH_SIZE  # Per class.
    test_set_size = (NUM_TEST_BATCHES // 2) * BATCH_SIZE  # Per class.

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

    # Helper array to track removed batches in the mmap-based defense.
    train_batch_id = np.zeros((NUM_TRAIN_BATCHES * BATCH_SIZE,), dtype=np.int32)
    for i in range(NUM_TRAIN_BATCHES):
        train_batch_id[i * BATCH_SIZE:((i + 1) * BATCH_SIZE)] = i

    return train_x, train_y, test_x, test_y, train_batch_id


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


def _remove_batches(xs, ys, batches_to_remove, train_batch_id):
    batches_to_remove_set = set(batches_to_remove)
    to_keep = [i for i in range(xs.shape[0])
               if i // BATCH_SIZE not in batches_to_remove_set]
    to_keep_set = set(to_keep)
    removed_batches = {batch_id for j, batch_id in enumerate(train_batch_id)
                       if j not in to_keep_set}
    print('Removed bad batches:', sorted(removed_batches))
    return xs[to_keep], ys[to_keep], train_batch_id[to_keep], removed_batches


def _dist(x1, x2):
    return np.linalg.norm(x1 - x2) ** 2


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
    print('(Warn: skewed indexing!) Identified bad batches:', bad_batches)
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
    r = 0.5 * _dist(mu0, mu1)

    for i in range(NUM_TRAIN_BATCHES):
        batch_x = train_x[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)]
        batch_y = train_y[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)]
        batch_x_0 = batch_x[np.argmax(batch_y, axis=1) == 0]
        batch_x_1 = batch_x[np.argmax(batch_y, axis=1) == 1]
        num_poisoned = len([x for x in batch_x_0 if _dist(x, mu0) >= r])
        num_poisoned += len([x for x in batch_x_1 if _dist(x, mu1) >= r])
        if num_poisoned >= BATCH_SIZE * limit_flip_proba:
            removed_batch_idx.append(i)

    # Remove the identified batches.
    train_x, train_y, _, _ = _remove_batches(train_x, train_y,
                                             removed_batch_idx, train_batch_id)

    # Fit the model.
    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=NUM_EPOCHS)

    return dense_nn, removed_batch_idx


def _run_model_with_slab(dense_nn, train_x, train_y, test_x, test_y,
                         train_batch_id, limit_flip_proba):
    pass


def _run_model_with_mmap(dense_nn, train_x, train_y, test_x, test_y, train_batch_id):
    """
    Helper method. Applies the mmap method to clean the data. Returns the final
    trained model and the indices of removed batches.

    Basically this is what happens in Experiment 3 (see above).
    """

    # All removed batches (original indexing).
    removed_batch_idx = []

    dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                 num_epochs=NUM_EPOCHS)

    last_mmap = dense_nn.epoch_mmaps[-1]
    bad_batch_idx = _find_bad_batch(last_mmap)

    # Remove bad batches and recalibrate until no more bad batches are found.
    while bad_batch_idx:
        train_x, train_y, train_batch_id, removed_batches = _remove_batches(train_x, train_y, bad_batch_idx, train_batch_id)
        removed_batch_idx.extend(removed_batches)

        print('Recalibrating...')
        # Recalibrate for 1 epoch and look at the last mmap again.
        dense_nn.fit(train_x, train_y, validation_data=(test_x, test_y),
                     num_epochs=1, continue_training=True)
        print('Recalibration done')

        last_mmap = dense_nn.epoch_mmaps[-1]
        bad_batch_idx = _find_bad_batch(last_mmap)

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

        train_x, train_y, test_x, test_y, train_batch_id = _basic_train_test()

        # Randomly select which batches to poison.
        poisoned_batch_idx = np.random.choice(np.arange(NUM_TRAIN_BATCHES),
                                              size=num_poisoned_batches,
                                              replace=False)
        # Perform poisoning.
        for batch_i in poisoned_batch_idx:
            start, end = batch_i * BATCH_SIZE, (batch_i + 1) * BATCH_SIZE
            train_y = _flip_labels(train_y, start, end, flip_proba)

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
    gaussian4()


if __name__ == '__main__':
    main()
