

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture

import utils_common
import utils_mnist
from constants_mnist import *
from ml import dense


MPL_COLOURS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def mnist17_01a():
    parent_dir = 'mnist17_01a'
    experiment_dir = parent_dir

    os.makedirs(experiment_dir, exist_ok=True)

    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    pred_cluster_labels_by_epoch = []  # Just for 2 clusters.
    cluster_names = ('No poisoning', 'Poisoned')
    cluster_colours = ('blue', 'red')
    mmap_pca_by_epoch = []

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        pca = PCA(n_components=2)
        mmap_pca = pca.fit_transform(mmap)
        mmap_pca_by_epoch.append(mmap_pca)
        explained_var = [round(val, 5) for val in pca.explained_variance_]
        singular_values = [round(val, 5) for val in pca.singular_values_]

        # Try to cluster into 2 clusters.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(mmap_pca)
        pred_cluster_masks = [kmeans.labels_ == i for i in range(2)]
        pred_cluster_labels_by_epoch.append(kmeans.labels_)

        # Components 1 and 2.
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 0, 1, model_dir, is_true=False)

    utils_common.cluster_analysis_2(mmap_pca_by_epoch, pred_cluster_labels_by_epoch, cluster_names, cluster_colours,
                                    model_dir, NUM_EPOCHS)

    # Save the final model to perform model-based poisoning later.
    dense_nn.model.save(experiment_dir + '/MNIST_for_poison.h5')

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def mnist17_01b_flip():
    parent_dir = 'mnist17_01b'
    experiment_dir = parent_dir

    os.makedirs(experiment_dir, exist_ok=True)

    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()
    # [(flip probability, [(start of batch, end of batch - exclusive)])]
    flip_settings = [
        (0.1, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.25, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.5, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
    ]
    for flip_proba, poisoned_batch_settings in flip_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_y = utils_common.flip_labels(train_y, start_idx, end_idx, flip_proba, BATCH_SIZE, copy=False)

    poisoned_batch_mask_01 = np.zeros(shape=(NUM_TRAIN_BATCHES,), dtype=np.bool)
    poisoned_batch_mask_025 = np.zeros(shape=(NUM_TRAIN_BATCHES,), dtype=np.bool)
    poisoned_batch_mask_05 = np.zeros(shape=(NUM_TRAIN_BATCHES,), dtype=np.bool)
    poisoned_batch_mask_01[10:15] = True
    poisoned_batch_mask_01[66:71] = True
    poisoned_batch_mask_025[30:35] = True
    poisoned_batch_mask_025[46:51] = True
    poisoned_batch_mask_05[39:42] = True
    poisoned_batch_mask = poisoned_batch_mask_01 + poisoned_batch_mask_025 + poisoned_batch_mask_05
    assert sum(poisoned_batch_mask) == 23  # 23 poisoned batches in total.

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    pred_cluster_labels_by_epoch = []  # Just for 2 clusters.
    cluster_names = ('No poisoning', 'Poisoned')
    cluster_colours = ('blue', 'red')
    mmap_pca_by_epoch = []

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        pca = PCA(n_components=2)
        mmap_pca = pca.fit_transform(mmap)
        mmap_pca_by_epoch.append(mmap_pca)
        explained_var = [round(val, 5) for val in pca.explained_variance_]
        singular_values = [round(val, 5) for val in pca.singular_values_]

        # Try to cluster into 2 clusters.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(mmap_pca)
        pred_cluster_masks = [kmeans.labels_ == i for i in range(2)]
        pred_cluster_labels_by_epoch.append(kmeans.labels_)

        # Components 1 and 2.
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 0, 1, model_dir, is_true=False)

    utils_common.cluster_analysis_2(mmap_pca_by_epoch, pred_cluster_labels_by_epoch, cluster_names, cluster_colours,
                                    model_dir, NUM_EPOCHS)

    # Save the final model to perform model-based poisoning later.
    dense_nn.model.save(experiment_dir + '/MNIST_for_poison.h5')

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def mnist17_01c_noise():
    parent_dir = 'mnist17_01c'
    experiment_dir = parent_dir

    os.makedirs(experiment_dir, exist_ok=True)

    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()
    # [(Gaussian noise STD, [(start of batch, end of batch - exclusive)])]
    noise_settings = [
        (0.3, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.5, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.7, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
    ]
    for noise_std, poisoned_batch_settings in noise_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_x = utils_common.noise(train_x, start_idx, end_idx, noise_std, experiment_dir)

    poisoned_batch_mask_01 = np.zeros(shape=(NUM_TRAIN_BATCHES,), dtype=np.bool)
    poisoned_batch_mask_025 = np.zeros(shape=(NUM_TRAIN_BATCHES,), dtype=np.bool)
    poisoned_batch_mask_05 = np.zeros(shape=(NUM_TRAIN_BATCHES,), dtype=np.bool)
    poisoned_batch_mask_01[10:15] = True
    poisoned_batch_mask_01[66:71] = True
    poisoned_batch_mask_025[30:35] = True
    poisoned_batch_mask_025[46:51] = True
    poisoned_batch_mask_05[39:42] = True
    poisoned_batch_mask = poisoned_batch_mask_01 + poisoned_batch_mask_025 + poisoned_batch_mask_05
    assert sum(poisoned_batch_mask) == 23  # 23 poisoned batches in total.

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    true_cluster_labels = [~poisoned_batch_mask, poisoned_batch_mask_01, poisoned_batch_mask_025, poisoned_batch_mask_05]
    pred_cluster_labels_by_epoch = []  # Just for 2 clusters.
    cluster_names = ('No poisoning', 'Poisoned')
    cluster_colours = ('blue', 'red')
    mmap_pca_by_epoch = []

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        pca = PCA(n_components=3)
        mmap_pca = pca.fit_transform(mmap)
        mmap_pca_by_epoch.append(mmap_pca)
        explained_var = [round(val, 5) for val in pca.explained_variance_]
        singular_values = [round(val, 5) for val in pca.singular_values_]

        # Try to cluster into 2 clusters.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(mmap_pca)
        pred_cluster_masks = [kmeans.labels_ == i for i in range(2)]
        pred_cluster_labels_by_epoch.append(kmeans.labels_)

        # Components 1 and 2.
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 0, 1, model_dir, is_true=False)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 0, 2, model_dir, is_true=False)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 1, 2, model_dir, is_true=False)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, true_cluster_labels,
                                   ('No poisoning', 'Poisoned 01', 'Poisoned 02', 'Poisoned 03'),
                                   ('blue', 'yellow', 'magenta', 'red'),
                                   explained_var, singular_values, 0, 1, model_dir, is_true=True)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, true_cluster_labels,
                                   ('No poisoning', 'Poisoned 01', 'Poisoned 02', 'Poisoned 03'),
                                   ('blue', 'yellow', 'magenta', 'red'),
                                   explained_var, singular_values, 0, 2, model_dir, is_true=True)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, true_cluster_labels,
                                   ('No poisoning', 'Poisoned 01', 'Poisoned 02', 'Poisoned 03'),
                                   ('blue', 'yellow', 'magenta', 'red'),
                                   explained_var, singular_values, 1, 2, model_dir, is_true=True)

    utils_common.cluster_analysis_2(mmap_pca_by_epoch, pred_cluster_labels_by_epoch, cluster_names, cluster_colours,
                                    model_dir, NUM_EPOCHS)

    # Save the final model to perform model-based poisoning later.
    dense_nn.model.save(experiment_dir + '/MNIST_for_poison.h5')

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def _pairwise_distances(vs):
    distances = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            distances.append(np.linalg.norm(vs[i] - vs[j]))
    return sorted(distances, reverse=True)


def _plot_distances_histogram(ds, current_epoch, model_dir):
    # Plot the histogram of distances.
    plt.hist(ds, bins=50, density=True, alpha=0.7)
    plt.title('Pairwise distances - epoch {}'.format(current_epoch))
    plt.grid(True)
    plt.savefig(model_dir + '/dist_epoch{}.png'.format(current_epoch), dpi=150)
    plt.gcf().clear()


def _plot_eigenvalues_by_epoch(eig_by_epoch, model_dir):
    all_eig_values = list(zip(*eig_by_epoch))

    colours = plt.cm.get_cmap('hsv', len(all_eig_values))

    plt.title('All eigenvalues by epoch')
    for i, eig_values in enumerate(all_eig_values):
        plt.plot(eig_values, label='Eig #{}'.format(i + 1), c=colours(i))
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(model_dir + '/eig_all.png', dpi=150)
    plt.gcf().clear()

    plt.title('Largest eigenvalue by epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.plot(all_eig_values[0], c=colours(0))
    plt.savefig(model_dir + '/eig_first.png', dpi=150)
    plt.gcf().clear()

    plt.title('All eigenvalues except the largest - by epoch')
    for i, eig_values in enumerate(all_eig_values):
        if i == 0:
            continue
        plt.plot(eig_values, label='Eig #{}'.format(i + 1), c=colours(i))
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(model_dir + '/eig_rest.png', dpi=150)
    plt.gcf().clear()


def _compute_wasserstein_dist(mx):
    assert mx.shape[0] == mx.shape[1]
    res = np.zeros(shape=mx.shape)
    for i in range(mx.shape[0]):
        for j in range(mx.shape[1]):
            res[i][j] = wasserstein_distance(mx[i], mx[j])
    return res


def _analyse_pairwise(pw_distances, model_dir, current_epoch):
    # Note: also tried to compute EMD just on the mmap (no pca and no PW distances, i.e using mmap as pw_distances),
    # but the results were much noisier.
    # TODO: symmetric_kl_div in addition to wasserstein_dist
    wasserstein_dist = _compute_wasserstein_dist(pw_distances)
    wasserstein_dist_means = np.mean(wasserstein_dist, axis=1)
    labels = np.arange(1, NUM_TRAIN_BATCHES + 1)
    plt.title('Mean Wasserstein distance to all other batches')
    plt.xlabel('Mini-batch')
    plt.ylabel('Value')
    clustering = MeanShift().fit(wasserstein_dist_means.reshape(-1, 1)).labels_
    for i, cluster in enumerate(sorted(set(clustering))):
        plt.scatter(labels[clustering == cluster], wasserstein_dist_means[clustering == cluster], s=1,
                    label='MeanShift Cluster {}'.format(i), c=MPL_COLOURS[i])
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'wass_means_epoch{}.png'.format(current_epoch)), dpi=150)
    plt.gcf().clear()


def mnist17_02a_base_distances(seed_value):
    parent_dir = 'mnist17_02a_base_distances_v3_{}'.format(seed_value)
    experiment_dir = parent_dir
    os.makedirs(experiment_dir, exist_ok=True)
    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()
    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    eigenvalues_by_epoch = []  # Eigenvalues of the covariance matrix, obtained from PCA.

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        # 3-component PCA for pairwise distances.
        pca = PCA(n_components=3)
        mmap_pca = pca.fit_transform(mmap)

        pw_distances = pairwise_distances(mmap_pca)
        _analyse_pairwise(pw_distances, model_dir, current_epoch)

        gmm1 = GaussianMixture(n_components=1, covariance_type='full')
        gmm1.fit(mmap_pca)
        gmm2 = GaussianMixture(n_components=2, covariance_type='full')
        gmm2.fit(mmap_pca)
        utils_common.plot_mmap_pca_gmm2(mmap_pca, 0, 1, gmm1, gmm2, current_epoch, model_dir)

        # Components 1 and 2.
        utils_common.plot_mmap_pca_simple(mmap_pca, 0, 1, current_epoch, model_dir)

        # 10-component PCA for eigenvalues.
        pca10 = PCA(n_components=10)
        mmap_pca10 = pca10.fit_transform(mmap)
        eigenvalues_by_epoch.append([round(val, 5) for val in pca10.explained_variance_])

    _plot_eigenvalues_by_epoch(eigenvalues_by_epoch, model_dir)

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def mnist17_02b_flip_distances():
    parent_dir = 'mnist17_02b_flip_distances_v3'
    experiment_dir = parent_dir
    os.makedirs(experiment_dir, exist_ok=True)
    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()

    # [(flip probability, [(start of batch, end of batch - exclusive)])]
    flip_settings = [
        (0.1, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.25, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.5, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
    ]
    for flip_proba, poisoned_batch_settings in flip_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_y = utils_common.flip_labels(train_y, start_idx, end_idx, flip_proba, BATCH_SIZE, copy=False)

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    eigenvalues_by_epoch = []  # Eigenvalues of the covariance matrix, obtained from PCA.

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        # 3-component PCA for pairwise distances.
        pca = PCA(n_components=3)
        mmap_pca = pca.fit_transform(mmap)

        pw_distances = pairwise_distances(mmap_pca)
        _analyse_pairwise(pw_distances, model_dir, current_epoch)

        gmm1 = GaussianMixture(n_components=1, covariance_type='full')
        gmm1.fit(mmap_pca)
        gmm2 = GaussianMixture(n_components=2, covariance_type='full')
        gmm2.fit(mmap_pca)
        utils_common.plot_mmap_pca_gmm2(mmap_pca, 0, 1, gmm1, gmm2, current_epoch, model_dir)

        # Components 1 and 2.
        utils_common.plot_mmap_pca_simple(mmap_pca, 0, 1, current_epoch, model_dir)

        # 10-component PCA for eigenvalues.
        pca10 = PCA(n_components=10)
        mmap_pca10 = pca10.fit_transform(mmap)
        eigenvalues_by_epoch.append([round(val, 5) for val in pca10.explained_variance_])

    _plot_eigenvalues_by_epoch(eigenvalues_by_epoch, model_dir)

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def mnist17_02c_noise_distances(seed_value):
    parent_dir = 'mnist17_02c_noise_distances_v3_{}'.format(seed_value)
    experiment_dir = parent_dir
    os.makedirs(experiment_dir, exist_ok=True)
    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()

    # [(Gaussian noise STD, [(start of batch, end of batch - exclusive)])]
    noise_settings = [
        (0.3, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.5, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.7, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
    ]
    for noise_std, poisoned_batch_settings in noise_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_x = utils_common.noise(train_x, start_idx, end_idx, noise_std, experiment_dir)

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    eigenvalues_by_epoch = []  # Eigenvalues of the covariance matrix, obtained from PCA.

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        # 3-component PCA for pairwise distances.
        pca = PCA(n_components=3)
        mmap_pca = pca.fit_transform(mmap)

        pw_distances = pairwise_distances(mmap_pca)
        _analyse_pairwise(pw_distances, model_dir, current_epoch)

        gmm1 = GaussianMixture(n_components=1, covariance_type='full')
        gmm1.fit(mmap_pca)
        gmm2 = GaussianMixture(n_components=2, covariance_type='full')
        gmm2.fit(mmap_pca)
        utils_common.plot_mmap_pca_gmm2(mmap_pca, 0, 1, gmm1, gmm2, current_epoch, model_dir)

        # Components 1 and 2.
        utils_common.plot_mmap_pca_simple(mmap_pca, 0, 1, current_epoch, model_dir)

        # 10-component PCA for eigenvalues.
        pca10 = PCA(n_components=10)
        mmap_pca10 = pca10.fit_transform(mmap)
        eigenvalues_by_epoch.append([round(val, 5) for val in pca10.explained_variance_])

    _plot_eigenvalues_by_epoch(eigenvalues_by_epoch, model_dir)

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def mnist17_02d_merge_distances(seed_value):
    parent_dir = 'mnist17_02d_merge_distances_v3_{}'.format(seed_value)
    experiment_dir = parent_dir
    os.makedirs(experiment_dir, exist_ok=True)
    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()

    # A certain percentage of 1s in these batches will be merged with 7s.
    merge_settings = [
        (0.1, [(10 * BATCH_SIZE, 15 * BATCH_SIZE), (66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.5, [(30 * BATCH_SIZE, 35 * BATCH_SIZE), (46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.9, [(39 * BATCH_SIZE, 42 * BATCH_SIZE)]),
    ]
    for merge_rate, poisoned_batch_settings in merge_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_x = utils_common.merge(train_x, train_y, start_idx, end_idx, merge_rate, experiment_dir)

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    eigenvalues_by_epoch = []  # Eigenvalues of the covariance matrix, obtained from PCA.

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        # 3-component PCA for pairwise distances.
        pca = PCA(n_components=3)
        mmap_pca = pca.fit_transform(mmap)

        pw_distances = pairwise_distances(mmap_pca)
        _analyse_pairwise(pw_distances, model_dir, current_epoch)

        gmm1 = GaussianMixture(n_components=1, covariance_type='full')
        gmm1.fit(mmap_pca)
        gmm2 = GaussianMixture(n_components=2, covariance_type='full')
        gmm2.fit(mmap_pca)
        utils_common.plot_mmap_pca_gmm2(mmap_pca, 0, 1, gmm1, gmm2, current_epoch, model_dir)

        # Components 1 and 2.
        utils_common.plot_mmap_pca_simple(mmap_pca, 0, 1, current_epoch, model_dir)

        # 10-component PCA for eigenvalues.
        pca10 = PCA(n_components=10)
        mmap_pca10 = pca10.fit_transform(mmap)
        eigenvalues_by_epoch.append([round(val, 5) for val in pca10.explained_variance_])

    _plot_eigenvalues_by_epoch(eigenvalues_by_epoch, model_dir)

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def mnist17_02e_two_poison_distances():
    parent_dir = 'mnist17_02e_two_poison_distances_v2'
    experiment_dir = parent_dir
    os.makedirs(experiment_dir, exist_ok=True)
    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()

    # 1st type of poisoning: merge.
    merge_settings = [
        (0.1, [(10 * BATCH_SIZE, 15 * BATCH_SIZE)]),
        (0.5, [(46 * BATCH_SIZE, 51 * BATCH_SIZE)]),
        (0.9, [(39 * BATCH_SIZE, 41 * BATCH_SIZE)]),
    ]
    for merge_rate, poisoned_batch_settings in merge_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_x = utils_common.merge(train_x, train_y, start_idx, end_idx, merge_rate, experiment_dir)

    # 2nd type of poisoning: noise.
    noise_settings = [
        (0.3, [(66 * BATCH_SIZE, 71 * BATCH_SIZE)]),
        (0.5, [(30 * BATCH_SIZE, 35 * BATCH_SIZE)]),
        (0.7, [(41 * BATCH_SIZE, 43 * BATCH_SIZE)]),
    ]
    for noise_std, poisoned_batch_settings in noise_settings:
        for start_idx, end_idx in poisoned_batch_settings:
            train_x = utils_common.noise(train_x, start_idx, end_idx, noise_std, experiment_dir)

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    pairwise_distances_by_epoch = []
    eigenvalues_by_epoch = []  # Eigenvalues of the covariance matrix, obtained from PCA.

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        # 3-component PCA for pairwise distances.
        pca = PCA(n_components=3)
        mmap_pca = pca.fit_transform(mmap)

        dist = _pairwise_distances(mmap_pca)
        pairwise_distances_by_epoch.append(dist)

        pw_distances = pairwise_distances(mmap_pca)
        _analyse_pairwise(pw_distances, model_dir, current_epoch)

        # Components 1 and 2.
        utils_common.plot_mmap_pca_simple(mmap_pca, 0, 1, current_epoch, model_dir)

        # Pairwise distances.
        _plot_distances_histogram(dist, current_epoch, model_dir)
        # Largest 20 pairwise distances.
        with open(model_dir + '/dist_epoch{}_largest20.txt'.format(current_epoch), 'w') as f:
            f.write('\n'.join([str(d) for d in dist[:20]]))

        # 10-component PCA for eigenvalues.
        pca10 = PCA(n_components=10)
        mmap_pca10 = pca10.fit_transform(mmap)
        eigenvalues_by_epoch.append([round(val, 5) for val in pca10.explained_variance_])

    _plot_eigenvalues_by_epoch(eigenvalues_by_epoch, model_dir)

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def main():
    for seed_value in [0, 1, 2, 3, 4]:
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        # 2. Set `python` built-in pseudo-random generator at a fixed value
        import random
        random.seed(seed_value)

        # 3. Set `numpy` pseudo-random generator at a fixed value
        import numpy as np
        np.random.seed(seed_value)

        # 4. Set the `tensorflow` pseudo-random generator at a fixed value
        import tensorflow as tf
        tf.random.set_random_seed(seed_value)
        tf.set_random_seed(seed_value)
        # for later versions:
        # tf.compat.v1.set_random_seed(seed_value)

        # 5. Configure a new global `tensorflow` session
        from keras import backend as K
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        # for later versions:
        # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        # tf.compat.v1.keras.backend.set_session(sess)

        mnist17_02a_base_distances(seed_value)
        mnist17_02c_noise_distances(seed_value)
        mnist17_02d_merge_distances(seed_value)


if __name__ == '__main__':
    main()
