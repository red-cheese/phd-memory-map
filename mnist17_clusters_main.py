import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import fowlkes_mallows_score, silhouette_score

from ml import dense
from mnist17.constants import *
from mnist17 import dataset, poisonings


NUM_ITER = 3  # TODO
MAX_NUM_CL = 10

COLOURS = ['royalblue', 'salmon', 'seagreen', 'magenta', 'turquoise', 'darkviolet', 'gold', 'lime', 'pink', 'coral', 'darkred', 'blue']


def plot_metrics_by_num_clusters(exp_dir, epoch_metrics):
    assert epoch_metrics[0][-1].shape[0] == NUM_EPOCHS
    xvalues = np.arange(MAX_NUM_CL) + 1

    for epoch_idx in range(NUM_EPOCHS):
        plt.title('Clustering metrics by number of clusters - epoch {}'.format(epoch_idx + 1))
        plt.xlabel('Num clusters')
        plt.ylabel('Value')
        plt.xticks(xvalues)
        for i, (metric_name, values) in enumerate(epoch_metrics):
            metric_mean = values[epoch_idx].mean(axis=1)
            metric_std = values[epoch_idx].std(axis=1)
            plt.plot(xvalues, metric_mean, label=metric_name, c=COLOURS[i])
            plt.plot(xvalues, metric_mean - metric_std, c=COLOURS[i], linestyle='dashed')
            plt.plot(xvalues, metric_mean + metric_std, c=COLOURS[i], linestyle='dashed')
        plt.legend()
        plt.savefig(os.path.join(exp_dir, 'cluster_metrics_epoch{}'.format(epoch_idx + 1)), dpi=150)
        plt.gcf().clear()


def plot_mmap_pca(mmap_pca, epoch, num_cl, cluster_labels, comp_x, comp_y, plot_dir, true=False):
    plt.title('Mmap PCA - components {} and {} - epoch {}'
              .format(comp_x + 1, comp_y + 1, epoch + 1))
    plt.xlabel('Component {}'.format(comp_x + 1))
    plt.ylabel('Component {}'.format(comp_y + 1))
    for i in range(num_cl):
        c_mask = cluster_labels == i
        plt.scatter(mmap_pca[c_mask, comp_x], mmap_pca[c_mask, comp_y],
                    c=COLOURS[i], marker='o', s=1, label='Cluster {}'.format(i + 1))
    plt.legend()
    file_path = os.path.join(plot_dir,
                             'cl{}_epoch{}_{}_mmap_pca_{}-{}.png'.format(
                                 num_cl, epoch + 1, 'true' if true else 'pred', comp_x + 1, comp_y + 1))
    plt.savefig(file_path, dpi=150)
    plt.gcf().clear()


def mnist17_cl_02(poisoning_type):  # E.g. 'c_noise'
    exp_name = 'mnist17_cl_02{}'.format(poisoning_type)
    os.makedirs(exp_name, exist_ok=False)

    # Reported metrics, by epoch: (epoch index, number of clusters - 1, iteration index) -> value
    silhouette_by_num_cl = np.zeros(shape=(NUM_EPOCHS, MAX_NUM_CL, NUM_ITER))
    fowlkes_mallows_by_num_cl = np.zeros(shape=(NUM_EPOCHS, MAX_NUM_CL, NUM_ITER))

    for iter_ in range(NUM_ITER):
        print('Iteration', iter_)
        _set_seed(iter_)

        iter_dir = os.path.join(exp_name, 'iter{}'.format(iter_))
        os.makedirs(iter_dir, exist_ok=False)

        train_x, train_y, test_x, test_y, train_batch_id = dataset.basic_train_test()
        if poisoning_type == 'a_base':
            train_x, train_y, true_batch_labels = poisonings.base(train_x, train_y)
        elif poisoning_type == 'b_flip':
            train_x, train_y, true_batch_labels = poisonings.flip_poisoning(train_x, train_y)
        elif poisoning_type == 'c_noise':
            train_x, train_y, true_batch_labels = poisonings.noise_poisoning(train_x, train_y, plot_dir=iter_dir, plot=True)
        # elif poisoning_type == 'd_merge':
        #     pass
        else:
            raise ValueError('Unknown poisoning type: {}'.format(poisoning_type))

        dense_nn = dense.DenseNN(parent_dir=iter_dir,
                                 name=iter_dir,
                                 input_dim=INPUT_DIM, h1_dim=H1_DIM, h2_dim=H2_DIM,
                                 classes=(1, 7), batch_size=BATCH_SIZE,
                                 mmap_normalise=False)

        for epoch in range(NUM_EPOCHS):
            dense_nn.fit(train_x, train_y,
                         validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)
            # Model is loaded during fit(). Start from scratch every time.
            assert len(dense_nn.epoch_mmaps) == epoch + 1

            mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
            assert mmap.shape[0] == NUM_TRAIN_BATCHES

            # Clusterise.
            pca = PCA(n_components=3)
            mmap_pca = pca.fit_transform(mmap)

            zero_labels = np.zeros(shape=(NUM_TRAIN_BATCHES,))
            fowlkes_mallows_by_num_cl[epoch, 0, iter_] = fowlkes_mallows_score(true_batch_labels, zero_labels)

            for n_clusters in range(2, MAX_NUM_CL + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=iter_).fit(mmap_pca)

                # 2 first PCs are enough.
                plot_mmap_pca(mmap_pca, epoch, n_clusters, kmeans.labels_, 0, 1, iter_dir, true=False)
                # TODO n_clusters
                plot_mmap_pca(mmap_pca, epoch, n_clusters, true_batch_labels, 0, 1, iter_dir, true=True)

                # Compute Silhouette.
                silhouette_by_num_cl[epoch, n_clusters - 1, iter_] = silhouette_score(mmap_pca, kmeans.labels_)
                # Compute Fowlkes-Mallows index.
                fowlkes_mallows_by_num_cl[epoch, n_clusters - 1, iter_] = fowlkes_mallows_score(true_batch_labels, kmeans.labels_)


    # Report metric.
    plot_metrics_by_num_clusters(exp_name,
                                 [
                                     ('Silhouette score', silhouette_by_num_cl),
                                     ('Fowlkes-Mallows score', fowlkes_mallows_by_num_cl)
                                 ])


def main():
    _set_seed(0)

    mnist17_cl_02(poisoning_type='a_base')
    mnist17_cl_02(poisoning_type='b_flip')
    mnist17_cl_02(poisoning_type='c_noise')


def _set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_random_seed(seed_value)
    tf.set_random_seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == '__main__':
    main()
