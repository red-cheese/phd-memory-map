

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from sklearn.metrics import silhouette_score


def flip_labels(orig_labels, start_idx, end_idx, flip_proba, batch_size,
                copy=True):
    print('Flip labels in the interval [{}, {}) (batches {}-{}) with probability {}'
          .format(start_idx, end_idx, start_idx // batch_size, end_idx // batch_size, flip_proba))

    labels = np.copy(orig_labels) if copy else orig_labels

    idx = np.random.choice(np.arange(start_idx, end_idx),
                           size=int(flip_proba * (end_idx - start_idx)),
                           replace=False)
    old_labels = labels[idx, :]
    labels[idx, :] = 1 - old_labels

    return labels


def noise(x, start_idx, end_idx, noise_std, model_dir):  # TODO Move to utils_mnist
    noise_sample = normal(0., noise_std, size=x[start_idx:end_idx].shape)
    x[start_idx:end_idx] += noise_sample
    x = np.clip(x, a_min=0, a_max=1)

    f, ax = plt.subplots(2, 5, figsize=(10, 5))
    ax = ax.flatten()
    for i in range(10):
        idx = start_idx + i
        ax[i].imshow(x[idx].reshape(28, 28))
    plt.savefig('./{}/noise_{}.png'.format(model_dir, noise_std), dpi=150)
    plt.gcf().clear()

    return x


def plot_mmap_pca(mmap_pca, epoch,
                cluster_masks, cluster_names, cluster_colours, cluster_explained_var, cluster_singular_values,
                comp_x, comp_y,
                model_dir,
                is_true=True):
    plt.title('Mmap PCA - components {} and {} - epoch {}\n'
              'Explained variance: C{} {}, C{} {}\n'
              'Singular values: C{} {}, C{} {}'
              .format(comp_x + 1, comp_y + 1, epoch,
                      comp_x + 1, cluster_explained_var[comp_x], comp_y + 1, cluster_singular_values[comp_y],
                      comp_x + 1, cluster_singular_values[comp_x], comp_y + 1, cluster_singular_values[comp_y]))
    plt.xlabel('Component {}'.format(comp_x + 1))
    plt.ylabel('Component {}'.format(comp_y + 1))
    for i, (c_mask, c_name, c_colour) in enumerate(zip(cluster_masks, cluster_names, cluster_colours)):
        plt.scatter(mmap_pca[c_mask, comp_x], mmap_pca[c_mask, comp_y],
                    c=c_colour, marker='o', s=1, label=c_name)
    plt.legend()
    plt.savefig('./{}/epoch{}_{}_mmap_pca_{}-{}.png'.format(model_dir, epoch, 'true' if is_true else 'pred',
                                                            comp_x + 1, comp_y + 1), dpi=150)
    plt.gcf().clear()


def cluster_analysis_2(mmap_pca_by_epoch, pred_cluster_labels_by_epoch, cluster_names, cluster_colours, model_dir,
                       num_epochs):
    assert len(cluster_names) == len(cluster_colours) == 2

    # Silhouette score epoch to epoch.
    silhouette_scores = []
    for epoch_idx, labels in enumerate(pred_cluster_labels_by_epoch):
        s_score = silhouette_score(mmap_pca_by_epoch[epoch_idx], labels)
        silhouette_scores.append(s_score)

    # Consistency score (unweighted).
    silhouette_scores = np.array(silhouette_scores)
    ups_values = silhouette_scores[1:] - silhouette_scores[:-1]
    ups = ups_values >= 0
    consistency_score = sum(ups) / len(ups)

    # Consistency score (weighted).
    ups_w = ups_values / sum(np.abs(ups_values))
    consistency_score_weighted = np.dot(ups, ups_w)

    # Area under curve.
    auc = np.trapz(silhouette_scores)

    plt.title('Silhouette scores (2 predicted clusters) by epoch\n'
              'Consistency score - unweighted: {0:.2f}, weighted: {1:.2f}\n'
              'AUC: {2:.2f}'
              .format(consistency_score, consistency_score_weighted, auc))
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.plot(list(range(1, num_epochs + 1)), silhouette_scores, color='blue')
    plt.savefig('./{}/silhouette_2_clusters.png'.format(model_dir), dpi=150)
    plt.gcf().clear()
