

import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


class MemoryMap(keras.callbacks.Callback):
    """
    Memory map of batch losses within one epoch.
    """

    _MAX_LOSS = 2000
    _MIN_LOSS = -2000

    def __init__(self, all_data, all_labels, model, batch_size, model_dir,
                 norm=True, epochs_done=0):
        assert len(all_data) == len(all_labels)
        assert len(all_data) % batch_size == 0

        super(MemoryMap, self).__init__()

        self.batch_size = batch_size
        num_batches = int(len(all_data) / self.batch_size)
        self.K = num_batches

        self.model = model
        self.all_data = all_data
        self.all_labels = all_labels

        # Rows: batches (fixed).
        # Columns: losses after each gradient step.
        # self.mmap[i, j] = loss on batch i after gradient update on batch j has been done.
        self.mmap = np.zeros(shape=(self.K, self.K))

        # Only relevant for binary classification.
        self.mmap0 = np.zeros(shape=(self.K, self.K))
        self.mmap1 = np.zeros(shape=(self.K, self.K))

        self.cur_batch_id = -1  # Will start with the mini-batch 0.
        self.cur_epoch_id = epochs_done - 1

        self.mmap_dir = os.path.join(model_dir, 'mmap')
        if not os.path.isdir(self.mmap_dir):
            os.makedirs(self.mmap_dir)

        self.norm = norm

        self.epoch_mmaps = []

    def on_batch_begin(self, batch, logs=None):
        self.cur_batch_id += 1

    def on_batch_end(self, batch, logs=None):
        for i in range(self.K):
            batch_start = i * self.batch_size
            batch_end = batch_start + self.batch_size
            b = self.all_data[batch_start:batch_end, ...]
            l = self.all_labels[batch_start:batch_end]
            loss = self.model.evaluate(b, l, verbose=0, batch_size=self.batch_size)
            self.mmap[i, self.cur_batch_id] = loss[0]

            # Only relevant for binary classification.
            # l0 = l[np.argmax(l, axis=1) == 0]
            # b0 = b[np.argmax(l, axis=1) == 0]
            # l1 = l[np.argmax(l, axis=1) == 1]
            # b1 = b[np.argmax(l, axis=1) == 1]
            # loss0 = self.model.evaluate(b0, l0, verbose=0, batch_size=len(b0))
            # loss1 = self.model.evaluate(b1, l1, verbose=0, batch_size=len(b1))
            # self.mmap0[i, self.cur_batch_id] = loss0[0]
            # self.mmap1[i, self.cur_batch_id] = loss1[0]

    def on_epoch_begin(self, epoch, logs=None):
        self.cur_epoch_id += 1

    def on_epoch_end(self, epoch, logs=None):
        mmap = self.mmap
        mmap0 = self.mmap0
        mmap1 = self.mmap1

        # TODO(alex) Do for mmap0 and mmap1
        isnan = np.isnan(mmap)
        if isnan.all():
            mmap[:, :] = self._MAX_LOSS
        else:
            nanmax, nanmin = min(np.nanmax(mmap), self._MAX_LOSS), max(np.nanmin(mmap), self._MIN_LOSS)
            mmap[isnan] = nanmax
            mmap = np.clip(mmap, nanmin, nanmax)

        assert not self.norm
        if self.norm:
            # Normalise mmap so that its scale is consistent across epochs.
            # Note that the mmap is no more non-negative at this point.
            mmap_mean, mmap_std = np.mean(mmap), np.std(mmap)
            print('Epoch:', epoch, '; mean:', mmap_mean, '; std:', mmap_std)
            mmap = (mmap - mmap_mean) / mmap_std

        sns.heatmap(mmap, xticklabels=self.K // 10, yticklabels=self.K // 10, cmap="YlGnBu")
        plt.title('Mini-batch losses: epoch {}'.format(self.cur_epoch_id + 1))
        plt.ylabel('Mini-batch')
        plt.xlabel('Training step')
        plt.savefig(os.path.join(self.mmap_dir,
                                 'epoch{}.png'.format(self.cur_epoch_id + 1)),
                    dpi=150)
        plt.gcf().clear()

        # sns.heatmap(mmap0, xticklabels=self.K // 10, yticklabels=self.K // 10, cmap="YlGnBu")
        # plt.title('Mini-batch losses: epoch {}'.format(self.cur_epoch_id + 1))
        # plt.ylabel('Mini-batch')
        # plt.xlabel('Training step')
        # plt.savefig(os.path.join(self.mmap_dir,
        #                          'm0_epoch{}.png'.format(self.cur_epoch_id + 1)),
        #             dpi=150)
        # plt.gcf().clear()
        #
        # sns.heatmap(mmap1, xticklabels=self.K // 10, yticklabels=self.K // 10, cmap="YlGnBu")
        # plt.title('Mini-batch losses: epoch {}'.format(self.cur_epoch_id + 1))
        # plt.ylabel('Mini-batch')
        # plt.xlabel('Training step')
        # plt.savefig(os.path.join(self.mmap_dir,
        #                          'm1_epoch{}.png'.format(self.cur_epoch_id + 1)),
        #             dpi=150)
        # plt.gcf().clear()

        # Save the mmap in the model.
        self.epoch_mmaps.append((np.copy(self.mmap), np.copy(self.mmap0), np.copy(self.mmap1)))

        # Reset memory map for the next epoch.
        self.mmap = np.zeros(shape=(self.K, self.K))
        self.mmap0 = np.zeros(shape=(self.K, self.K))
        self.mmap1 = np.zeros(shape=(self.K, self.K))
        self.cur_batch_id = -1
