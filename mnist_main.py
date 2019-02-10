"""
Experiments with memory maps on the MNIST dataset.
"""


import mnist_get_data
import numpy as np
import random
from ml import dense
from ml import evaluate

random.seed(0)
np.random.seed(0)


BATCH_SIZE = 64


def classify_2(digit_A, digit_B):
    """
    Alternating batches that consist solely of one of the two given digits.
    10 batches of A -> 10 batches of B -> 10 of A -> 10 of B -> ...

    Experiment design:
      - Initialise a 2-class classifier based on a Dense NN.
      - Start training the classifier on digit A only.
      - Then switch to training the classifier on digit B.
      - Then switch the training back to digit A.
      - ...
      - Monitor the memory map. All the batches must consist solely either of A or B.
    """

    train_A_data = mnist_get_data.get_class_data(digit_A, train=True, batch_size=BATCH_SIZE, max_num=3000)
    train_B_data = mnist_get_data.get_class_data(digit_B, train=True, batch_size=BATCH_SIZE, max_num=3000)

    # Combine into one training dataset.
    n_batches = 10  # Will do n_batches of A, then n_batches of B, ...
    min_len = min(train_A_data.shape[0], train_B_data.shape[0])
    i = 0
    train_data = np.array([], dtype=np.int64).reshape((0, train_A_data.shape[1]))
    train_labels = np.array([], dtype=np.int64).reshape((0, 2))
    while i < min_len:
        next_i = min(min_len, i + BATCH_SIZE * n_batches)
        add_data_A = train_A_data[i:next_i, :]
        add_data_B = train_B_data[i:next_i, :]
        train_data = np.vstack((train_data, add_data_A, add_data_B))
        add_labels_A = mnist_get_data.generate_class_labels(digit_A, next_i - i, 2)
        add_labels_B = mnist_get_data.generate_class_labels(digit_B, next_i - i, 2)
        train_labels = np.vstack((train_labels, add_labels_A, add_labels_B))
        i = next_i

    dense_nn = dense.DenseNN(input_dim=train_A_data.shape[1], classes=[digit_A, digit_B], batch_size=BATCH_SIZE)
    dense_nn.fit(train_data, train_labels)


def main():
    classify_2(0, 1)


if __name__ == '__main__':
    main()
