"""
Experiments with memory maps on the MNIST dataset.
"""

# Set seeds first, even though it is practically useless because there is no
# reproducibility in keras + tf anyway.
# Also it is a mystery why the DNN can't train properly with certain seeds.
import random
random.seed(1)
import numpy as np
np.random.seed(1)

import mnist_get_data
from ml import dense
from ml import evaluate


BATCH_SIZE = 64


def obsolete_classify_2(digit_A, digit_B, num_batches_A, num_batches_B):
    """
    Alternating batches that consist solely of one of the two given digits:
        num_batches_A batches of A ->
        num_batches_B batches of B ->
        num_batches_A of A ->
        num_batches_B of B -> ...

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
    min_len = min(train_A_data.shape[0], train_B_data.shape[0])
    i_A = 0
    i_B = 0
    train_data = np.array([], dtype=np.int64).reshape((0, train_A_data.shape[1]))
    train_labels = np.array([], dtype=np.int64).reshape((0, 2))
    while i_A < min_len and i_B < min_len:
        next_i_A = min(min_len, i_A + BATCH_SIZE * num_batches_A)
        next_i_B = min(min_len, i_B + BATCH_SIZE * num_batches_B)
        add_data_A = train_A_data[i_A:next_i_A, :]
        add_data_B = train_B_data[i_B:next_i_B, :]
        train_data = np.vstack((train_data, add_data_A, add_data_B))
        add_labels_A = mnist_get_data.generate_class_labels(digit_A, next_i_A - i_A, 2)
        add_labels_B = mnist_get_data.generate_class_labels(digit_B, next_i_B - i_B, 2)
        train_labels = np.vstack((train_labels, add_labels_A, add_labels_B))
        i_A = next_i_A
        i_B = next_i_B

    dense_nn = dense.DenseNN(input_dim=train_A_data.shape[1], classes=[digit_A, digit_B], batch_size=BATCH_SIZE)
    dense_nn.fit(train_data, train_labels)

    print('=====')
    print('Evaluate:')
    print(dense_nn.evaluate(train_data, train_labels))


def main():
    obsolete_classify_2(digit_A=0, digit_B=1, num_batches_A=1, num_batches_B=1)


if __name__ == '__main__':
    main()
