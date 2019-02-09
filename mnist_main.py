"""
Memory maps for Dense DNN on the MNIST dataset with ordered classes.

Experiment design:
  - Initialise a 2-class classifier based on a Dense NN.
  - Start training the classifier on digit A only.
  - Then switch to training the classifier on digit B.
  - Monitor the memory map. All the batches must consist solely either of A or B.
"""


import mnist_get_data
import numpy as np
import random
from keras import utils
from ml import dense
from ml import evaluate

random.seed(0)
np.random.seed(0)


BATCH_SIZE = 64


def classify_2(digit_A, digit_B):
    train_A_data = mnist_get_data.get_class_data(digit_A, train=True, batch_size=BATCH_SIZE, max_num=3000)
    train_A_labels = np.zeros(shape=(len(train_A_data),), dtype=np.int64)
    train_A_labels[:] = digit_A
    train_A_labels = utils.to_categorical(train_A_labels, 2)
    # test_A = mnist_get_data.get_class_data(digit_A, train=False, batch_size=BATCH_SIZE)
    # train_B = mnist_get_data.get_class_data(digit_B, train=True, batch_size=BATCH_SIZE)
    # test_B = mnist_get_data.get_class_data(digit_B, train=False, batch_size=BATCH_SIZE)

    dense_nn = dense.DenseNN(input_dim=train_A_data[0].shape[0], classes=[digit_A, digit_B], batch_size=BATCH_SIZE)
    dense_nn.fit(train_A_data, train_A_labels)


def main():
    classify_2(0, 1)


if __name__ == '__main__':
    main()
