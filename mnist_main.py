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
from ml import dense
from ml import evaluate


def classify_2(digit_A, digit_B):
    train_A = mnist_get_data.get_class_data(digit_A, train=True)
    test_A = mnist_get_data.get_class_data(digit_A, train=False)
    train_B = mnist_get_data.get_class_data(digit_B, train=True)
    test_B = mnist_get_data.get_class_data(digit_B, train=False)

    dense_nn = dense.DenseNN(input_dim=train_A[0].shape[0], classes=[digit_A, digit_B])
    dense_nn.fit(train_A, np.asarray([digit_A] * len(train_A)))


def main():
    classify_2(0, 1)


if __name__ == '__main__':
    main()
