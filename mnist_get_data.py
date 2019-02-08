"""
http://yann.lecun.com/exdb/mnist/
"""


import mnist
import numpy as np


RAW_DATA_DIR = './data/raw'


def get_class_data(class_label, train=True):
    print('Get data for class', class_label, '; train =', train)

    mndata = mnist.MNIST(RAW_DATA_DIR)
    if train:
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()

    class_images = []
    for img, label in zip(images, labels):
        if label == class_label:
            class_images.append(np.array(img, dtype=np.int64))
    # np.random.shuffle(class_images)  # TODO DO shuffle.

    print('Number of entries:', len(class_images))
    return class_images


def main():
    mndata = mnist.MNIST(RAW_DATA_DIR)
    images, labels = mndata.load_training()
    print(mndata.display(images[6]))
    print(images[6])
    print(labels[6], type(labels[6]))


if __name__ == '__main__':
    main()
