"""
http://yann.lecun.com/exdb/mnist/
"""


import mnist
import numpy as np


RAW_DATA_DIR = './data/raw'


def get_class_data(class_label, train=True, batch_size=None, max_num=None):
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
            if max_num is not None and len(class_images) >= max_num:
                break
    # np.random.shuffle(class_images)  # TODO DO shuffle.
    print('Total number of entries:', len(class_images))

    if batch_size is not None:
        crop = len(class_images) - (len(class_images) % batch_size)
        class_images = class_images[:crop]
        print('Cropped number of entries:', len(class_images))

    return np.asarray(class_images, dtype=np.int64)


def main():
    mndata = mnist.MNIST(RAW_DATA_DIR)
    images, labels = mndata.load_training()
    print(mndata.display(images[6]))
    print(images[6])
    print(labels[6], type(labels[6]))


if __name__ == '__main__':
    main()
