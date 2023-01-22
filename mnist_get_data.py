"""
http://yann.lecun.com/exdb/mnist/
"""


from keras import utils
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
    # np.random.shuffle(class_images)
    print('Total number of entries:', len(class_images))

    if batch_size is not None:
        crop = len(class_images) - (len(class_images) % batch_size)
        class_images = class_images[:crop]
        print('Cropped number of entries:', len(class_images))

    print('Samples:')
    print(mndata.display(class_images[0]))
    print(mndata.display(class_images[24]))
    print(mndata.display(class_images[-1]))

    return np.asarray(class_images, dtype=np.int64)


def generate_class_labels(class_label, n, num_classes):
    assert class_label < num_classes
    labels = np.zeros(shape=(n,), dtype=np.int64)
    labels[:] = class_label
    labels = utils.to_categorical(labels, num_classes)
    return labels


def main():
    mndata = mnist.MNIST(RAW_DATA_DIR)
    images, labels = mndata.load_training()
    print(mndata.display(images[6]))
    print(images[6])
    print(labels[6], type(labels[6]))


if __name__ == '__main__':
    main()
