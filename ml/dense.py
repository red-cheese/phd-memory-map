"""
MLP model.
See https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py.
"""


from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from ml import base
from ml import metrics


class DenseNN(base.BaseModel):

    def build_model(self, name,
                    input_dim, h1_dim, h2_dim,
                    batch_size, classes):
        model = Sequential()
        model.add(Dense(h1_dim, activation='relu', input_shape=(input_dim,)))
        # model.add(Dropout(0.2))
        model.add(Dense(h2_dim, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(len(classes), activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(),
                      metrics=['accuracy',  # TODO Reimplement accuracy just to be sure
                               metrics.binary_precision,
                               metrics.binary_recall,
                               metrics.binary_f1])

        model_dir = '{}_inputdim={}_classes={}_batch={}'.format(
            name, input_dim, '-'.join([str(c) for c in classes]), batch_size)

        return model, model_dir
