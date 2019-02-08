"""
MLP model.
See https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py.
"""


from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from ml import base


class DenseNN(base.BaseModel):

    def __init__(self, input_dim, classes, batch_size=None):
        super(DenseNN, self).__init__(batch_size=batch_size)
        self.input_dim = input_dim
        self.classes = classes
        self.num_classes = len(self.classes)

    @property
    def model_dir(self):
        return 'Dense_inputdim={}_classes={}_batch={}'.format(
            self.input_dim, '-'.join([str(c) for c in self.classes]), self.batch_size)

    def build_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(self.input_dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
