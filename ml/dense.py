"""
MLP model.
See https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py.
"""


from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from ml import base


class DenseNN(base.BaseModel):

    def build_model(self, input_dim, batch_size, classes):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
        # model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(len(classes), activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        model_dir = 'Dense_inputdim={}_classes={}_batch={}'.format(
            input_dim, '-'.join([str(c) for c in classes]), batch_size)

        return model, model_dir
