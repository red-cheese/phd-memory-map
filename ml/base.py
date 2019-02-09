

import os
from ml import mmap


class BaseModel:

    _DEFAULT_BATCH_SIZE = 64
    _DEFAULT_NUM_EPOCHS = 10

    def __init__(self, *args, **kwargs):
        self.model, self.model_dir = self.build_model(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, x, y, batch_size=_DEFAULT_BATCH_SIZE, num_epochs=_DEFAULT_NUM_EPOCHS):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        mmap_callback = mmap.MemoryMap(
            all_data=x, all_labels=y, model=self.model,
            batch_size=batch_size, model_dir=self.model_dir)
        self.model.fit(x=x, y=y, batch_size=batch_size, verbose=1,
                       callbacks=[mmap_callback], epochs=num_epochs,
                       # shuffle=False for memory maps!
                       shuffle=False)

        model_filepath = os.path.join(self.model_dir, '1.model')
        self.model.save_weights(model_filepath, overwrite=True)

    def evaluate(self, x, y):
        self.model.evaluate(x, y)
