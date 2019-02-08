

import os
from keras import models
from ml import mmap


class BaseModel(models.Model):

    _DEFAULT_BATCH_SIZE = 64

    def __init__(self, batch_size=None):
        super(BaseModel, self).__init__()
        self.batch_size = batch_size if batch_size is not None else self._DEFAULT_BATCH_SIZE

    @property
    def model_dir(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def fit(self, x=None, y=None, batch_size=64, **kwargs):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        mmap_callback = mmap.MemoryMap(all_data=x, all_labels=y, model=self)
        super(BaseModel, self).fit(x=x, y=y, batch_size=batch_size, verbose=1,
                                   callbacks=[mmap_callback],
                                   **kwargs)

        model_filepath = os.path.join(self.model_dir, '1.model')
        self.save_weights(model_filepath, overwrite=True)

    def save(self, filepath, overwrite=True, include_optimizer=True):
        pass

    def calc_loss(self, data, labels):
        return self.evaluate(data, labels)[0]  # TODO Is loss the 1st element here? Debug this!
