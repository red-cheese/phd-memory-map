

import os
import pickle
from ml import mmap


class BaseModel:

    _DEFAULT_BATCH_SIZE = 64
    _DEFAULT_NUM_EPOCHS = 5

    def __init__(self, *args, mmap_normalise=True, **kwargs):
        self.model, self.model_dir = self.build_model(*args, **kwargs)
        self.mmap_normalise = mmap_normalise
        self.epoch_mmaps = []

    def build_model(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, x, y, validation_data=None, batch_size=_DEFAULT_BATCH_SIZE,
            num_epochs=_DEFAULT_NUM_EPOCHS,
            # If True, will load the existing model and continue training.
            continue_training=False):
        os.makedirs(self.model_dir, exist_ok=True)

        model_filepath = os.path.join(self.model_dir, '1.model')
        mmap_filepath = os.path.join(self.model_dir, '1.mmap')

        if os.path.isfile(model_filepath):
            assert os.path.isfile(mmap_filepath)
            print('Loading model...')
            self.model.load_weights(model_filepath)
            with open(mmap_filepath, 'rb') as f:
                self.epoch_mmaps = pickle.load(f)
            print('Loading complete')

            if not continue_training:
                return

        # Continue training.
        print('Training model...')
        print('Training set shape:', x.shape)
        # Set up a new mmap callback.
        mmap_callback = mmap.MemoryMap(
            all_data=x, all_labels=y, model=self.model,
            batch_size=batch_size, model_dir=self.model_dir,
            norm=self.mmap_normalise, epochs_done=len(self.epoch_mmaps))
        self.model.fit(x=x, y=y, batch_size=batch_size, verbose=1,
                       callbacks=[mmap_callback], epochs=num_epochs,
                       validation_data=validation_data,
                       # shuffle=False for memory maps!
                       shuffle=False)
        self.epoch_mmaps.extend(mmap_callback.epoch_mmaps)
        print('Training complete')

        print('Saving weights and mmaps...')
        self.model.save_weights(model_filepath, overwrite=True)
        with open(mmap_filepath, 'wb') as f:
            pickle.dump(self.epoch_mmaps, f)
        print('Saving complete')

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
