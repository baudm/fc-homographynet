#!/usr/bin/env python

import os.path
import sys

import math

from keras.callbacks import ModelCheckpoint

from homographynet import data
from homographynet.losses import image_consistency
from homographynet.models import create_models


def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [existing model.h5]'.format(name))
        exit(1)

    model = create_models()[0]
    if len(sys.argv) == 2:
        model.load_weights(sys.argv[1])

    # Configuration
    batch_size = 64
    target_iterations = 2 * 90000 # at batch_size = 64

    model.compile(optimizer='nadam', loss=image_consistency)
    model.summary()

    save_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = ModelCheckpoint(os.path.join(save_path, 'model.{epoch:02d}.h5'), save_weights_only=True)

    # In the paper, the 90,000 iterations was for batch_size = 64
    # So scale appropriately
    target_iterations = 64 * target_iterations // batch_size
    # As stated in Keras docs
    steps_per_epoch = data.TRAIN_SAMPLES // batch_size
    epochs = int(math.ceil(target_iterations / steps_per_epoch))

    loader = data.loader(data.TRAIN_PATH, batch_size)

    val_loader = data.loader(data.TEST_PATH, batch_size)
    val_steps = data.TEST_SAMPLES // batch_size

    # Train
    model.fit_generator(loader, steps_per_epoch, epochs,
                        callbacks=[checkpoint],
                        validation_data=val_loader, validation_steps=val_steps)


if __name__ == '__main__':
    main()
