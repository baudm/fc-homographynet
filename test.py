#!/usr/bin/env python

import os.path
import sys

from homographynet import data
from homographynet.models import create_models
from homographynet.losses import mean_corner_error

import numpy as np

def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [trained model.h5]'.format(name))
        exit(1)

    train_model, test_model = create_models(use_weights=True)

    if len(sys.argv) == 2:
        train_model.load_weights(sys.argv[1])

    train_model.summary()

    batch_size = 64 * 2

    loader = data.loader(data.TEST_PATH, batch_size, shuffle=True, mode='test')
    steps = int(data.TEST_SAMPLES / batch_size)

    # Optimizer doesn't matter in this case, we just want to set the loss and metrics
    test_model.compile('sgd', loss='mean_squared_error', metrics=[mean_corner_error])
    evaluation = test_model.evaluate_generator(loader, steps)
    print('Test loss:', evaluation)
    loader = data.loader(data.TEST_PATH, 1, shuffle=True, mode='demo')
    (patches, corners, images), offsets = next(loader)

    import matplotlib.pyplot as plt

    pred = train_model.predict_on_batch([patches, corners, images])
    p = test_model.predict_on_batch(patches)
    print('pred:', p*32.)
    print('gt:', offsets*32.)

    patches = (patches + 1.) / 2.
    patches = np.clip(patches, 0., 1.)

    pred = (pred + 1.) / 2.
    pred = np.clip(pred, 0., 1.)

    plt.subplot(311)
    plt.imshow(patches[0, :, :, 0], cmap='gray')

    plt.subplot(312)
    plt.imshow(patches[0, :, :, 1], cmap='gray')

    plt.subplot(313)
    plt.imshow(pred[0].squeeze(), cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
