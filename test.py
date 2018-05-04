#!/usr/bin/env python

import os.path
import sys

from keras.models import load_model

from homographynet import data
from homographynet.models import create_mobilenet_model as create_model
from homographynet.losses import mean_corner_error

import numpy as np

def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [trained model.h5]'.format(name))
        exit(1)

    orig, model = create_model(use_weights=True)

    if len(sys.argv) == 2:
        model.load_weights(sys.argv[1])

    model.summary()

    batch_size = 64 * 2

    loader = data.loader(data.TEST_PATH, 1, shuffle=True, test=True)
    steps = int(data.TEST_SAMPLES / batch_size)

    # Optimizer doesn't matter in this case, we just want to set the loss and metrics
    model.compile('sgd', loss='mean_squared_error', metrics=[mean_corner_error])
    #evaluation = model.evaluate_generator(loader, steps)
    #print('Test loss:', evaluation)
    (patches, corners, images), (targets, offsets) = next(loader)

    import matplotlib.pyplot as plt

    pred = model.predict_on_batch([patches, corners, images])
    p = orig.predict_on_batch(patches)
    print('pred:', p*32.)
    print('gt:', offsets)

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
