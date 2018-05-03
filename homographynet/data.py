#!/usr/bin/env python

import os.path
import glob

import numpy as np


_SAMPLES_PER_ARCHIVE = 7680

TRAIN_PATH = '/mnt/data/datasets/homographynet/unsup/repack'
TRAIN_SAMPLES = 65 * _SAMPLES_PER_ARCHIVE

TEST_PATH = '/mnt/data/datasets/homographynet/unsup/test-set'
TEST_SAMPLES = 7 * _SAMPLES_PER_ARCHIVE


def loader(path, batch_size=64, normalize=True, shuffle=True):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        if shuffle:
            np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            patches = archive['patches']
            corners = archive['corners']
            images = archive['images']
            
            del archive

            if shuffle:
                p = np.random.permutation(len(corners))
                patches = patches[p]
                corners = corners[p]
                images = images[p]

            # Split into mini batches
            num_batches = len(corners) // batch_size
            patches = np.array_split(patches, num_batches)
            corners = np.array_split(corners, num_batches)
            images = np.array_split(images, num_batches)

            while corners:
                batch_patches = patches.pop()
                batch_corners = corners.pop()
                batch_images = images.pop()
                if normalize:
                    batch_patches = (batch_patches - 127.5) / 127.5
                    batch_images = (batch_images - 127.5) / 127.5
                yield [batch_patches, batch_corners.reshape(-1, 8, 1), batch_images.reshape(-1, 240,320, 1)], batch_patches[:, :, :, 1].reshape(-1, 128, 128, 1)
