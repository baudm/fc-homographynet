#!/usr/bin/env python

import os.path
import glob

import numpy as np


_SAMPLES_PER_ARCHIVE = 7680

TRAIN_PATH = '/mnt/data/datasets/homographynet/unsup/repack'
TRAIN_SAMPLES = 65 * _SAMPLES_PER_ARCHIVE

TEST_PATH = '/mnt/data/datasets/homographynet/unsup/test-set'
TEST_SAMPLES = 7 * _SAMPLES_PER_ARCHIVE


def loader(path, batch_size=64, normalize=True, shuffle=True, mode='train'):
    """Generator to be used with model.fit_generator()"""
    train = mode is 'train'
    test = mode is 'test'
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        if shuffle:
            np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            patches = archive['patches']
            corners = archive['corners'].reshape(-1, 8, 1)
            images = np.expand_dims(archive['images'], -1)
            offsets = archive['offsets'].reshape(-1, 8, 1)

            del archive

            if shuffle:
                p = np.random.permutation(len(corners))
                patches = patches[p]
                corners = corners[p]
                images = images[p]
                offsets = offsets[p]

            # Split into mini batches
            num_batches = len(corners) // batch_size
            patches = np.array_split(patches, num_batches)
            corners = np.array_split(corners, num_batches)
            images = np.array_split(images, num_batches)
            offsets = np.array_split(offsets, num_batches)

            while corners:
                batch_patches = patches.pop()
                batch_corners = corners.pop()
                batch_images = images.pop()
                batch_offsets = offsets.pop()
                if normalize:
                    batch_patches = (batch_patches - 127.5) / 127.5
                    batch_images = (batch_images - 127.5) / 127.5

                if train:
                    targets = np.expand_dims(batch_patches[:, :, :, 1], -1)
                    yield [batch_patches, batch_corners, batch_images], targets
                elif test:
                    yield batch_patches, batch_offsets
                else: # demo
                    yield [batch_patches, batch_corners, batch_images], batch_offsets
