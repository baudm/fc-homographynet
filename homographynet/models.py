#!/usr/bin/env python

import os.path

from keras.applications import MobileNet
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.layers import Conv2D, Concatenate, Input, Reshape, Lambda

from .layers import Homography, ImageTransformer

WEIGHTS_PATH = 'https://github.com/baudm/unsupervised-homographynet/raw/master/weights/hnet_weights_tf_dim_ordering_tf_kernels.h5'


def create_models(use_weights=False):
    patch_size = 128

    base_model = MobileNet(input_shape=(patch_size, patch_size, 2), include_top=False, weights=None)
    # The output shape just before the pooling and dense layers is: (4, 4, 1024)
    x = base_model.output

    # 4 Conv layers in parallel with 2 4x4 filters each
    x = [Conv2D(2, 4, name='conv2d_{}'.format(i))(x) for i in range(1, 5)]
    x = Concatenate(name='concatenate_1')(x)
    offsets = Reshape((8, 1))(x)
    offsets_scaled = Lambda(lambda x: x * 32)(offsets)

    # Additional inputs for unsupervised training
    full_image = Input((240, 320, 1), name='full_image')
    corners = Input((8, 1), name='corners')

    H = Homography()([corners, offsets_scaled])

    warped = ImageTransformer(320, 240, patch_size)([full_image, H, corners])

    train_model = Model([base_model.input, corners, full_image], warped, name='homographynet')
    test_model = Model(base_model.input, offsets, name='homographynet_test')

    if use_weights:
        weights_name = os.path.basename(WEIGHTS_PATH)
        weights_path = get_file(weights_name, WEIGHTS_PATH,
                                cache_subdir='weights',
                                file_hash='4d52b0810d4c940dd8176a1ec1fb9641782c98570280a0ede38badc6e62288d0')
        test_model.load_weights(weights_path)

    return train_model, test_model
