#!/usr/bin/env python

import os.path

from keras.applications import MobileNet
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.layers import Conv2D, Concatenate, Input, Reshape, Lambda

from .layers import Homography, ImageTransformer

MOBILENET_WEIGHTS_PATH = 'https://github.com/baudm/HomographyNet/raw/master/models/mobile_homographynet_weights_tf_dim_ordering_tf_kernels.h5'


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

    full_image = Input((240, 320, 1), name='full_image')
    corners = Input((8, 1), name='corners')

    H = Homography()([corners, offsets_scaled])

    warped = ImageTransformer(320, 240, patch_size)([full_image, H, corners])

    train_model = Model([base_model.input, corners, full_image], warped, name='mobile_homographynet')
    test_model = Model(base_model.input, offsets, name='mobile_homographynet_test')

    if use_weights:
        weights_name = os.path.basename(MOBILENET_WEIGHTS_PATH)
        weights_path = get_file(weights_name, MOBILENET_WEIGHTS_PATH,
                                cache_subdir='models',
                                file_hash='e161aabc5a04ff715a6f5706855a339d598d1216a4a5f45b90b8dbf5f8bcedc3')
        test_model.load_weights(weights_path)

    return train_model, test_model
