#!/usr/bin/env python

import os.path

from keras.applications import MobileNet
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.layers import Conv2D, Concatenate, Input, Reshape, Lambda

from .layers import Homography, ImageTransformer

WEIGHTS_PATH = 'https://github.com/baudm/fc-homographynet/raw/master/weights/hnet_weights_tf_dim_ordering_tf_kernels.h5'


def create_models(use_weights=False):
    patch_size = 128

    base_model = MobileNet(input_shape=(patch_size, patch_size, 2), include_top=False, weights=None)
    # The output shape just before the pooling and dense layers is: (4, 4, 1024)
    x = base_model.output

    x = Conv2D(8, 4, name='conv2d_1')(x)
    offsets = Reshape((8, 1))(x)

    # Additional inputs for self-supervised training
    full_image = Input((240, 320, 1), name='full_image')
    corners = Input((8, 1), name='corners')

    # Compute the 3x3 homography matrix from the 4-pt formulation
    H = Homography()([corners, offsets])

    # Warp the original image using the estimated homography
    warped = ImageTransformer(320, 240, patch_size)([full_image, H, corners])

    train_model = Model([base_model.input, corners, full_image], warped, name='homographynet')
    test_model = Model(base_model.input, offsets, name='homographynet_test')

    if use_weights:
        weights_name = os.path.basename(WEIGHTS_PATH)
        proj_root = os.path.split(os.path.dirname(__file__))[0]
        weights_path = get_file(weights_name, WEIGHTS_PATH,
                                cache_subdir='weights',
                                cache_dir=proj_root,
                                file_hash='7fb1d81bd7d2c01a73b6a94d65cc46c712cc8c6eebae8e0653e4f7660d9a1eb3')
        test_model.load_weights(weights_path)

    return train_model, test_model
