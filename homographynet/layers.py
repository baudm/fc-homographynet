#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from keras.layers import Layer

from .vendor.tf_spatial_transformer import transformer
from .vendor.nguyen2017unsupervised import find_homography


class Homography(Layer):
    """Layer for finding the homography

    Parameters
    ----------
    pts_1_tile : tuple of x, y-coordinates
        bounding box of the patch
    pred_h4p_tile: tuple of floats (4pt parameterization)
        The output of the
        homography prediction network
    """

    def call(self, inputs, **kwargs):
        pts_1_tile, pred_h4p_tile = inputs
        return find_homography(pts_1_tile, pred_h4p_tile)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 3, 3)


class ImageTransformer(Layer):

    def __init__(self, img_w, img_h, **kwargs):
        super().__init__(**kwargs)
        self._img_w = img_w
        self._img_h = img_h

    def call(self, inputs, **kwargs):
        I, H_mat, pts1_batch = inputs

        batch_size = tf.shape(I)[0]
        
        # Constants and variables used for spatial transformer
        M = np.array([[self._img_w / 2.0, 0., self._img_w / 2.0],
                      [0., self._img_h / 2.0, self._img_h / 2.0],
                      [0., 0., 1.]]).astype(np.float32)
        M_tensor = tf.constant(M, tf.float32)
        M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
        # Inverse of M
        M_inv = np.linalg.inv(M)
        M_tensor_inv = tf.constant(M_inv, tf.float32)
        M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])

        print('--shape of M_tile_inv:', M_tile_inv.get_shape().as_list())
        print('--shape of M:', M_tile.get_shape().as_list())
        print('--shape of M:', H_mat.get_shape().as_list())

        # Transform H_mat since we scale image indices in transformer
        H_mat = tf.matmul(tf.matmul(M_tile_inv, H_mat), M_tile)
        # Transform image 1 (large image) to image 2
        out_size = (self._img_h, self._img_w)
        warped_images, _ = transformer(I, H_mat, out_size)
        # TODO: warp image 2 to image 1

        # Crop patch
        patch_size = 128

        x = pts1_batch[:, 0, :]
        y = pts1_batch[:, 1, :]
        x2 = x + patch_size
        y2 = y + patch_size

        x = tf.divide(x, 320)
        x2 = tf.divide(x2, 320)
        y = tf.divide(y, 240)
        y2 = tf.divide(y2, 240)

        boxes = tf.reshape(tf.cast(tf.stack([y, x, y2, x2], axis=-1), 'float32'), [-1, 4])
        a = tf.image.crop_and_resize(warped_images, boxes, tf.range(0, batch_size), tf.constant([128, 128], dtype='int32'))

        return a


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 128, 128, 1)
