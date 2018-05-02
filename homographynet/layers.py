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
        I, H_mat = inputs

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

        return warped_images
        # TODO

        # TODO: warp image 2 to image 1

        # Extract the warped patch from warped_images by flatting the whole batch before using indices
        # Note that input I  is  3 channels so we reduce to gray
        # warped_gray_images = tf.reduce_mean(warped_images, 3)
        # warped_images_flat = tf.reshape(warped_gray_images, [-1])
        # patch_indices_flat = tf.reshape(self.patch_indices, [-1])
        # pixel_indices = patch_indices_flat + self.batch_indices_tensor
        # pred_I2_flat = tf.gather(warped_images_flat, pixel_indices)
        #
        # self.pred_I2 = tf.reshape(pred_I2_flat,
        #                           [batch_size, patch_size, patch_size, 1])
        # return self.pred_I2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128)
