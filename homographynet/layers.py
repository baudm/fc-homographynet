#!/usr/bin/env python3

import tensorflow as tf
from keras.layers import Layer

from .vendor.spatial_transformer import transformer
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
        return (input_shape[0], 3, 3)


class ImageTransformer(Layer):

    def call(self, inputs, **kwargs):
        I = inputs

        # Transform H_mat since we scale image indices in transformer
        H_mat = tf.matmul(tf.matmul(self.M_tile_inv, self.H_mat), self.M_tile)
        # Transform image 1 (large image) to image 2
        out_size = (self.params.img_h, self.params.img_w)
        warped_images, _ = transformer(self.I, H_mat, out_size)
        # TODO: warp image 2 to image 1

        # Extract the warped patch from warped_images by flatting the whole batch before using indices
        # Note that input I  is  3 channels so we reduce to gray
        warped_gray_images = tf.reduce_mean(warped_images, 3)
        warped_images_flat = tf.reshape(warped_gray_images, [-1])
        patch_indices_flat = tf.reshape(self.patch_indices, [-1])
        pixel_indices = patch_indices_flat + self.batch_indices_tensor
        pred_I2_flat = tf.gather(warped_images_flat, pixel_indices)

        self.pred_I2 = tf.reshape(pred_I2_flat,
                                  [self.params.batch_size, self.params.patch_size, self.params.patch_size, 1])
        return self.pred_I2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128)
