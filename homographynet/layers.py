#!/usr/bin/env python3

from keras.layers import Layer

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
