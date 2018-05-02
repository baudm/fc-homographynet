#!/usr/bin/env python3

import numpy as np

from keras.layers import Input
from keras.models import Model
import keras.backend as K

from homographynet.layers import Homography, ImageTransformer

import cv2

def main():
    p1 = Input((8, 1), name='p1')
    p2 = Input((8, 1), name='p2')
    y = Homography()([p1, p2])
    m = Model([p1, p2], y)
    m.summary()

    m.compile('adam', 'mse')

    top = 50
    left = 100

    data_p1 = np.array([[left, top, left + 128, top, left + 128, top + 128, left, top + 128]]).reshape(-1, 8, 1)
    data_p2 = np.array([[-10, 4, 32, -8, -20, 5, 0, -15]]).reshape(-1, 8, 1)
    labels = np.array([[[7.02728475e-01, 3.96032036e-01, -4.81192937e+00],
                        [-1.78547446e-01, 1.33162906e+00, 2.43080358e+00],
                        [-2.13648383e-03, 3.22019432e-03, 1.00000000e+00]]])

    m.fit([data_p1, data_p2], labels, batch_size=1)

    a = Homography().call([p1, p2])

    img_tensor = Input((240, 320, 1), name='img_tensor')
    H = Input((3, 3), name='H')

    corners = Input((8, 1), name='corners', dtype='int32')

    img = cv2.imread('dataset/000000084752.jpg', cv2.IMREAD_GRAYSCALE).reshape(1, 240, 320, 1)
    print(img.shape)
    b = ImageTransformer(320, 240).call([img_tensor, H, corners])


    sess = K.get_session()

    r = sess.run(a, feed_dict={p1: data_p1.astype('float32'), p2: data_p2.astype('float32')})
    print(r)

    r2 = sess.run(b, feed_dict={img_tensor: img, H: labels, corners: data_p1})
    # print(r2.shape)

    import matplotlib.pyplot as plt

    plt.imshow(r2[0].squeeze(), cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()