#!/usr/bin/env python3

import numpy as np

from keras.layers import Input
from keras.models import Model
import keras.backend as K

import tensorflow as tf

from homographynet.layers import DirectLinearTransform

def main():
    p1 = Input((8, 1), name='p1')
    p2 = Input((8, 1), name='p2')
    y = DirectLinearTransform()([p1, p2])
    m = Model([p1, p2], y)
    m.summary()

    m.compile('adam', 'mse')

    top = 50
    left = 100

    data_p1 = np.array([[left,top,left+128,top,left+128,top+128,left,top+128]]).reshape(1,8,1)
    data_p2 = np.array([[-10, 4, 32, -8, -20, 5, 0, -15]]).reshape(1,8,1)
    #labels = np.array([[[1,2,3],[4,5,6], [7,8,9]]])
    labels = np.array([[[ 7.02728475e-01,  3.96032036e-01, -4.81192937e+00],
       [-1.78547446e-01,  1.33162906e+00,  2.43080358e+00],
       [-2.13648383e-03,  3.22019432e-03,  1.00000000e+00]]])

    m.fit([data_p1, data_p2], labels, batch_size=1)

    a = DirectLinearTransform().call([p1, p2])

    sess = K.get_session()

    r = sess.run(a, feed_dict={p1: data_p1.astype('float32'), p2: data_p2.astype('float32')})
    print(r)







if __name__ == '__main__':
    main()