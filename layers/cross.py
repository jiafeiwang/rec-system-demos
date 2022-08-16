# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : cross.py

import tensorflow as tf
from tensorflow.keras.layers import Layer


class CrossLayer(Layer):
    def __init__(self, cross_nums, reg_w=1e-3, reg_b=1e-3):
        super().__init__()
        self.cross_nums = cross_nums
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.ws = [self.add_weight(name='w_' + str(i),
                                   shape=(input_shape[1], 1),
                                   initializer=tf.random_normal_initializer(),
                                   regularizer=tf.keras.regularizers.l2(self.reg_w),
                                   trainable=True)
                   for i in range(self.cross_nums)]

        self.bs = [self.add_weight(name='b_' + str(i),
                                   shape=(input_shape[1], 1),
                                   initializer=tf.zeros_initializer(),
                                   regularizer=tf.keras.regularizers.l2(self.reg_b),
                                   trainable=True)
                   for i in range(self.cross_nums)]

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)  # [None, f, 1]
        xl = x0
        for i in range(self.cross_nums):
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.ws[i])  # [None, 1, f]x[f, 1] = [None, 1, 1]

            xl = tf.matmul(x0, xl_w) + self.bs[
                i] + xl  # [None, f, 1]x[None, 1, 1] + [f,1] + [None, f, 1] = [None, f, 1]

        output = tf.squeeze(xl, axis=2)  # [None, f]
        return output
