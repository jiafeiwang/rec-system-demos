# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : linear.py

import tensorflow as tf
from tensorflow.keras.layers import Layer

class LinearLayer(Layer):
    def __init__(self,w_reg):
        super(LinearLayer, self).__init__()
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w',shape=(input_shape[-1],1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.b = self.add_weight(name='b',shape=(1,),
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        output = tf.matmul(inputs,self.w) + self.b
        return output
