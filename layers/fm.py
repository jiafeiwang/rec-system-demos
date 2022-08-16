# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : fm.py

import tensorflow as tf
from tensorflow.keras.layers import Layer

class FMLayer(Layer):
    def __init__(self, k_dim, reg_w, reg_v):
        super(FMLayer, self).__init__()
        self.k_dim = k_dim
        self.reg_w = reg_w
        self.reg_v = reg_v

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=(input_shape[-1],1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.reg_w)
                                 )
        self.b = self.add_weight(name='b',
                                 shape=(1,),
                                 initializer=tf.zeros_initializer(),
                                 trainable=True
                                 )
        self.v = self.add_weight(name='v',
                                 shape=(input_shape[-1],self.k_dim),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.reg_v)
                                 )

    def call(self, inputs, **kwargs):
        linear_part = tf.matmul(inputs,self.w) + self.b
        square_of_sum = tf.pow(tf.matmul(inputs,self.v),2)
        sum_of_square = tf.matmul(tf.pow(inputs,2),tf.pow(self.v,2))
        inter_part = 0.5*tf.reduce_sum(square_of_sum-sum_of_square,axis=-1,keepdims=True)
        output = linear_part+inter_part
        return output



