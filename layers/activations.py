# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : activations.py

import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization

class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()

    def build(self, input_shape):
        self.bn_layer = BatchNormalization()
        self.alpha = self.add_weight(name='alpha',shape=(1,),trainable=True)

    def call(self, inputs, **kwargs):
        px = self.bn_layer(inputs)
        px = tf.nn.sigmoid(px)
        output = px*inputs + (1-px)*self.alpha*inputs
        return output
