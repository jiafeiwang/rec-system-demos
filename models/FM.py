# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : FM.py

import tensorflow as tf
from tensorflow.keras.models import Model
from layers.fm import FMLayer

class FM(Model):
    def __init__(self, w_reg, v_reg, k_unit):
        super(FM, self).__init__()
        self.k_unit = k_unit
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.fm_layer = FMLayer(self.k_unit, self.w_reg, self.v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm_layer(inputs)
        return output