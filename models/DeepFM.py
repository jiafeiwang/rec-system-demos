# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : DeepFM.py

import tensorflow as tf
from tensorflow.keras.models import Model

from layers.fm import FMLayer
from models.base.Deep import Deep
from models.base.Embeddings import Embeds

class DeepFM(Model):
    def __init__(self,k_unit,w_reg,v_reg,hidden_units,dense_feature_info,sparse_feature_info):
        super(DeepFM, self).__init__()
        self.k_unit = k_unit
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.hidden_units = hidden_units

    def build(self, input_shape):
        self.embeds = Embeds(self.sparse_feature_info)
        self.fm_layer = FMLayer(self.k_unit, self.w_reg, self.v_reg)
        self.deep = Deep(self.hidden_units)

    def call(self, inputs, training=None, mask=None):
        embed_x = self.embeds(inputs)
        embed_x = tf.reshape(embed_x,[-1,embed_x.shape[1]*embed_x.shape[2]])
        embed_x_shape = tf.shape(embed_x)
        dense_x = tf.gather(inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)
        x = tf.concat([dense_x,embed_x],axis=1)
        fm_output = self.fm_layer(x)
        dense_output = self.deep(x)
        output = tf.nn.sigmoid(0.5 * (fm_output + dense_output))
        return output
