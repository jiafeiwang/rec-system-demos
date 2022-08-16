# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : WideDeep.py

import tensorflow as tf
from tensorflow.keras.models import Model
from layers.linear import LinearLayer
from models.base.Deep import Deep
from models.base.Embeddings import Embeds

class WideDeep(Model):
    def __init__(self,dense_feature_info, sparse_feature_info, onehot_feature_info, w_reg, hidden_units, output_unit = 1, dropout=None):
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.onehot_feature_info = onehot_feature_info
        self.w_reg = w_reg
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.output_unit = output_unit

    def build(self, input_shape):
        self.linear = LinearLayer(self.w_reg)
        self.embed = Embeds(self.sparse_feature_info)
        self.deep = Deep(self.hidden_units)

    def call(self, inputs, training=None, mask=None):
        dense_x = tf.gather(inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)
        onehot_x = tf.gather(inputs, [ofeat['idx'] for ofeat in self.onehot_feature_info], axis=1)
        wide_x = tf.concat([dense_x,onehot_x],axis=1)
        embed_x = self.embed(inputs)
        wide = self.linear(wide_x)
        deep = self.deep(embed_x)
        output = 0.5*(wide+deep)
        return tf.nn.sigmoid(output)



