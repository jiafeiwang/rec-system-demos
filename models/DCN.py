# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : DCN.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from layers.cross import CrossLayer
from models.base.Embeddings import Embeds
from models.base.Deep import Deep

class DCN(Model):
    def __init__(self, dense_feature_info, sparse_feature_info, hidden_units, cross_nums, output_unit=1, dropout=None):
        super(DCN, self).__init__()
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.res_nums = cross_nums
        self.output_unit = output_unit

    def build(self, input_shape):
        self.cross_layer = CrossLayer(self.cross_layer)
        self.embed = Embeds(self.sparse_feature_info)
        self.deep = Deep(self.hidden_units)
        self.output_layer = Dense(1,activation=None)

    def call(self, inputs, training=None, mask=None):
        embed = self.embed(inputs)
        embed = tf.reshape(embed,[-1,embed.shape[1]*embed.shape[2]])
        dense_x = tf.gather(inputs,[dfeat['idx'] for dfeat in self.dense_feature_info],axis=1)
        x = tf.concat([embed,dense_x],axis=1)

        cross = self.cross_layer(x)
        deep = self.deep(x)

        output = tf.concat([cross,deep],axis=1)
        output = self.output_layer(output)
        return tf.nn.sigmoid(output)


