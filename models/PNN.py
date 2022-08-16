# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : FNN.py

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from layers.products import InnerProductLayer
from base.Embeddings import Embeds
from base.Deep import ResDeep


class PNN(Model):
    def __init__(self, dense_feature_info, sparse_feature_info, hidden_units, res_nums, output_unit=1, dropout=None):
        super(PNN, self).__init__()
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.res_nums = res_nums
        self.output_unit = output_unit

    def build(self, input_shape):
        self.embeds = Embeds(self.sparse_feature_info)
        self.res_deep = ResDeep(self.hidden_units, self.dropout)
        self.inner_product_layer = InnerProductLayer()
        self.output_layer = Dense(self.output_unit)

    def call(self, inputs, training=None, mask=None):
        embed_x = self.embeds(inputs)
        dense_x = tf.gather(inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)
        product = self.inner_product_layer(embed_x)
        x = tf.concat([tf.reshape(embed_x,[-1,embed_x.shape[1]*embed_x.shape[2]]), product, dense_x])

        for num in self.res_nums:
            x = self.res_deep(x)

        output = self.output_layer(x)
        return tf.nn.sigmoid(output)
