# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : DeepCrossing.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from models.base.Embeddings import Embeds
from models.base.Deep import ResDeep

class DeepCrossing(Model):
    def __init__(self,dense_feature_info,sparse_feature_info, hidden_units, res_nums, output_unit = 1, dropout=None):
        super(DeepCrossing, self).__init__()
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.res_nums = res_nums
        self.output_unit = output_unit

    def build(self, input_shape):
        self.embeds = Embeds(self.sparse_feature_info)
        self.res_deep = ResDeep(self.hidden_units,self.dropout)
        self.output_layer = Dense(self.output_unit)

    def call(self, inputs, training=None, mask=None):
        embed_x = self.embeds(inputs)
        embed_x = tf.reshape(embed_x,[-1,embed_x.shape[1]*embed_x.shape[2]])
        dense_x = tf.gather(inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)
        x = tf.concat([embed_x,dense_x],axis=1)
        for num in self.res_nums:
            x = self.res_deep(x)

        output = self.output_layer(x)
        return tf.nn.sigmoid(output)

