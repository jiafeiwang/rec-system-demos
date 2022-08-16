# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : NFM.py

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from models.base.Deep import Deep
from models.base.Embeddings import Embeds


class NFM(Model):
    def __init__(self,dense_feature_info,sparse_feature_info,hidden_units,out_unit=1,dropout=0.1):
        super(NFM, self).__init__()
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.hidden_units = hidden_units
        self.out_unit = out_unit
        self.dropout = dropout

    def build(self, input_shape):
        self.embeds = Embeds(self.sparse_feature_info)
        self.bn_layer = BatchNormalization()
        self.deep = Deep(self.hidden_units)

    def call(self, inputs, training=None, mask=None):
        embed_x = self.embeds(inputs)
        # embedding 向量进行交叉
        inter_x =0.5*(tf.pow(tf.reduce_sum(embed_x,axis=1),2)-tf.reduce_sum(tf.pow(embed_x,2),axis=1))
        dense_x = tf.gather(inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)
        x = tf.concat([inter_x,dense_x],axis=1)
        x = self.bn_layer(x)
        output = self.deep(x)
        return tf.nn.sigmoid(output)