# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : AFM.py

import tensorflow as tf
from tensorflow.keras.models import Model
from layers.attentions import AFMAttention
from layers.products import FieldInteractionLayer
from models.base.Embeddings import Embeds


class AFM(Model):
    def __init__(self,sparse_feature_info, dense_feature_info, k_unit):
        super(AFM, self).__init__()
        self.sparse_feature_info = sparse_feature_info
        self.dense_feature_info = dense_feature_info
        self.k_unit = k_unit

    def build(self, input_shape):
        self.embed = Embeds(self.sparse_feature_info)
        self.field_interact_layer = FieldInteractionLayer()
        self.att_layer = AFMAttention(self.k_unit)

    def call(self, inputs, training=None, mask=None):
        embed = self.embed(inputs)
        inter_product = self.field_interact_layer(embed)
        output = self.att_layer(inter_product)
        return tf.nn.sigmoid(output)


