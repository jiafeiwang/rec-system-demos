# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : Embeddings.py

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model


class Embeds(Model):
    def __init__(self, sparse_feature_info):
        super(Embeds, self).__init__()
        self.sparse_feature_info = sparse_feature_info

    def build(self, input_shape):
        self.embed_layers = {sfeat['name']: Embedding(sfeat['onehot_dims'], sfeat['embed_dims'])
                             for sfeat in self.sparse_feature_info}

    def call(self, inputs, training=None, mask=None):
        embed_x = [self.embed_layers[sfeat['name']](inputs[:, sfeat['idx']]) for sfeat in
                   self.sparse_feature_info]
        embed_x = tf.convert_to_tensor(embed_x)
        output = tf.transpose(embed_x, [1, 0, 2])  # output [None, f, k]
        return output
