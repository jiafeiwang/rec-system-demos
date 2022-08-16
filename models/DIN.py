# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : DIN.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, BatchNormalization
from layers.attentions import DINAttention
from layers.activations import Dice
from models.base.Embeddings import Embeds
from models.base.Deep import Deep
import tensorflow.keras.backend as bk

class DIN(Model):
    def __init__(self,dense_feature_info, sparse_feature_info, item_onhot_dim, item_k_unit,
                 att_hidden_units, ffn_hidden_units, dropout):
        super(DIN, self).__init__()
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.item_onehot_dim = item_onhot_dim
        self.item_k_unit = item_k_unit
        self.att_hidden_units = att_hidden_units
        self.ffn_hidden_units = ffn_hidden_units
        self.dropout = dropout

    def build(self, input_shape):
        self.behavior_embed = Embedding(self.item_onehot_dim,self.item_k_unit)
        self.sparse_embed = Embeds(self.sparse_feature_info)
        self.attention = DINAttention(self.att_hidden_units)
        self.bn_layer = BatchNormalization()
        self.ffn_deep = Deep(self.ffn_hidden_units,activation=Dice(),dropout=self.dropout)

    def call(self, inputs, training=None, mask=None):
        # normal_inputs: [None, normal_feature_num]
        # behavior_seq_inputs:  [None, T, 1] 历史上的T次item
        # candidate_inputs: (None, 1) 本次的候选item
        normal_inputs, behavior_seq_inputs, candidate_inputs = inputs

        if bk.ndim(behavior_seq_inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect 3 dimensions" % (bk.ndim(behavior_seq_inputs))
            )

        sparse_x = self.sparse_embed(normal_inputs)
        sparse_x = tf.reshape(sparse_x,[-1,sparse_x.shape[1]*sparse_x.shape[2]])
        dense_x = tf.gather(normal_inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)
        normal_x = tf.concat([sparse_x, dense_x], axis=1)

        seq_embed = self.behavior_embed(behavior_seq_inputs) # [None, T, k]
        item_embed = self.behavior_embed(candidate_inputs) # [None, k]

        mask = tf.cast(tf.not_equal(behavior_seq_inputs[:, :, 0], 0), dtype=tf.float32)  # [None, T]
        att_emb = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # [None, k]

        x = tf.concat([att_emb, item_embed, normal_x], axis=-1)
        x = self.bn_layer(x)
        output = self.ffn_deep(x)
        return tf.nn.sigmoid(output)

