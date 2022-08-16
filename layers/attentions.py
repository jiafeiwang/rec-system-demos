# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : attentions.py

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, PReLU
import tensorflow.keras.backend as bk

class AFMAttention(Layer):
    def __init__(self,k_unit):
        super(AFMAttention, self).__init__()
        self.k_unit = k_unit

    def build(self, input_shape):
        self.h1 = Dense(self.k_unit,activation='relu')
        self.h2 = Dense(1,activation=None)

    def call(self, inputs, **kwargs): # [None, T, e] => [None, e]
        if bk.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect 3 dimensions" % (bk.ndim(inputs))
            )
        x = self.h1(inputs) # [None, T, k]
        x = self.h2(x) # [None, T, 1]
        score = tf.nn.softmax(x,axis=1) # [None, T, 1]
        score = tf.transpose(score,[0,2,1]) # [None, 1, T]
        value = tf.matmul(score,inputs) # [None, 1, T]x[None, T, e] = [None, 1, e]
        output = tf.reshape(value,[-1,inputs.shape[-1]]) # [None, e]
        return output

class DINAttention(Layer):
    def __init__(self,hidden_units):
        super(DINAttention, self).__init__()
        self.hidden_units = hidden_units
        self.activation = PReLU()

    def build(self, input_shape):
        self.dense_layers = [Dense(unit,activation=self.activation) for unit in self.hidden_units]
        self.output_layer = Dense(1,activation=None)

    def call(self, inputs, **kwargs):
        query,keys,values,mask = inputs
        # query [None, k] 候选item的embedding
        # keys [None, T, k] user历史所有item的embedding
        # values [None, T,k] 与keys相同
        # mask [None,T]

        # query
        # keys user历史交互行为的item的embedding
        # values values = keys

        query = tf.expand_dims(query,axis=1)  # [None, 1, k]
        query = tf.tile(query,[1,keys.shape[1],1]) # [None,T,k]
        x = tf.concat([query,keys,query-keys,query*keys],axis=-1) # [None,T,4*k]

        for layer in self.dense_layers:
            x = layer(x)

        score = self.output_layer(x) # [None, T, 1]
        score = tf.squeeze(score,axis=-1) # [None, T]
        padding = tf.ones_like(score)*(-2**32+1) # [None, T]
        score = tf.where(tf.equal(mask,0),padding,score) # [None, T]
        score = tf.nn.softmax(score) # [None, T]
        output = tf.matmul(tf.expand_dims(score,axis=1),values) # [None,1, T]x [None, T, k] = [None, 1, k]
        output = tf.squeeze(output,axis=1) # [None, k]
        return output
