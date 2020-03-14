# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""


# This is NASNetA (6 @ 4032) implementation.

import tensorflow as tf
import models.NASNet_config as NASNet_config
from models.NASNet_config import g

class NASNet:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'NASNet'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=NASNet_config.weight_decay)
        self.filters_multiplier = NASNet_config.filters_multiplier
        self.num_repeated_blocks = NASNet_config.num_repeated_blocks
        self.num_reduction_cells = NASNet_config.num_reduction_cells

    def _adjust_block(self, current, prev, is_training, filters):
        if prev.get_shape().as_list()[1:3] != current.get_shape().as_list()[1:3]:
            prev = tf.nn.relu(prev)
            prev_1 = tf.layers.average_pooling2d(prev, (1, 1), strides=(2, 2), padding='valid')
            prev_1 = tf.layers.conv2d(prev_1, filters // 2, (1, 1), padding='same', use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)

            prev_2 = tf.pad(prev, [[0, 0], [0, 1], [0, 1], [0, 0]])
            prev_2 = prev_2[:, 1:, 1:, :]
            prev_2 = tf.layers.average_pooling2d(prev_2, (1, 1), strides=(2, 2), padding='valid')
            prev_2 = tf.layers.conv2d(prev_2, filters // 2, (1, 1), padding='same', use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)

            prev = tf.concat([prev_1, prev_2], axis=-1)
            prev = tf.layers.batch_normalization(prev, training=is_training)

        elif prev.get_shape().as_list()[-1] != filters:
            prev = tf.nn.relu(prev)
            prev = tf.layers.conv2d(prev, filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            prev = tf.layers.batch_normalization(prev, training=is_training)

        return prev

    def _depthwise_sep_block(self, x, is_training, filters, k_size, strides):
        dw_weights_shape = [k_size[0], k_size[1], x.get_shape().as_list()[-1], 1]
        w = tf.get_variable('dw_weight', shape=dw_weights_shape, dtype=tf.float32, initializer=self.k_init)
        
        depthwise = tf.nn.depthwise_conv2d(x, w, (1, strides[0], strides[1], 1), padding='SAME')
        pointwise = tf.layers.conv2d(depthwise, filters, (1, 1), 1, 'SAME', use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        return pointwise

    def _separable_block(self, x , is_training, filters, k_size, strides, name):
        with tf.variable_scope(name):
            with tf.variable_scope('rep_1'):
                separable_1 = tf.nn.relu(x)
                separable_1 = self._depthwise_sep_block(x, is_training, filters, k_size, strides)
                separable_1 = tf.layers.batch_normalization(separable_1, training=is_training)
            
            with tf.variable_scope('rep_2'):
                separable_2 = tf.nn.relu(separable_1)
                separable_2 = self._depthwise_sep_block(separable_1, is_training, filters, k_size, strides=(1, 1))
                separable_2 = tf.layers.batch_normalization(separable_2, training=is_training)
                
            return separable_2
            
    def _reduction_cell_a(self, current, prev, is_training, filters, name):
        with tf.variable_scope(name):
            if prev is None:
                prev = current
            else:
                prev = self._adjust_block(current, prev, is_training, filters)   
                # adjust the shape of "prev" tensor according to "current" tensor
                # to make the channels and spatial dimensions equal.

            # below code is to make the channels of "current" tensor equal to the "filters" argument.
            current_mod = tf.nn.relu(current)
            current_mod = tf.layers.conv2d(current_mod, filters, (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            current_mod = tf.layers.batch_normalization(current_mod, training=is_training)
            
            with tf.variable_scope('block_1'):
                x1_1 = self._separable_block(current_mod, is_training, filters, (5, 5), strides=(2, 2), name='sep_1')
                x1_2 = self._separable_block(prev, is_training, filters, (7, 7), strides=(2, 2), name='sep_2')
                x1 = tf.add(x1_1, x1_2)

            with tf.variable_scope('block_2'):
                x2_1 = tf.layers.max_pooling2d(current_mod, (3, 3), (2, 2), padding='same')
                x2_2 = self._separable_block(prev, is_training, filters, (7, 7), strides=(2, 2), name='sep_1')
                x2 = tf.add(x2_1, x2_2)

            with tf.variable_scope('block_3'):
                x3_1 = tf.layers.average_pooling2d(current_mod, (3, 3), (2, 2), 'same')
                x3_2 = self._separable_block(prev, is_training, filters, (5, 5), strides=(2, 2), name='sep_1')
                x3 = tf.add(x3_1, x3_2)

            with tf.variable_scope('block_4'):
                x4 = tf.layers.average_pooling2d(x1, (3, 3), (1, 1), 'same')
                x4 = tf.add(x4, x2)

            with tf.variable_scope('block_5'):
                x5_1 = tf.layers.max_pooling2d(current_mod, (3, 3), (2, 2), 'same')
                x5_2 = self._separable_block(x1, is_training, filters, (3, 3), strides=(1, 1), name='sep_1')
                x5 = tf.add(x5_1, x5_2)

            x_ = tf.concat([x5, x4, x3, x2], axis=-1)
        return x_, current
    
    def _normal_cell_a(self, current, prev, is_training, filters, name):
        with tf.variable_scope(name):
            prev = self._adjust_block(current, prev, is_training, filters)
            # adjust the shape of "prev" tensor according to "current" tensor
            # to make the channels and spatial dimensions equal.

            # below code is to make the channels of "current" tensor equal to the "filters" argument.
            current_mod = tf.nn.relu(current)
            current_mod = tf.layers.conv2d(current_mod, filters, (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            current_mod = tf.layers.batch_normalization(current_mod, training=is_training)

            with tf.variable_scope('block_1'):
                x1 = self._separable_block(current_mod, is_training, filters, (3, 3), strides=(1, 1), name='sep_1')
                x1 = tf.add(x1, current_mod)

            with tf.variable_scope('block_2'):
                x2_1 = self._separable_block(prev, is_training, filters, (3, 3), strides=(1, 1), name='sep_1_1')
                x2_2 = self._separable_block(current_mod, is_training, filters, (5, 5), strides=(1, 1), name='sep_1_2')
                x2 = tf.add(x2_1, x2_2)

            with tf.variable_scope('block_3'):
                x3 = tf.layers.average_pooling2d(current_mod, (3, 3), (1, 1), 'same')
                x3 = tf.add(x3, prev)

            with tf.variable_scope('block_4'):
                x4_1 = tf.layers.average_pooling2d(prev, (3, 3), (1, 1), 'same')
                x4_2 = tf.layers.average_pooling2d(prev, (3, 3), (1, 1), 'same')
                x4 = tf.add(x4_1, x4_2)

            with tf.variable_scope('block_5'):
                x5_1 = self._separable_block(prev, is_training, filters, (5, 5), strides=(1, 1), name='sep_1_1')
                x5_2 = self._separable_block(prev, is_training, filters, (3, 3), strides=(1, 1), name='sep_1_2')
                x5 = tf.add(x5_1, x5_2)

            x_ = tf.concat([prev, x1, x2, x3, x4, x5], axis=-1)
        return x_, current
        

    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]
        
        filters = NASNet_config.penultimate_filters / ((2 ** self.num_reduction_cells) * 6)     # 4032 / 24 == 168
        stem_filters = 96

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("Stem"):
                print("---- Stem ----")
                conv_1 = tf.layers.conv2d(x, stem_filters, (3, 3), 2, padding='valid', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv_1 = tf.layers.batch_normalization(conv_1, training=is_training)
                g(conv_1)
                # [B x 31 x 31 x 96]
                
                prev = None
                current = conv_1
                current, prev = self._reduction_cell_a(current, prev, is_training, filters // (self.filters_multiplier ** 2), 'stem_1')
                g(current)
                g(prev)
                # current: [B x 16 x 16 x 168]
                # prev   : [B x 31 x 31 x 96]
                
                current, prev = self._reduction_cell_a(current, prev, is_training, filters // (self.filters_multiplier), 'stem_2')
                g(current)
                g(prev)
                # current: [B x 8 x 8 x 336]
                # prev   : [B x 16 x 16 x 168]

            print("---- Normal_Block_1 ----")
            for i in range(self.num_repeated_blocks):
                current, prev = self._normal_cell_a(current, prev, is_training, filters, 'normal_block_1_' + str(i+1))
            g(current)
            g(prev)
            # current: [B x 8 x 8 x 1008]
            # prev   : [B x 8 x 8 x 1008]              
            
            print("---- Reduction_Block_1 ----")
            current, prev = self._reduction_cell_a(current, prev, is_training, filters * self.filters_multiplier, 'reduction_block_1')
            g(current)
            g(prev)
            # current: [B x 4 x 4 x 1344]
            # prev   : [B x 8 x 8 x 1008]
            
            print("---- Normal_Block_2 ----")
            for i in range(self.num_repeated_blocks):
                current, prev = self._normal_cell_a(current, prev, is_training, filters * self.filters_multiplier, 'normal_block_2_' + str(i+1))
            g(current)
            g(prev)
            # current: [B x 4 x 4 x 2016]
            # prev   : [B x 4 x 4 x 2016]

            print("---- Reduction_Block_2 ----")
            current, prev = self._reduction_cell_a(current, prev, is_training, filters * (self.filters_multiplier ** 2), 'reduction_block_2')
            g(current)
            g(prev)
            # current: [B x 2 x 2 x 2688]
            # prev   : [B x 4 x 4 x 2016]
            
            print("---- Normal_Block_3 ----")
            for i in range(self.num_repeated_blocks):
                current, prev = self._normal_cell_a(current, prev, is_training, filters * (self.filters_multiplier ** 2), 'normal_block_3_' + str(i+1))
            g(current)
            g(prev)
            # current: [B x 2 x 2 x 4032]
            # prev   : [B x 2 x 2 x 4032]
            

            with tf.variable_scope("Tail"):
                print("---- Tail ----")
                current = tf.nn.relu(current)
                
                kernel_size = current.get_shape().as_list()[1:3]      
                avg_pool = tf.layers.average_pooling2d(current, pool_size=kernel_size, strides=1, padding='valid')
                g(avg_pool)
                # [None x 1 x 1 x 4032]
                
                dropout = tf.squeeze(tf.layers.dropout(avg_pool, rate=0.5), axis=[1, 2])
                g(dropout)
                # [None x 4032]
                
                logits = tf.layers.dense(dropout, self.num_classes, activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='logits')
                g(logits)
                # [None x 200]
                
                op = tf.nn.softmax(logits, axis=-1, name='softmaxed_output')
                g(op)
                # [None x 200]
                
            return logits, op
