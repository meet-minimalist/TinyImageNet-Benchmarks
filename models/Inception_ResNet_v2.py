# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 23:17:03 2020

@author: Meet
"""

### Note: This implementation has been inspired and taken from : https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py


import tensorflow as tf
import models.Inception_ResNet_v2_config as Inception_ResNet_v2_config
from models.Inception_ResNet_v2_config import g

class Inception_ResNet_v2:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'Inception_ResNet_v2'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=Inception_ResNet_v2_config.weight_decay)
    
    def _conv_layer(self, x, is_training, k_size, no_filters, stride, padding):
        conv = tf.layers.conv2d(x, no_filters, k_size, stride, padding, activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, training=is_training)
        conv = tf.nn.relu(conv)
        return conv
    
    def _input_stem(self, x, is_training):
        conv_1 = self._conv_layer(x, is_training, (3, 3), 32, 2, 'valid')
        conv_2 = self._conv_layer(conv_1, is_training, (3, 3), 32, 1, 'valid')
        conv_3 = self._conv_layer(conv_2, is_training, (3, 3), 64, 1, 'same')

        mxpool_4 = tf.layers.max_pooling2d(conv_3, (3, 3), (2, 2), padding='valid')
        conv_4 = self._conv_layer(conv_3, is_training, (3, 3), 96, 2, 'valid')
        concat_4 = tf.concat([mxpool_4, conv_4], axis=-1)
        
        branch_1 = self._conv_layer(concat_4, is_training, (1, 1), 64, 1, 'same')
        branch_1 = self._conv_layer(branch_1, is_training, (3, 3), 96, 1, 'valid')
        
        branch_2 = self._conv_layer(concat_4, is_training, (1, 1), 64, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (1, 7), 64, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (7, 1), 64, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (3, 3), 96, 1, 'valid')
        
        concat_5 = tf.concat([branch_1, branch_2], axis=-1)
        
        # here, we are not downsizing as we are working with 64 x 64 dims (TinyImageNet) not 299 x 299 original ImageNet
        mxpool_6 = tf.layers.max_pooling2d(concat_5, (3, 3), (1, 1), padding='same')
        conv_6 = self._conv_layer(concat_5, is_training, (3, 3), 192, 1, 'same')
        concat_6 = tf.concat([mxpool_6, conv_6], axis=-1)
        
        return concat_6

    
    def _inception_a(self, x, is_training):
        branch_1 = self._conv_layer(x, is_training, (1, 1), 32, 1, 'same')
                
        branch_2 = self._conv_layer(x, is_training, (1, 1), 32, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (3, 3), 32, 1, 'same')
        
        branch_3 = self._conv_layer(x, is_training, (1, 1), 32, 1, 'same')
        branch_3 = self._conv_layer(branch_3, is_training, (3, 3), 48, 1, 'same')
        branch_3 = self._conv_layer(branch_3, is_training, (3, 3), 64, 1, 'same')
        
        concat = tf.concat([branch_1, branch_2, branch_3], axis=-1)

        concat_1x1 = self._conv_layer(concat, is_training, (1, 1), 384, 1, 'same')

        return tf.nn.relu(tf.add(x, concat_1x1))
        
    def _inception_a_reduction(self, x, is_training):
        branch_1 = tf.layers.max_pooling2d(x, (3, 3), (2, 2), padding='valid')

        branch_2 = self._conv_layer(x, is_training, (3, 3), 384, 2, 'valid')

        branch_3 = self._conv_layer(x, is_training, (1, 1), 256, 1, 'same')
        branch_3 = self._conv_layer(branch_3, is_training, (3, 3), 256, 1, 'same')
        branch_3 = self._conv_layer(branch_3, is_training, (3, 3), 384, 2, 'valid')

        concat = tf.concat([branch_1, branch_2, branch_3], axis=-1)
        return concat
    
    def _inception_b(self, x, is_training):
        branch_1 = self._conv_layer(x, is_training, (1, 1), 192, 1, 'same')
        
        branch_2 = self._conv_layer(x, is_training, (1, 1), 128, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (1, 7), 160, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (7, 1), 192, 1, 'same')
        
        concat = tf.concat([branch_1, branch_2], axis=-1)
        
        concat_1x1 = self._conv_layer(concat, is_training, (1, 1), 1152, 1, 'same')
        
        return tf.nn.relu(tf.add(concat_1x1, x))
    
    def _inception_b_reduction(self, x, is_training):
        branch_1 = tf.layers.max_pooling2d(x, (3, 3), (2, 2), padding='valid')

        branch_2 = self._conv_layer(x, is_training, (1, 1), 256, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (3, 3), 384, 2, 'valid')

        branch_3 = self._conv_layer(x, is_training, (1, 1), 256, 1, 'same')
        branch_3 = self._conv_layer(branch_3, is_training, (3, 3), 288, 2, 'valid')

        branch_4 = self._conv_layer(x, is_training, (1, 1), 256, 1, 'same')
        branch_4 = self._conv_layer(branch_4, is_training, (3, 3), 288, 1, 'same')
        branch_4 = self._conv_layer(branch_4, is_training, (3, 3), 320, 2, 'valid')

        concat = tf.concat([branch_1, branch_2, branch_3, branch_4], axis=-1)
        return concat
    
    def _inception_c(self, x, is_training):
        branch_1 = self._conv_layer(x, is_training, (1, 1), 192, 1, 'same')
        
        branch_2 = self._conv_layer(x, is_training, (1, 1), 192, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (1, 3), 224, 1, 'same')
        branch_2 = self._conv_layer(branch_2, is_training, (3, 1), 256, 1, 'same')
        
        concat = tf.concat([branch_1, branch_2], axis=-1)

        concat_1x1 = self._conv_layer(concat, is_training, (1, 1), 2144, 1, 'same')
        
        return tf.nn.relu(tf.add(concat_1x1, x))
    
    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]
        
        with tf.variable_scope(self.model_name):
            with tf.variable_scope("Stem"):
                print("---- Conv Stem ----")
                ip_stem = self._input_stem(x, is_training)
                g(ip_stem)
                # [None x 12 x 12 x 384]
                
            with tf.variable_scope("Inception_A"):
                print("---- Inception A ----")
            
                inception_a_1 = self._inception_a(ip_stem, is_training)
                inception_a_2 = self._inception_a(inception_a_1, is_training)
                inception_a_3 = self._inception_a(inception_a_2, is_training)
                inception_a_4 = self._inception_a(inception_a_3, is_training)
                inception_a_5 = self._inception_a(inception_a_4, is_training)
                g(inception_a_5)
                # [None x 12 x 12 x 384]
                inception_a_reduction = self._inception_a_reduction(inception_a_5, is_training)
                g(inception_a_reduction)
                # [None x 5 x 5 x 1152]
                
            with tf.variable_scope("Inception_B"):
                print("---- Inception B ----")

                inception_b_1 = self._inception_b(inception_a_reduction, is_training)
                inception_b_2 = self._inception_b(inception_b_1, is_training)
                inception_b_3 = self._inception_b(inception_b_2, is_training)
                inception_b_4 = self._inception_b(inception_b_3, is_training)
                inception_b_5 = self._inception_b(inception_b_4, is_training)
                inception_b_6 = self._inception_b(inception_b_5, is_training)
                inception_b_7 = self._inception_b(inception_b_6, is_training)
                inception_b_8 = self._inception_b(inception_b_7, is_training)
                inception_b_9 = self._inception_b(inception_b_8, is_training)
                inception_b_10 = self._inception_b(inception_b_9, is_training)
                g(inception_b_10)
                # [None x 5 x 5 x 1152]
                
                inception_b_reduction = self._inception_b_reduction(inception_b_10, is_training)
                g(inception_b_reduction)
                # [None x 2 x 2 x 2144]
                
            with tf.variable_scope("Inception_C"):
                print("---- Inception C ----")

                inception_c_1 = self._inception_c(inception_b_reduction, is_training)
                inception_c_2 = self._inception_c(inception_c_1, is_training)
                inception_c_3 = self._inception_c(inception_c_2, is_training)
                inception_c_4 = self._inception_c(inception_c_3, is_training)
                inception_c_5 = self._inception_c(inception_c_4, is_training)
                g(inception_c_5)
                # [None x 2 x 2 x 2144]

            with tf.variable_scope("Tail"):
                print("---- Tail ----")

                kernel_size = inception_c_5.get_shape().as_list()[1:3]                
                avg_pool = tf.layers.average_pooling2d(inception_c_5, pool_size=kernel_size, strides=1, padding='valid')
                g(avg_pool)
                # [None x 1 x 1 x 2144]
                dropout = tf.squeeze(tf.layers.dropout(avg_pool, rate=0.8), axis=[1, 2])
                g(dropout)
                # [None x 2144]
                
                logits = tf.layers.dense(dropout, self.num_classes, activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='logits')
                g(logits)
                # [None x 200]
                
                op = tf.nn.softmax(logits, axis=-1, name='softmaxed_output')
                g(op)
                # [None x 200]
                
            return logits, op
