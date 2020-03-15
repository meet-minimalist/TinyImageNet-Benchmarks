# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from models import VGG16_config 
from models.VGG16_config import g
import os


class VGG16:
    def __init__(self, input_dims=(64, 64), num_classes=200, fully_convo=True, for_tinyimagenet=True):
        self.model_name = 'VGG16'
        self.fully_convo = fully_convo
        self.for_tinyimagenet = for_tinyimagenet
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        #self.k_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=VGG16_config.weight_decay)
    
    def _vgg_block(self, ip, layer_reps, filter_num, is_training, scope_name):
        _conv = ip
        with tf.variable_scope(scope_name):
            for _ in range(layer_reps):
                _conv = tf.layers.conv2d(_conv, filter_num, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)         
            return _conv
                
    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            conv_1_1 = tf.layers.conv2d(x, 64, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_1_2 = tf.layers.conv2d(conv_1_1, 64, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            g(conv_1_2)
            max_pool_1 = tf.layers.max_pooling2d(conv_1_2, (2, 2), 2, 'same')
            g(max_pool_1)
            # [None x 32 x 32 x 64]
            
            conv_2_1 = tf.layers.conv2d(max_pool_1, 128, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_2_2 = tf.layers.conv2d(conv_2_1, 128, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            g(conv_2_2)
            max_pool_2 = tf.layers.max_pooling2d(conv_2_2, (2, 2), 2, 'same')
            g(max_pool_2)
            # [None x 16 x 16 x 128]

            conv_2_1 = tf.layers.conv2d(max_pool_2, 256, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_2_2 = tf.layers.conv2d(conv_2_1, 256, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_2_3 = tf.layers.conv2d(conv_2_2, 256, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            g(conv_2_3)
            max_pool_3 = tf.layers.max_pooling2d(conv_2_3, (2, 2), 2, 'same')
            g(max_pool_3)
            # [None x 8 x 8 x 256]

            conv_3_1 = tf.layers.conv2d(max_pool_3, 512, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_3_2 = tf.layers.conv2d(conv_3_1, 512, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_3_3 = tf.layers.conv2d(conv_3_2, 512, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            g(conv_3_3)
            max_pool_4 = tf.layers.max_pooling2d(conv_3_3, (2, 2), 2, 'same')
            g(max_pool_4)
            # [None x 4 x 4 x 512]

            conv_4_1 = tf.layers.conv2d(max_pool_4, 512, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_4_2 = tf.layers.conv2d(conv_4_1, 512, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_4_3 = tf.layers.conv2d(conv_4_2, 512, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            g(conv_4_3)
            max_pool_5 = tf.layers.max_pooling2d(conv_4_3, (2, 2), 2, 'same')
            g(max_pool_5)
            # [None x 2 x 2 x 512]
            
            if not self.fully_convo:
                global_avg_pool = tf.reduce_mean(max_pool_5, axis=[1, 2])
                g(global_avg_pool)

                dense_1 = tf.layers.dense(global_avg_pool, 4096, tf.nn.relu, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                dense_1 = tf.layers.dropout(dense_1, rate=0.5, training=is_training)
                g(dense_1)

                dense_2 = tf.layers.dense(dense_1, 4096, tf.nn.relu, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                dense_2 = tf.layers.dropout(dense_2, rate=0.5, training=is_training)
                g(dense_2)
                
                fc_logits = tf.layers.dense(dense_2, self.num_classes, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                g(fc_logits)

                fc_op = tf.nn.softmax(fc_logits, name='softmax_op')
                g(fc_op)
                
                return fc_logits, fc_op
            
            else:
                if self.for_tinyimagenet:
                    # In tinyimagenet there are 200 classes only, which is 1/5th of Imagenet classes
                    # So reduced number of convolutional channels
                    channels = 1024
                else:
                    channels = 4096

                k_size = max_pool_5.get_shape().as_list()[1:3]
                conv_global_avg_pool = tf.layers.conv2d(max_pool_5, channels, k_size, 1, 'valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv_global_avg_pool = tf.layers.dropout(conv_global_avg_pool, rate=0.5)
                g(conv_global_avg_pool)
                
                conv_1 = tf.layers.conv2d(conv_global_avg_pool, channels, (1, 1), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv_1 = tf.layers.dropout(conv_1, rate=0.5, training=is_training)
                g(conv_1)

                conv_2 = tf.layers.conv2d(conv_1, self.num_classes, (1, 1), 1, 'valid', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                g(conv_2)

                conv_2 = tf.squeeze(conv_2, axis=[1, 2])
                conv_2_op = tf.nn.softmax(conv_2, name='softmax_op')
                g(conv_2_op)

                return conv_2, conv_2_op

