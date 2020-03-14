# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.ResNet18_v2_config as ResNet18_v2_config
from models.ResNet18_v2_config import g

class ResNet18_v2:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'ResNet18_v2'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()  # as per paper
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=ResNet18_v2_config.weight_decay)


    def _res_block(self, ip, filter_num, stride, is_training):
        pre_act_1 = tf.layers.batch_normalization(ip, momentum=0.9, training=is_training)
        pre_act_1 = tf.nn.relu(pre_act_1)
        conv_1 = tf.layers.conv2d(pre_act_1, filter_num, (3, 3), stride, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)

        pre_act_2 = tf.layers.batch_normalization(conv_1, momentum=0.9, training=is_training)
        pre_act_2 = tf.nn.relu(pre_act_2)
        conv_2 = tf.layers.conv2d(pre_act_2, filter_num, (3, 3), 1, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)

        if stride == 2 or ip.get_shape().as_list()[-1] != filter_num:
            identity_branch = tf.layers.conv2d(pre_act_1, filter_num, (1, 1), stride, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        else:
            identity_branch = ip
            
        return conv_2 + identity_branch
        
    
    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 64, (7, 7), 2, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, momentum=0.9, training=is_training)
                conv1 = tf.nn.relu(conv1)
                g(conv1)
                # [None x 32 x 32 x 64]
                
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=2, padding='same')
                g(pool1)
                # [None x 16 x 16 x 64]
            
            with tf.variable_scope("block_1"):
                block_1 = self._res_block(pool1, 64, 1, is_training)
                block_1 = self._res_block(block_1, 64, 1, is_training)
                g(block_1)
                # [None x 16 x 16 x 64]
            
            with tf.variable_scope("block_2"):
                block_2 = self._res_block(block_1, 128, 2, is_training)
                block_2 = self._res_block(block_2, 128, 1, is_training)
                g(block_2)
                # [None x 8 x 8 x 128]

            with tf.variable_scope("block_3"):
                block_3 = self._res_block(block_2, 256, 2, is_training)
                block_3 = self._res_block(block_3, 256, 1, is_training)
                g(block_3)
                # [None x 4 x 4 x 256]

            with tf.variable_scope("block_4"):
                block_4 = self._res_block(block_3, 512, 2, is_training)
                block_4 = self._res_block(block_4, 512, 1, is_training)
                g(block_4)
                # [None x 2 x 2 x 512]

            with tf.variable_scope("tail"):
                tail_bn_relu = tf.layers.batch_normalization(block_4, momentum=0.9, training=is_training)
                tail_bn_relu = tf.nn.relu(tail_bn_relu)
                g(tail_bn_relu)
                # [None x 2 x 2 x 512]
                
                gap = tf.reduce_mean(tail_bn_relu, axis=[1, 2], name='global_avg_pool')
                g(gap)
                # [None x 512]
                
                fc_logits = tf.layers.dense(gap, self.num_classes, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='fc_layer')
                fc_op = tf.nn.softmax(fc_logits, name='softmax_op')
                g(fc_op)
                # [B x 200]
                
            return fc_logits, fc_op
