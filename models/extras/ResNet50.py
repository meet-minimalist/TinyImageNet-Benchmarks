# -*- coding: utf-8 -*-
"""
Created on Sat Mar 08 17:11:01 2020

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.ResNet50_config as ResNet50_config
from models.ResNet50_config import g

class ResNet50:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'ResNet50'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=ResNet50_config.weight_decay)
    
    def _residual_branch(self, ip, filter_num, stride, is_training):
        conv = tf.layers.conv2d(ip, filter_num // 4, (1, 1), stride, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, momentum=0.9, training=is_training)
        conv = tf.nn.relu(conv)
        
        conv = tf.layers.conv2d(conv, filter_num // 4, (3, 3), 1, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, momentum=0.9, training=is_training)
        conv = tf.nn.relu(conv)
        
        conv = tf.layers.conv2d(conv, filter_num, (1, 1), 1, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, momentum=0.9, training=is_training)
        return conv
        
    def _identity_branch(self, ip, filter_num, stride, is_training):
        if stride == 2 or ip.get_shape().as_list()[-1] != filter_num:
            ip = tf.layers.conv2d(ip, filter_num, (1, 1), stride, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            ip = tf.layers.batch_normalization(ip, momentum=0.9, training=is_training)
        return ip
    
    def _res_block(self, ip, filter_num, stride, is_training):
        identity_branch = self._identity_branch(ip, filter_num, stride, is_training)
        res_branch = self._residual_branch(ip, filter_num, stride, is_training)
        return tf.nn.relu(identity_branch + res_branch)
        
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
                block_1 = self._res_block(pool1, 256, 1, is_training)
                block_1 = self._res_block(block_1, 256, 1, is_training)
                block_1 = self._res_block(block_1, 256, 1, is_training)
                g(block_1)
            
            with tf.variable_scope("block_2"):
                block_2 = self._res_block(block_1, 512, 2, is_training)
                block_2 = self._res_block(block_2, 512, 1, is_training)
                block_2 = self._res_block(block_2, 512, 1, is_training)
                block_2 = self._res_block(block_2, 512, 1, is_training)
                g(block_2)
            
            with tf.variable_scope("block_3"):
                block_3 = self._res_block(block_2, 1024, 2, is_training)
                block_3 = self._res_block(block_3, 1024, 1, is_training)
                block_3 = self._res_block(block_3, 1024, 1, is_training)
                block_3 = self._res_block(block_3, 1024, 1, is_training)
                block_3 = self._res_block(block_3, 1024, 1, is_training)
                block_3 = self._res_block(block_3, 1024, 1, is_training)
                g(block_3)
            
            with tf.variable_scope("block_4"):
                block_4 = self._res_block(block_3, 2048, 2, is_training)
                block_4 = self._res_block(block_4, 2048, 1, is_training)
                block_4 = self._res_block(block_4, 2048, 1, is_training)
                g(block_4)
            
            with tf.variable_scope("tail"):
                gap = tf.reduce_mean(block_4, axis=[1, 2], name='global_avg_pool')
                g(gap)
                fc_logits = tf.layers.dense(gap, self.num_classes, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='fc_layer')
                fc_op = tf.nn.softmax(fc_logits, name='softmax_op')
                g(fc_op)
                
            return fc_logits, fc_op
