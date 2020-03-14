# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.MobileNet_v1_config as MobileNet_v1_config
from models.MobileNet_v1_config import g

class MobileNet_v1:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'MobileNet_v1'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()  # as per paper
        # variance scaling is good for models with relu activation
        # whereas xavier initialization is good for models with sigmoid activation
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=MobileNet_v1_config.weight_decay)
    
    def _depthwise_conv(self, ip, training, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True):
        dw_weights_shape = [k_size[0], k_size[1], ip.get_shape().as_list()[-1], depth_mul]
        
        w = tf.get_variable('dw_weight', shape=dw_weights_shape, dtype=tf.float32, initializer=self.k_init)
        # no need of using regularizer in depthwise convo as there are very less number of parameters. -- as per paper
        
        depthwise = tf.nn.depthwise_conv2d(ip, w, (1, stride, stride, 1), padding)
        if use_batch_norm:
            depthwise = tf.layers.batch_normalization(depthwise, training=training)
        depthwise = tf.nn.relu(depthwise)
        return depthwise

    
    def _pointwise_conv(self, ip, training, op_filters, use_batch_norm=True):
        pointwise = tf.layers.conv2d(ip, op_filters, (1, 1), 1, 'SAME', use_bias=not use_batch_norm, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        # we can use kernel regularizer in pointwise convolution
        if use_batch_norm:
            pointwise = tf.layers.batch_normalization(pointwise, training=training)
        pointwise = tf.nn.relu(pointwise)
        return pointwise

        
    def _depthwise_sep_conv2d(self, ip, training, op_filters, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block'):
        with tf.variable_scope(scope_name):
            depth_wise_conv = self._depthwise_conv(ip, training, k_size, stride, padding, depth_mul, use_batch_norm)
            pointwise_conv = self._pointwise_conv(depth_wise_conv, training, op_filters, use_batch_norm)
            return pointwise_conv
    

    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 32, (3, 3), 2, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, training=is_training)
                conv1 = tf.nn.relu(conv1)
                g(conv1)
                # [None x 32 x 32 x 32]
            depth_sep_1 = self._depthwise_sep_conv2d(conv1, is_training, 64, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_1')
            g(depth_sep_1)
            # [None x 32 x 32 x 64]
            
            depth_sep_2 = self._depthwise_sep_conv2d(depth_sep_1, is_training, 128, (3, 3), stride=2, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_2')
            g(depth_sep_2)
            # [None x 16 x 16 x 128]

            depth_sep_3 = self._depthwise_sep_conv2d(depth_sep_2, is_training, 128, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_3')
            g(depth_sep_3)
            # [None x 16 x 16 x 128]

            depth_sep_4 = self._depthwise_sep_conv2d(depth_sep_3, is_training, 256, (3, 3), stride=2, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_4')
            g(depth_sep_4)
            # [None x 8 x 8 x 256]

            depth_sep_5 = self._depthwise_sep_conv2d(depth_sep_4, is_training, 256, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_5')
            g(depth_sep_5)
            # [None x 8 x 8 x 256]

            depth_sep_6 = self._depthwise_sep_conv2d(depth_sep_5, is_training, 512, (3, 3), stride=2, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_6')
            g(depth_sep_6)
            # [None x 4 x 4 x 512]

            #***
            depth_sep_7 = self._depthwise_sep_conv2d(depth_sep_6, is_training, 512, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_7')
            g(depth_sep_7)
            # [None x 4 x 4 x 512]

            depth_sep_8 = self._depthwise_sep_conv2d(depth_sep_7, is_training, 512, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_8')
            g(depth_sep_8)
            # [None x 4 x 4 x 512]

            depth_sep_9 = self._depthwise_sep_conv2d(depth_sep_8, is_training, 512, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_9')
            g(depth_sep_9)
            # [None x 4 x 4 x 512]

            depth_sep_10 = self._depthwise_sep_conv2d(depth_sep_9, is_training, 512, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_10')
            g(depth_sep_10)
            # [None x 4 x 4 x 512]

            depth_sep_11 = self._depthwise_sep_conv2d(depth_sep_10, is_training, 512, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_11')
            g(depth_sep_11)
            # [None x 4 x 4 x 512]

            depth_sep_12 = self._depthwise_sep_conv2d(depth_sep_11, is_training, 1024, (3, 3), stride=2, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_12')
            g(depth_sep_12)
            # [None x 2 x 2 x 1024]

            depth_sep_13 = self._depthwise_sep_conv2d(depth_sep_12, is_training, 1024, (3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True, scope_name='dw_block_13')
            g(depth_sep_13)
            # [None x 2 x 2 x 1024]
            
            with tf.variable_scope("tail"):                
                gap = tf.reduce_mean(depth_sep_13, axis=[1, 2], name='global_avg_pool')
                g(gap)
                # [None x 1024]
                
                fc_logits = tf.layers.dense(gap, self.num_classes, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='fc_layer')
                fc_op = tf.nn.softmax(fc_logits, name='softmax_op')
                g(fc_op)
                # [B x 200]
                
            return fc_logits, fc_op
