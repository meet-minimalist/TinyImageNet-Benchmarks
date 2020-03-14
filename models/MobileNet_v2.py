# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.MobileNet_v2_config as MobileNet_v2_config
from models.MobileNet_v2_config import g

class MobileNet_v2:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'MobileNet_v2'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()  # as per paper
        # variance scaling is good for models with relu activation
        # whereas xavier initialization is good for models with sigmoid activation
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=MobileNet_v2_config.weight_decay)
    
    def _depthwise_conv(self, ip, training, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True):
        dw_weights_shape = [k_size[0], k_size[1], ip.get_shape().as_list()[-1], depth_mul]
        
        w = tf.get_variable('dw_weight', shape=dw_weights_shape, dtype=tf.float32, initializer=self.k_init)
        # no need of using regularizer in depthwise convo as there are very less number of parameters. -- as per paper
        
        depthwise = tf.nn.depthwise_conv2d(ip, w, (1, stride, stride, 1), padding)
        if use_batch_norm:
            depthwise = tf.layers.batch_normalization(depthwise, momentum=0.9, training=training)
        depthwise = tf.nn.relu6(depthwise)
        return depthwise

    
    def _pointwise_conv(self, ip, training, op_filters, use_batch_norm=True):
        pointwise = tf.layers.conv2d(ip, op_filters, (1, 1), 1, 'SAME', use_bias=not use_batch_norm, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        # we can use kernel regularizer in pointwise convolution
        if use_batch_norm:
            pointwise = tf.layers.batch_normalization(pointwise, momentum=0.9, training=training)
        # No activation to be applied here as this is only a linear layer. Only batchnorm can be applied here.
        return pointwise

        
    def _inv_res_block(self, ip, training, op_filters, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=1.0, use_batch_norm=True, scope_name='inv_res_block'):
        with tf.variable_scope(scope_name):
            expanded_filters = ip.get_shape().as_list()[-1] * expansion_factor
            bottleneck_conv = tf.layers.conv2d(ip, expanded_filters, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            bottleneck_conv = tf.layers.batch_normalization(bottleneck_conv, momentum=0.9, training=training)
            bottleneck_conv = tf.nn.relu6(bottleneck_conv)

            depth_wise_conv = self._depthwise_conv(bottleneck_conv, training, k_size, stride, padding, depth_mul, use_batch_norm)

            pointwise_conv = self._pointwise_conv(depth_wise_conv, training, op_filters, use_batch_norm)
            
            if stride == 2:
                return pointwise_conv
            elif stride == 1:
                if ip.get_shape().as_list()[-1] != op_filters:
                    strided_ip = tf.layers.conv2d(ip, op_filters, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                    return strided_ip + pointwise_conv
                else:
                    return ip + pointwise_conv
    

    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 32, (3, 3), 2, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, momentum=0.9, training=is_training)
                conv1 = tf.nn.relu6(conv1)
                g(conv1)
                # [None x 32 x 32 x 32]
                
            inv_res_block_1_1 = self._inv_res_block(conv1, is_training, 16, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=1.0, use_batch_norm=True, scope_name='inv_res_block_1_1')
            g(inv_res_block_1_1)
            # [None x 32 x 32 x 16]
            
            inv_res_block_2_1 = self._inv_res_block(inv_res_block_1_1, is_training, 24, (3, 3), stride=2, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_2_1')
            inv_res_block_2_2 = self._inv_res_block(inv_res_block_2_1, is_training, 24, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_2_2')
            g(inv_res_block_2_2)
            # [None x 16 x 16 x 24]

            inv_res_block_3_1 = self._inv_res_block(inv_res_block_2_2, is_training, 32, (3, 3), stride=2, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_3_1')
            inv_res_block_3_2 = self._inv_res_block(inv_res_block_3_1, is_training, 32, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_3_2')
            inv_res_block_3_3 = self._inv_res_block(inv_res_block_3_2, is_training, 32, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_3_3')
            g(inv_res_block_3_3)
            # [None x 8 x 8 x 32]

            inv_res_block_4_1 = self._inv_res_block(inv_res_block_3_3, is_training, 64, (3, 3), stride=2, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_4_1')
            inv_res_block_4_2 = self._inv_res_block(inv_res_block_4_1, is_training, 64, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_4_2')
            inv_res_block_4_3 = self._inv_res_block(inv_res_block_4_2, is_training, 64, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_4_3')
            inv_res_block_4_4 = self._inv_res_block(inv_res_block_4_3, is_training, 64, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_4_4')
            g(inv_res_block_4_4)
            # [None x 4 x 4 x 64]

            inv_res_block_5_1 = self._inv_res_block(inv_res_block_4_4, is_training, 96, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_5_1')
            inv_res_block_5_2 = self._inv_res_block(inv_res_block_5_1, is_training, 96, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_5_2')
            inv_res_block_5_3 = self._inv_res_block(inv_res_block_5_2, is_training, 96, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_5_3')
            g(inv_res_block_5_3)
            # [None x 4 x 4 x 96]

            inv_res_block_6_1 = self._inv_res_block(inv_res_block_5_3, is_training, 160, (3, 3), stride=2, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_6_1')
            inv_res_block_6_2 = self._inv_res_block(inv_res_block_6_1, is_training, 160, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_6_2')
            inv_res_block_6_3 = self._inv_res_block(inv_res_block_6_2, is_training, 160, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_6_3')
            g(inv_res_block_6_3)
            # [None x 2 x 2 x 160]

            inv_res_block_7_1 = self._inv_res_block(inv_res_block_6_3, is_training, 320, (3, 3), stride=1, padding='SAME', depth_mul=1.0, expansion_factor=6.0, use_batch_norm=True, scope_name='inv_res_block_7_1')
            g(inv_res_block_7_1)
            # [None x 2 x 2 x 320]

            with tf.variable_scope("conv2"):    
                conv2 = tf.layers.conv2d(inv_res_block_7_1, 1280, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv2 = tf.layers.batch_normalization(conv2, momentum=0.9, training=is_training)
                conv2 = tf.nn.relu6(conv2)
                g(conv2)
                # [None x 2 x 2 x 1280]

            
            with tf.variable_scope("tail"):                
                gap = tf.reduce_mean(conv2, axis=[1, 2], name='global_avg_pool', keepdims=True)
                g(gap)
                # [None x 1 x 1 x 1280]
                   
                conv3 = tf.layers.conv2d(gap, self.num_classes, (1, 1), 1, 'SAME', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                g(conv3)
                # [None x 1 x 1 x 200]
                logits = tf.squeeze(conv3, axis=[1, 2], name='logits')
                g(logits)
                # [None x 200]
                
                outputs = tf.nn.softmax(logits, name='softmax_op')
                g(outputs)
                # [B x 200]
            
            return logits, outputs
