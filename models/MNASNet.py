# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:45:12 2020

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.MNASNet_config as MNASNet_config
from models.MNASNet_config import g

class MNASNet:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'MNASNet'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()  # as per paper
        # variance scaling is good for models with relu activation
        # whereas xavier initialization is good for models with sigmoid activation
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=MNASNet_config.weight_decay)
    
    def _depthwise_conv(self, ip, training, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True):
        dw_weights_shape = [k_size[0], k_size[1], ip.get_shape().as_list()[-1], depth_mul]
        
        w = tf.get_variable('dw_weight', shape=dw_weights_shape, dtype=tf.float32, initializer=self.k_init)
        # no need of using regularizer in depthwise convo as there are very less number of parameters. -- as per paper
        
        depthwise = tf.nn.depthwise_conv2d(ip, w, (1, stride, stride, 1), padding)
        if use_batch_norm:
            depthwise = tf.layers.batch_normalization(depthwise, momentum=0.99, training=training)
        depthwise = tf.nn.relu6(depthwise)
        return depthwise

    def _pointwise_conv(self, ip, training, op_filters, use_batch_norm=True):
        pointwise = tf.layers.conv2d(ip, op_filters, (1, 1), 1, 'SAME', use_bias=not use_batch_norm, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        # we can use kernel regularizer in pointwise convolution
        if use_batch_norm:
            pointwise = tf.layers.batch_normalization(pointwise, momentum=0.99, training=training)
        # No activation to be applied here as this is only a linear layer. Only batchnorm can be applied here.
        return pointwise

    def _se_block(self, x, se_ratio=0.25):
        with tf.variable_scope('se_block'):
            num_ip_filters = x.get_shape().as_list()[-1]
            reduced_filters = int(num_ip_filters * se_ratio)

            squeezed = tf.reduce_mean(x, axis=[1, 2])
            
            excite_reduced = tf.layers.dense(squeezed, reduced_filters, activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            excite_expanded = tf.layers.dense(excite_reduced, num_ip_filters, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            excite_expanded = tf.reshape(excite_expanded, [-1, 1, 1, excite_expanded.get_shape().as_list()[-1]])
            return x * excite_expanded            

    def _mb_block(self, x, is_training, filters, expansion_factor, k_size, stride, padding, apply_se, name):
        with tf.variable_scope(name):
            dw_filters = x.get_shape().as_list()[-1] * expansion_factor
            bottleneck_conv = tf.layers.conv2d(x, dw_filters, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            bottleneck_conv = tf.layers.batch_normalization(bottleneck_conv, momentum=0.99, training=is_training)
            bottleneck_conv = tf.nn.relu(bottleneck_conv)

            depth_wise_conv = self._depthwise_conv(bottleneck_conv, is_training, k_size, stride, padding, 1.0, True)

            if apply_se:
                depth_wise_conv = self._se_block(depth_wise_conv, se_ratio=0.25)

            pointwise_conv = self._pointwise_conv(depth_wise_conv, is_training, filters, True)
            
            if stride == 1 and x.get_shape().as_list()[-1] == filters:
                return x + pointwise_conv
            else:
                return pointwise_conv


    def _separable_block(self, x, is_training, filters, k_size, stride, padding, name):
        with tf.variable_scope(name):
            depth_wise_conv = self._depthwise_conv(x, is_training, k_size, stride, padding, depth_mul=1.0, use_batch_norm=True)

            conv = tf.layers.conv2d(depth_wise_conv, filters, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv = tf.layers.batch_normalization(conv, momentum=0.99, training=is_training)            
            return conv

    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 32, (3, 3), 2, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, momentum=0.99, training=is_training)
                conv1 = tf.nn.relu(conv1)
                g(conv1)
                # [None x 32 x 32 x 32]
                
            sep_block_1 = self._separable_block(conv1, is_training, 16, (3, 3), stride=1, padding='SAME', name='sep_block_1')
            g(sep_block_1)
            # [None x 32 x 32 x 16]

            mb_conv_1 = self._mb_block(sep_block_1, is_training, 24, 6, (3, 3), stride=2, padding='SAME', apply_se=False, name='mb_block_6_1_1')
            g(mb_conv_1)
            mb_conv_2 = self._mb_block(mb_conv_1, is_training, 24, 6, (3, 3), stride=1, padding='SAME', apply_se=False, name='mb_block_6_1_2')
            g(mb_conv_2)
            # [None x 16 x 16 x 24]

            mb_conv_3 = self._mb_block(mb_conv_2, is_training, 40, 3, (5, 5), stride=2, padding='SAME', apply_se=True, name='mb_block_3_1_1')
            g(mb_conv_3)
            mb_conv_4 = self._mb_block(mb_conv_3, is_training, 40, 3, (5, 5), stride=1, padding='SAME', apply_se=True, name='mb_block_3_1_2')
            g(mb_conv_4)
            mb_conv_5 = self._mb_block(mb_conv_4, is_training, 40, 3, (5, 5), stride=1, padding='SAME', apply_se=True, name='mb_block_3_1_3')
            g(mb_conv_5)
            # [None x 8 x 8 x 40]            

            mb_conv_6 = self._mb_block(mb_conv_5, is_training, 80, 6, (3, 3), stride=2, padding='SAME', apply_se=False, name='mb_block_6_2_1')
            g(mb_conv_6)
            mb_conv_7 = self._mb_block(mb_conv_6, is_training, 80, 6, (3, 3), stride=1, padding='SAME', apply_se=False, name='mb_block_6_2_2')
            g(mb_conv_7)
            mb_conv_8 = self._mb_block(mb_conv_7, is_training, 80, 6, (3, 3), stride=1, padding='SAME', apply_se=False, name='mb_block_6_2_3')
            g(mb_conv_8)
            mb_conv_9 = self._mb_block(mb_conv_8, is_training, 80, 6, (3, 3), stride=1, padding='SAME', apply_se=False, name='mb_block_6_2_4')
            g(mb_conv_9)
            # [None x 4 x 4 x 80]

            mb_conv_10 = self._mb_block(mb_conv_9, is_training, 112, 6, (3, 3), stride=1, padding='SAME', apply_se=True, name='mb_block_6_3_1')
            g(mb_conv_10)
            mb_conv_11 = self._mb_block(mb_conv_10, is_training, 112, 6, (3, 3), stride=1, padding='SAME', apply_se=True, name='mb_block_6_3_2')
            g(mb_conv_11)
            # [None x 4 x 4 x 112]


            mb_conv_12 = self._mb_block(mb_conv_11, is_training, 160, 6, (5, 5), stride=2, padding='SAME', apply_se=True, name='mb_block_6_4_1')
            g(mb_conv_12)
            mb_conv_13 = self._mb_block(mb_conv_12, is_training, 160, 6, (5, 5), stride=1, padding='SAME', apply_se=True, name='mb_block_6_4_2')
            g(mb_conv_13)
            mb_conv_14 = self._mb_block(mb_conv_13, is_training, 160, 6, (5, 5), stride=1, padding='SAME', apply_se=True, name='mb_block_6_4_3')
            g(mb_conv_14)
            # [None x 2 x 2 x 160]

            mb_conv_15 = self._mb_block(mb_conv_14, is_training, 320, 6, (3, 3), stride=1, padding='SAME', apply_se=False, name='mb_block_6_5')
            g(mb_conv_15)
            # [None x 2 x 2 x 320]

            with tf.variable_scope("tail"):                
                k_size = mb_conv_15.get_shape().as_list()[1:3]
                gap = tf.layers.average_pooling2d(mb_conv_15, k_size, (1, 1), 'valid')
                g(gap)
                # [None x 1 x 1 x 320]
                gap = tf.layers.dropout(tf.squeeze(gap, axis=[1, 2]), rate=0.2)
                g(gap)
                # [None x 320]

                logits = tf.layers.dense(gap, self.num_classes, activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                g(logits)
                # [None x 200]
                
                outputs = tf.nn.softmax(logits, name='softmax_op')
                g(outputs)
                # [None x 200]
            
            return logits, outputs
