# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:13:20 2020

@author: Meet
"""

# This code has been inspired and taken from : https://github.com/qubvel/efficientnet/

'''
Model name			width_coeff, depth_coeff, default_res, dropout_rate

EfficientNetB0		1.0, 		 1.0, 		  224, 		   0.2

EfficientNetB1		1.0, 		 1.1, 		  240,  	   0.2

EfficientNetB2		1.1, 		 1.2, 		  260,  	   0.3
		
EfficientNetB3		1.2, 		 1.4, 		  300, 		   0.3
		
EfficientNetB4		1.4, 		 1.8, 		  380, 		   0.4
		
EfficientNetB5		1.6, 		 2.2, 		  456, 		   0.4

EfficientNetB6		1.8, 		 2.6, 		  528,   	   0.5
		
EfficientNetB7		2.0, 		 3.1, 		  600, 		   0.5

EfficientNetL2 		4.3, 		 5.3, 		  800, 		   0.5
'''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.EfficientNet_config as EfficientNet_config
from models.EfficientNet_config import g
import math
import numpy as np

class EfficientNet:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'EfficientNet'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()  # as per paper
        # variance scaling is good for models with relu activation
        # whereas xavier initialization is good for models with sigmoid activation
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=EfficientNet_config.weight_decay)
        
        
    def round_filters(self, filters, width_coefficient, depth_divisor):
        """Round number of filters based on width multiplier."""

        filters *= width_coefficient
        new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
        new_filters = max(depth_divisor, new_filters)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += depth_divisor
        return int(new_filters)


    def round_repeats(self, repeats, depth_coefficient):
        """Round number of repeats based on depth multiplier."""

        return int(math.ceil(depth_coefficient * repeats))


    def swish_act(self, x):
        return x * tf.nn.sigmoid(x)

    def _depthwise_conv(self, ip, training, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True):
        dw_weights_shape = [k_size[0], k_size[1], ip.get_shape().as_list()[-1], depth_mul]
        
        w = tf.get_variable('dw_weight', shape=dw_weights_shape, dtype=tf.float32, initializer=self.k_init)
        # no need of using regularizer in depthwise convo as there are very less number of parameters. -- as per paper
        
        depthwise = tf.nn.depthwise_conv2d(ip, w, (1, stride, stride, 1), padding)
        if use_batch_norm:
            depthwise = tf.layers.batch_normalization(depthwise, momentum=0.99, training=training)
        depthwise = self.swish_act(depthwise)
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
            
            excite_reduced = tf.layers.dense(squeezed, reduced_filters, activation=self.swish_act, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            excite_expanded = tf.layers.dense(excite_reduced, num_ip_filters, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            excite_expanded = tf.reshape(excite_expanded, [-1, 1, 1, excite_expanded.get_shape().as_list()[-1]])
            return x * excite_expanded            

    def _mb_block(self, x, is_training, filters, expansion_factor, k_size, stride, padding, apply_se, name):
        with tf.variable_scope(name):
            
            ip_filters = self.round_filters(x.get_shape().as_list()[-1], EfficientNet_config.width_coefficient, EfficientNet_config.depth_divisor)
            dw_filters = ip_filters * expansion_factor

            op_filters = self.round_filters(filters, EfficientNet_config.width_coefficient, EfficientNet_config.depth_divisor) 

            bottleneck_conv = tf.layers.conv2d(x, dw_filters, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            bottleneck_conv = tf.layers.batch_normalization(bottleneck_conv, momentum=0.99, training=is_training)
            bottleneck_conv = self.swish_act(bottleneck_conv)

            depth_wise_conv = self._depthwise_conv(bottleneck_conv, is_training, k_size, stride, padding, 1.0, True)

            if apply_se:
                depth_wise_conv = self._se_block(depth_wise_conv, se_ratio=0.25)

            pointwise_conv = self._pointwise_conv(depth_wise_conv, is_training, op_filters, True)
            
            if stride == 1 and x.get_shape().as_list()[-1] == op_filters:
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

        new_size = [int(np.ceil(64 * EfficientNet_config.resolution_coefficient)), int(np.ceil(64 * EfficientNet_config.resolution_coefficient))]
        x = tf.image.resize_nearest_neighbor(x, new_size)

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, self.round_filters(32, EfficientNet_config.width_coefficient, EfficientNet_config.depth_divisor), (3, 3), 2, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, momentum=0.99, training=is_training)
                conv1 = self.swish_act(conv1)
                g(conv1)
                # [None x 32 x 32 x 32]

            mb_conv_1 = conv1
            for i in range(self.round_repeats(1, EfficientNet_config.depth_coefficient)):
                mb_conv_1 = self._mb_block(mb_conv_1, is_training, 16, 1, (3, 3), stride=1, padding='SAME', apply_se=False, name='mb_block_1_' + str(i + 1))
                g(mb_conv_1)
            # [None x 32 x 32 x 16]

            mb_conv_2 = mb_conv_1
            for i in range(self.round_repeats(2, EfficientNet_config.depth_coefficient)):
                if i == 0:
                    stride_ = 2
                else:
                    stride_ = 1
                mb_conv_2 = self._mb_block(mb_conv_2, is_training, 24, 6, (3, 3), stride=stride_, padding='SAME', apply_se=False, name='mb_block_2_' + str(i + 1))
                g(mb_conv_2)
            # [None x 16 x 16 x 24]            

            mb_conv_3 = mb_conv_2
            for i in range(self.round_repeats(2, EfficientNet_config.depth_coefficient)):
                if i == 0:
                    stride_ = 2
                else:
                    stride_ = 1
                mb_conv_3 = self._mb_block(mb_conv_3, is_training, 40, 6, (5, 5), stride=stride_, padding='SAME', apply_se=False, name='mb_block_3_' + str(i + 1))
                g(mb_conv_3)
            # [None x 8 x 8 x 40]

            mb_conv_4 = mb_conv_3
            for i in range(self.round_repeats(3, EfficientNet_config.depth_coefficient)):
                if i == 0:
                    stride_ = 2
                else:
                    stride_ = 1
                mb_conv_4 = self._mb_block(mb_conv_4, is_training, 80, 6, (3, 3), stride=stride_, padding='SAME', apply_se=False, name='mb_block_4_' + str(i + 1))
                g(mb_conv_4)
            # [None x 4 x 4 x 80]

            mb_conv_5 = mb_conv_4
            for i in range(self.round_repeats(3, EfficientNet_config.depth_coefficient)):
                mb_conv_5 = self._mb_block(mb_conv_5, is_training, 112, 6, (5, 5), stride=1, padding='SAME', apply_se=False, name='mb_block_5_' + str(i + 1))
                g(mb_conv_5)
            # [None x 4 x 4 x 112]

            mb_conv_6 = mb_conv_5
            for i in range(self.round_repeats(4, EfficientNet_config.depth_coefficient)):
                if i == 0:
                    stride_ = 2
                else:
                    stride_ = 1
                mb_conv_6 = self._mb_block(mb_conv_6, is_training, 192, 6, (5, 5), stride=stride_, padding='SAME', apply_se=False, name='mb_block_6_' + str(i + 1))
                g(mb_conv_6)
            # [None x 2 x 2 x 192]

            mb_conv_7 = mb_conv_6
            for i in range(self.round_repeats(1, EfficientNet_config.depth_coefficient)):
                mb_conv_7 = self._mb_block(mb_conv_7, is_training, 320, 6, (3, 3), stride=1, padding='SAME', apply_se=False, name='mb_block_7_' + str(i + 1))
                g(mb_conv_7)
            # [None x 2 x 2 x 320]

            with tf.variable_scope("tail"):                
                conv2 = tf.layers.conv2d(mb_conv_7, self.round_filters(1280, EfficientNet_config.width_coefficient, EfficientNet_config.depth_divisor), (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv2 = tf.layers.batch_normalization(conv2, momentum=0.99, training=is_training)
                conv2 = self.swish_act(conv2)
                g(conv2)
                # [None x 2 x 2 x 1280]
                
                k_size = conv2.get_shape().as_list()[1:3]
                gap = tf.layers.average_pooling2d(conv2, k_size, (1, 1), 'valid')
                g(gap)
                # [None x 1 x 1 x 1280]
                gap = tf.layers.dropout(tf.squeeze(gap, axis=[1, 2]), rate=EfficientNet_config.dropout)
                g(gap)
                # [None x 1280]

                logits = tf.layers.dense(gap, self.num_classes, activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                g(logits)
                # [None x 200]
                
                outputs = tf.nn.softmax(logits, name='softmax_op')
                g(outputs)
                # [None x 200]
                
            return logits, outputs
