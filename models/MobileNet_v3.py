# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.MobileNet_v3_config as MobileNet_v3_config
from models.MobileNet_v3_config import g

class MobileNet_v3:
    def __init__(self, input_dims=(64, 64), num_classes=200, mode='large'):
        self.model_name = 'MobileNet_v3'
        self.mode = mode
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()  # as per paper
        # variance scaling is good for models with relu activation
        # whereas xavier initialization is good for models with sigmoid activation
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=MobileNet_v3_config.weight_decay)
    
    
    def relu6(self, x):
        return tf.nn.relu6(x)
    
    def hard_sigmoid(self, x):
        return (tf.nn.relu6(x + 3)) / 6
    
    def hard_swish(self, x):
        return x * self.hard_sigmoid(x)
    
    def _depthwise_conv(self, ip, training, activation=None, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, use_batch_norm=True):
        dw_weights_shape = [k_size[0], k_size[1], ip.get_shape().as_list()[-1], depth_mul]
        
        w = tf.get_variable('dw_weight', shape=dw_weights_shape, dtype=tf.float32, initializer=self.k_init, regularizer=self.k_reg)
        
        depthwise = tf.nn.depthwise_conv2d(ip, w, (1, stride, stride, 1), padding)
        if use_batch_norm:
            depthwise = tf.layers.batch_normalization(depthwise, training=training)
        depthwise = activation(depthwise)
        return depthwise

    
    def _pointwise_conv(self, ip, training, op_filters, use_batch_norm=True):
        pointwise = tf.layers.conv2d(ip, op_filters, (1, 1), 1, 'SAME', use_bias=not use_batch_norm, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        # we can use kernel regularizer in pointwise convolution
        if use_batch_norm:
            pointwise = tf.layers.batch_normalization(pointwise, training=training)
        # No activation to be applied here as this is only a linear layer. Only batchnorm can be applied here.
        return pointwise

    def _se_module(self, ip, ratio):
        # [B x H x W x C]
        with tf.variable_scope('se_module'):
            squeeze_filter_num = int(ip.get_shape().as_list()[-1] * ratio) 
            gap = tf.reduce_mean(ip, axis=[1, 2])
            # [B x C]

            fc_1 = tf.layers.dense(gap, squeeze_filter_num, activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            fc_1 = self.relu6(fc_1)
            # [B x C/4]
            
            fc_2 = tf.layers.dense(fc_1, ip.get_shape().as_list()[-1], activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            fc_2 = self.hard_sigmoid(fc_2)
            # [B x C]

            se_multiplier = tf.reshape(fc_2, [-1, 1, 1, ip.get_shape().as_list()[-1]])
            # [B x 1 x 1 x C]

            scaled_ip = ip * se_multiplier
            
            return scaled_ip


    def bottelneck_layer(self, ip, training, expanded_size, op_filters, k_size=(3, 3), stride=1, padding='SAME', activation='RE', apply_se=False, depth_mul=1.0, se_ratio=1.0, use_batch_norm=True, scope_name='bottleneck'):
        if activation == 'RE':
            act = self.relu6
        elif activation == 'HS':
            act = self.hard_swish
        
        
        with tf.variable_scope(scope_name):
            bottleneck_conv = tf.layers.conv2d(ip, expanded_size, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            bottleneck_conv = tf.layers.batch_normalization(bottleneck_conv, training=training)
            bottleneck_conv = act(bottleneck_conv)

            depth_wise_conv = self._depthwise_conv(bottleneck_conv, training, act, k_size, stride, padding, depth_mul, use_batch_norm)

            if apply_se:
                depth_wise_conv = self._se_module(depth_wise_conv, se_ratio)
            
            pointwise_conv = self._pointwise_conv(depth_wise_conv, training, op_filters, use_batch_norm)
            
            if stride == 2:
                return pointwise_conv
            elif stride == 1:
                if ip.get_shape().as_list()[-1] != op_filters:
                    strided_ip = tf.layers.conv2d(ip, op_filters, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                    return strided_ip + pointwise_conv
                else:
                    return ip + pointwise_conv
    
    def mnetv3_large(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 16, (3, 3), 2, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, training=is_training)
                conv1 = self.hard_swish(conv1)
                g(conv1)
                # [None x 32 x 32 x 16]
                
            bneck_1 = self.bottelneck_layer(conv1, is_training, 16, 16, (3, 3), stride=1, padding='SAME', activation='RE', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_1')
            bneck_2 = self.bottelneck_layer(bneck_1, is_training, 64, 24, (3, 3), stride=2, padding='SAME', activation='RE', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_2')
            g(bneck_2)
            # [None x 16 x 16 x 24]

            bneck_3 = self.bottelneck_layer(bneck_2, is_training, 72, 24, (3, 3), stride=1, padding='SAME', activation='RE', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_3')
            bneck_4 = self.bottelneck_layer(bneck_3, is_training, 72, 40, (5, 5), stride=2, padding='SAME', activation='RE', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_4')
            g(bneck_4)
            # [None x 8 x 8 x 40]

            bneck_5 = self.bottelneck_layer(bneck_4, is_training, 120, 40, (5, 5), stride=1, padding='SAME', activation='RE', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_5')
            bneck_6 = self.bottelneck_layer(bneck_5, is_training, 120, 40, (5, 5), stride=1, padding='SAME', activation='RE', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_6')
            g(bneck_5)
            # [None x 8 x 8 x 40]

            bneck_7 = self.bottelneck_layer(bneck_6, is_training, 240, 80, (3, 3), stride=2, padding='SAME', activation='HS', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_7')
            bneck_8 = self.bottelneck_layer(bneck_7, is_training, 200, 80, (3, 3), stride=1, padding='SAME', activation='HS', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_8')
            g(bneck_7)
            # [None x 4 x 4 x 80]

            bneck_9 = self.bottelneck_layer(bneck_8, is_training, 184, 80, (3, 3), stride=1, padding='SAME', activation='HS', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_9')
            bneck_10 = self.bottelneck_layer(bneck_9, is_training, 184, 80, (3, 3), stride=1, padding='SAME', activation='HS', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_10')
            g(bneck_9)
            # [None x 4 x 4 x 80]

            bneck_11 = self.bottelneck_layer(bneck_10, is_training, 480, 112, (3, 3), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_11')
            bneck_12 = self.bottelneck_layer(bneck_11, is_training, 672, 112, (3, 3), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_12')
            g(bneck_12)
            # [None x 4 x 4 x 112]

            bneck_13 = self.bottelneck_layer(bneck_12, is_training, 672, 160, (5, 5), stride=2, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_13')
            bneck_14 = self.bottelneck_layer(bneck_13, is_training, 960, 160, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_14')
            g(bneck_14)
            # [None x 2 x 2 x 160]

            bneck_15 = self.bottelneck_layer(bneck_14, is_training, 960, 160, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_15')
            g(bneck_15)
            # [None x 2 x 2 x 160]
            
            
            with tf.variable_scope("conv2"):    
                conv2 = tf.layers.conv2d(bneck_15, 960, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv2 = tf.layers.batch_normalization(conv2, training=is_training)
                conv2 = self.hard_swish(conv2)
                g(conv2)
                # [None x 2 x 2 x 960]

            
            with tf.variable_scope("tail"):                
                gap = tf.reduce_mean(conv2, axis=[1, 2], name='global_avg_pool', keepdims=True)
                g(gap)
                # [None x 1 x 1 x 960]
                   
                conv3 = tf.layers.conv2d(gap, 1280, (1, 1), 1, 'SAME', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='conv3')
                conv3 = self.hard_swish(conv3)
                g(conv3)
                # [None x 1 x 1 x 1280]
                
                conv4 = tf.layers.conv2d(conv3, self.num_classes, (1, 1), 1, 'SAME', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='conv4')
                g(conv4)
                # [None x 1 x 1 x 200]

                logits = tf.squeeze(conv4, axis=[1, 2], name='logits')
                g(logits)
                # [None x 200]
                
                outputs = tf.nn.softmax(logits, name='softmax_op')
                g(outputs)
                # [B x 200]
            
            return logits, outputs


    def mnetv3_small(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 16, (3, 3), 2, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, training=is_training)
                conv1 = self.hard_swish(conv1)
                g(conv1)
                # [None x 32 x 32 x 16]
                
            bneck_1 = self.bottelneck_layer(conv1, is_training, 16, 16, (3, 3), stride=2, padding='SAME', activation='RE', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_1')
            g(bneck_1)
            # [None x 16 x 16 x 16]

            bneck_2 = self.bottelneck_layer(bneck_1, is_training, 72, 24, (3, 3), stride=2, padding='SAME', activation='RE', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_2')
            g(bneck_2)
            # [None x 8 x 8 x 24]

            bneck_3 = self.bottelneck_layer(bneck_2, is_training, 88, 24, (3, 3), stride=1, padding='SAME', activation='RE', apply_se=False, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_3')
            bneck_4 = self.bottelneck_layer(bneck_3, is_training, 96, 40, (5, 5), stride=2, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_4')
            g(bneck_4)
            # [None x 4 x 4 x 40]

            bneck_5 = self.bottelneck_layer(bneck_4, is_training, 240, 40, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_5')
            bneck_6 = self.bottelneck_layer(bneck_5, is_training, 240, 40, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_6')
            g(bneck_5)
            # [None x 4 x 4 x 40]

            bneck_7 = self.bottelneck_layer(bneck_6, is_training, 120, 48, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_7')
            bneck_8 = self.bottelneck_layer(bneck_7, is_training, 144, 48, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_8')
            g(bneck_7)
            # [None x 4 x 4 x 48]

            bneck_9 = self.bottelneck_layer(bneck_8, is_training, 288, 96, (5, 5), stride=2, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_9')
            bneck_10 = self.bottelneck_layer(bneck_9, is_training, 576, 96, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_10')
            bneck_11 = self.bottelneck_layer(bneck_10, is_training, 576, 96, (5, 5), stride=1, padding='SAME', activation='HS', apply_se=True, depth_mul=1.0, se_ratio=1.0/4.0, use_batch_norm=True, scope_name='bottleneck_11')
            g(bneck_11)
            # [None x 2 x 2 x 96]

            
            with tf.variable_scope("conv2"):    
                conv2 = tf.layers.conv2d(bneck_11, 576, (1, 1), 1, 'SAME', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv2 = tf.layers.batch_normalization(conv2, training=is_training)
                conv2 = self.hard_swish(conv2)
                conv2 = self._se_module(conv2, 1.0/4.0)
                g(conv2)
                # [None x 2 x 2 x 576]

            
            with tf.variable_scope("tail"):                
                gap = tf.reduce_mean(conv2, axis=[1, 2], name='global_avg_pool', keepdims=True)
                g(gap)
                # [None x 1 x 1 x 576]
                   
                conv3 = tf.layers.conv2d(gap, 1024, (1, 1), 1, 'SAME', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='conv3')
                conv3 = self.hard_swish(conv3)
                g(conv3)
                # [None x 1 x 1 x 1024]
                
                conv4 = tf.layers.conv2d(conv3, self.num_classes, (1, 1), 1, 'SAME', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='conv4')
                g(conv4)
                # [None x 1 x 1 x 200]

                logits = tf.squeeze(conv4, axis=[1, 2], name='logits')
                g(logits)
                # [None x 200]
                
                outputs = tf.nn.softmax(logits, name='softmax_op')
                g(outputs)
                # [B x 200]
            
            return logits, outputs



    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]
        if self.mode == 'large':
            logits, outputs = self.mnetv3_large(x, is_training)
        elif self.mode == 'small':
            logits, outputs = self.mnetv3_small(x, is_training)
        return logits, outputs