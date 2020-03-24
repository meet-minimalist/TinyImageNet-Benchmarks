# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:28:20 2020

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.Xception_config as Xception_config
from models.Xception_config import g

class Xception:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'Xception'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=Xception_config.weight_decay)
    

    def _depthwise_conv(self, ip, training, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0):
        dw_weights_shape = [k_size[0], k_size[1], ip.get_shape().as_list()[-1], depth_mul]
        
        w = tf.get_variable('dw_weight', shape=dw_weights_shape, dtype=tf.float32, initializer=self.k_init)
        # no need of using regularizer in depthwise convo as there are very less number of parameters. -- as per paper
        
        depthwise = tf.nn.depthwise_conv2d(ip, w, (1, stride, stride, 1), padding)
        depthwise = tf.layers.batch_normalization(depthwise, training=training)
        return depthwise

    
    def _pointwise_conv(self, ip, training, op_filters):
        pointwise = tf.layers.conv2d(ip, op_filters, (1, 1), 1, 'SAME', use_bias=False, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        # we can use kernel regularizer in pointwise convolution
        # No activation to be applied here as this is only a linear layer.
        return pointwise

        
    def _separable_block(self, ip, training, op_filters, k_size=(3, 3), stride=1, padding='SAME', depth_mul=1.0, name='sep_block'):
        with tf.variable_scope(name):
            depth_wise_conv = self._depthwise_conv(ip, training, k_size, 1, padding, depth_mul)     # No activation to be applied here
            pointwise_conv = self._pointwise_conv(depth_wise_conv, training, op_filters)
            return pointwise_conv
            
    
    def _entry_block(self, x, is_training, filters, apply_relu_on_ip=True, name='entry_block'):
        with tf.variable_scope(name):
            if apply_relu_on_ip:
                relu_x = tf.nn.relu(x)
            else:
                relu_x = x

            sep_block_1 = self._separable_block(relu_x, is_training, filters, k_size=(3, 3), stride=1, name='sep_block_1')
            sep_block_1 = tf.layers.batch_normalization(sep_block_1, training=is_training)        
            sep_block_1 = tf.nn.relu(sep_block_1)

            sep_block_2 = self._separable_block(sep_block_1, is_training, filters, k_size=(3, 3), stride=1, name='sep_block_2')
            sep_block_2 = tf.layers.batch_normalization(sep_block_2, training=is_training)        
            
            sep_block_mxpool = tf.layers.max_pooling2d(sep_block_2, (3, 3), (2, 2), 'same')

            residual = tf.layers.conv2d(x, filters, (1, 1), strides=2, padding='same', use_bias=False, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            residual = tf.layers.batch_normalization(residual, training=is_training)
            
            return tf.add(sep_block_mxpool, residual)


    
    def _middle_block(self, x, is_training, filters, name='middle_block'):
        with tf.variable_scope(name):
            relu_x = tf.nn.relu(x)
            
            sep_block_1 = self._separable_block(relu_x, is_training, filters, k_size=(3, 3), stride=1, name='sep_block_1')
            sep_block_1 = tf.layers.batch_normalization(sep_block_1, training=is_training)        
            sep_block_1 = tf.nn.relu(sep_block_1)

            sep_block_2 = self._separable_block(sep_block_1, is_training, filters, k_size=(3, 3), stride=1, name='sep_block_2')
            sep_block_2 = tf.layers.batch_normalization(sep_block_2, training=is_training)        
            sep_block_2 = tf.nn.relu(sep_block_2)

            sep_block_3 = self._separable_block(sep_block_2, is_training, filters, k_size=(3, 3), stride=1, name='sep_block_3')
            sep_block_3 = tf.layers.batch_normalization(sep_block_3, training=is_training)
            
            return tf.add(sep_block_3, x)



    def _exit_block(self, x, is_training, filters_list):
        with tf.variable_scope('block_1'):
            relu_x = tf.nn.relu(x)
            
            sep_block_1 = self._separable_block(relu_x, is_training, filters_list[0], k_size=(3, 3), stride=1, name='sep_block_1')
            sep_block_1 = tf.layers.batch_normalization(sep_block_1, training=is_training)        
            sep_block_1 = tf.nn.relu(sep_block_1)

            sep_block_2 = self._separable_block(sep_block_1, is_training, filters_list[1], k_size=(3, 3), stride=1, name='sep_block_2')
            sep_block_2 = tf.layers.batch_normalization(sep_block_2, training=is_training)        
            
            sep_block_mxpool = tf.layers.max_pooling2d(sep_block_2, (3, 3), (2, 2), 'same')

            residual = tf.layers.conv2d(x, filters_list[1], (1, 1), strides=2, padding='same', use_bias=False, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            residual = tf.layers.batch_normalization(residual, training=is_training)
            
            exit_block_1 = tf.add(sep_block_mxpool, residual)
            g(exit_block_1)
            # [None x 2 x 2 x 1024]
        
        with tf.variable_scope('block_2'):
            sep_block_3 = self._separable_block(exit_block_1, is_training, filters_list[2], k_size=(3, 3), stride=1, name='sep_block_3')
            sep_block_3 = tf.layers.batch_normalization(sep_block_3, training=is_training)        
            sep_block_3 = tf.nn.relu(sep_block_3)

            sep_block_4 = self._separable_block(sep_block_3, is_training, filters_list[3], k_size=(3, 3), stride=1, name='sep_block_4')
            sep_block_4 = tf.layers.batch_normalization(sep_block_4, training=is_training)        
            sep_block_4 = tf.nn.relu(sep_block_4)

            return sep_block_4


    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv_stem"):
                conv1 = tf.layers.conv2d(x, 32, (3, 3), 2, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, training=is_training)
                conv1 = tf.nn.relu(conv1)
                g(conv1)
                # [None x 32 x 32 x 32]
                
                conv2 = tf.layers.conv2d(conv1, 64, (3, 3), 1, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv2 = tf.layers.batch_normalization(conv2, training=is_training)
                conv2 = tf.nn.relu(conv2)
                g(conv2)
                # [None x 32 x 32 x 64]
            
            with tf.variable_scope("block_entry"):
                block_1 = self._entry_block(conv2, is_training, 128, False, name='entry_block_1')
                g(block_1)
                # [None x 16 x 16 x 128]

                block_2 = self._entry_block(block_1, is_training, 256, True, name='entry_block_2')
                g(block_2)
                # [None x 8 x 8 x 256]
                
                block_3 = self._entry_block(block_2, is_training, 728, True, name='entry_block_3')
                g(block_3)
                # [None x 4 x 4 x 728]
                
            with tf.variable_scope("block_middle"):
                block_middle = block_3
                for i in range(8):
                    block_middle = self._middle_block(block_middle, is_training, 728, name='middle_block_' + str(i+1))
                g(block_middle)
                # [None x 4 x 4 x 728]

            with tf.variable_scope("block_exit"):
                block_exit = self._exit_block(block_middle, is_training, [728, 1024, 1536, 2048])
                g(block_exit)
                # [None x 2 x 2 x 2048]

            with tf.variable_scope("tail"):
                gap = tf.reduce_mean(block_exit, axis=[1, 2], name='global_avg_pool')
                g(gap)
                # [None x 2048]

                fc_logits = tf.layers.dense(gap, self.num_classes, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='fc_layer')
                fc_op = tf.nn.softmax(fc_logits, name='softmax_op')
                g(fc_op)
                # [None x 200]

            return fc_logits, fc_op
