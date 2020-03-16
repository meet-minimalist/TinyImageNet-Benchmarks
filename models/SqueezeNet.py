# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.SqueezeNet_config as SqueezeNet_config
from models.SqueezeNet_config import g

class SqueezeNet:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'SqueezeNet'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=SqueezeNet_config.weight_decay)
    
    
    def _squeeze_layer(self, ip, s1x1):
        squeeze = tf.layers.conv2d(ip, s1x1, (1, 1), 1, 'valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        return squeeze


    def _extract_layer(self, ip, e1x1, e3x3):
        extract_1x1 = tf.layers.conv2d(ip, e1x1, (1, 1), 1, 'valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        extract_3x3 = tf.layers.conv2d(ip, e3x3, (3, 3), 1, 'same', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        return tf.concat([extract_1x1, extract_3x3], axis=-1)
        
    
    def _fire_block(self, ip, s1x1, e1x1, e3x3, scope='fire_block'):
        with tf.variable_scope(scope):
            squeeze = self._squeeze_layer(ip, s1x1)
            extract = self._extract_layer(squeeze, e1x1, e3x3)
            return extract
        
    
    def __call__(self, x, is_training=None):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 96, (7, 7), 2, 'valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                g(conv1)
                # [None x 29 x 29 x 96]
                
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=2)
                g(pool1)
                # [None x 14 x 14 x 96]
            
            with tf.variable_scope('fire_blocks'):
                fire_block_2 = self._fire_block(pool1, 16, 64, 64, 'fire_block_2')
                fire_block_3 = self._fire_block(fire_block_2, 16, 64, 64, 'fire_block_3')
                fire_block_4_ip = fire_block_2 + fire_block_3
                fire_block_4 = self._fire_block(fire_block_4_ip, 32, 128, 128, 'fire_block_4')
                g(fire_block_4)
                # [None x 14 x 14 x 256]
                
                pool2 = tf.layers.max_pooling2d(fire_block_4, pool_size=(3, 3), strides=2)
                g(pool2)
                # [None x 6 x 6 x 256]
                    
                fire_block_5 = self._fire_block(pool2, 32, 128, 128, 'fire_block_5')
                fire_block_6_ip = fire_block_5 + pool2
                fire_block_6 = self._fire_block(fire_block_6_ip, 48, 192, 192, 'fire_block_6')
                fire_block_7 = self._fire_block(fire_block_6, 48, 192, 192, 'fire_block_7')
                fire_block_8_ip = fire_block_7 + fire_block_6
                fire_block_8 = self._fire_block(fire_block_8_ip, 64, 256, 256, 'fire_block_8')
                g(fire_block_8)
                # [None x 6 x 6 x 512]
            
                pool3 = tf.layers.max_pooling2d(fire_block_8, pool_size=(3, 3), strides=2)
                g(pool3)
                # [None x 2 x 2 x 512]

                fire_block_9 = self._fire_block(pool3, 64, 256, 256, 'fire_block_9')
                fire_block_9_op = fire_block_9 + pool3
                fire_block_9_op = tf.layers.dropout(fire_block_9_op, rate=0.5)
                g(fire_block_9_op)
                # [None x 2 x 2 x 512]
            
            with tf.variable_scope('tail'):
                conv2 = tf.layers.conv2d(fire_block_9_op, 200, (1, 1), 1, 'valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                g(conv2)
                # [None x 2 x 2 x 200]

                avg_pool = tf.reduce_mean(conv2, axis=[1, 2])
                g(avg_pool)
                # [None x 200]
                
                logits = tf.identity(avg_pool, name='logits')
                g(logits)
                # [None x 200]
                
                outputs = tf.nn.softmax(logits, axis=-1, name='outputs')
                g(outputs)
                # [None x 200]
                
                return logits, outputs
