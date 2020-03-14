# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

### Note: This implementation has been inspired and taken from : https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py

import tensorflow as tf
import models.Inception_v2_config as Inception_v2_config
from models.Inception_v2_config import g

class Inception_v2:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'Inception_v2'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=Inception_v2_config.weight_decay)
                
    def _conv_layer(self, x, is_training, k_size, no_filters, stride, padding):
        conv = tf.layers.conv2d(x, no_filters, k_size, stride, padding, activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, training=is_training)
        conv = tf.nn.relu(conv)
        return conv
    
    def _inception_block(self, x, is_training, kernel_1x1, kernel_3x3_1, kernel_3x3_2, kernel_mx):
        conv1x1 = self._conv_layer(x, is_training, (1, 1), kernel_1x1, 1, 'same')
        
        conv3x3_1 = self._conv_layer(x, is_training, (1, 1), kernel_3x3_1[0], 1, 'same')
        conv3x3_1 = self._conv_layer(conv3x3_1, is_training, (3, 3), kernel_3x3_1[1], 1, 'same')
        
        conv3x3_2 = self._conv_layer(x, is_training, (1, 1), kernel_3x3_2[0], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 3), kernel_3x3_2[1], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 3), kernel_3x3_2[2], 1, 'same')
        
        avgpool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=1, padding='same')
        avgpool =  self._conv_layer(avgpool, is_training, (1, 1), kernel_mx, 1, 'same')
        
        concat = tf.concat([conv1x1, conv3x3_1, conv3x3_2, avgpool], axis=-1)
        return concat
    
    def _inception_block_s2(self, x, is_training, kernel_3x3_1, kernel_3x3_2):
        conv3x3_1 = self._conv_layer(x, is_training, (1, 1), kernel_3x3_1[0], 1, 'same')
        conv3x3_1 = self._conv_layer(conv3x3_1, is_training, (3, 3), kernel_3x3_1[1], 2, 'same')
        
        conv3x3_2 = self._conv_layer(x, is_training, (1, 1), kernel_3x3_2[0], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 3), kernel_3x3_2[1], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 3), kernel_3x3_2[2], 2, 'same')
        
        mxpool = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=2, padding='same')
        
        concat = tf.concat([conv3x3_1, conv3x3_2, mxpool], axis=-1)
        return concat
    
    def __call__(self, x, is_training=None):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                print("---- Conv_1 ----")
                conv1 = self._conv_layer(x, is_training, (7, 7), 64, 2, 'same')
                g(conv1)
                # [None x 32 x 32 x 64]
                
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=2, padding='same')
                g(pool1)
                # [None x 16 x 16 x 64]
                
            with tf.variable_scope("conv2"):
                print("---- Conv_2 ----")
                conv2_1 = self._conv_layer(pool1, is_training, (1, 1), 64, 1, 'same')
                g(conv2_1)
                # [None x 16 x 16 x 64]

                conv2_2 = self._conv_layer(conv2_1, is_training, (3, 3), 192, 1, 'same')
                g(conv2_2)
                # [None x 16 x 16 x 192]
                
                pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(3, 3), strides=2, padding='same')
                g(pool2)
                # [None x 8 x 8 x 192]            
                
            
            with tf.variable_scope("inception_3"):
                print("---- Inception : 3 ----")
                inception_3a = self._inception_block(pool2, is_training, 64, [64, 64], [64, 96, 96], 32)
                inception_3b = self._inception_block(inception_3a, is_training, 64, [64, 96], [64, 96, 96], 64)
                g(inception_3b)
                # [None x 8 x 8 x 320]            

            with tf.variable_scope("inception_4"):
                print("---- Inception : 4 ----")
                inception_4a = self._inception_block_s2(inception_3b, is_training, [128, 160], [64, 96, 96])
                inception_4b = self._inception_block(inception_4a, is_training, 224, [64, 96], [96, 128, 128], 128)
                inception_4c = self._inception_block(inception_4b, is_training, 192, [96, 128], [96, 128, 128], 128)
                inception_4d = self._inception_block(inception_4c, is_training, 160, [128, 160], [128, 160, 160], 96)
                inception_4e = self._inception_block(inception_4d, is_training, 96, [128, 192], [160, 192, 192], 96)
                g(inception_4e)
                # [None x 4 x 4 x 576]     
                
            with tf.variable_scope("inception_5"):
                print("---- Inception : 5 ----")
                inception_5a = self._inception_block_s2(inception_4e, is_training, [128, 192], [192, 256, 256])
                inception_5b = self._inception_block(inception_5a, is_training, 352, [192, 320], [160, 224, 224], 128)
                inception_5c = self._inception_block(inception_5b, is_training, 352, [192, 320], [192, 224, 224], 128)
                g(inception_5c)
                # [None x 2 x 2 x 1024]
                
                kernel_size = inception_5c.get_shape().as_list()[1:3]
                pool5 = tf.layers.average_pooling2d(inception_5c, pool_size=kernel_size, padding='valid', strides=1)
                g(pool5)
                # [None x 1 x 1 x 1024]            

            with tf.variable_scope("tail"):
                print("---- Tail ----")
                dropout = tf.layers.dropout(pool5, rate=0.2)
                # [None x 1 x 1 x 1024]
                
                logits = tf.layers.conv2d(dropout, self.num_classes, (1, 1), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='logits')
                g(logits)
                # [None x 1 x 1 x 200]
                logits = tf.squeeze(logits, axis=[1, 2], name='squeezed_logits')
                g(logits)
                # [None x 200]
                
                op = tf.nn.softmax(logits, axis=-1, name='softmaxed_output')
                g(op)
                # [None x 200]
            return logits, op
        