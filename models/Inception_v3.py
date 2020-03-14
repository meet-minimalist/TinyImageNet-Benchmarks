# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""


### Note: This implementation has been inspired and taken from : https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py


import tensorflow as tf
import models.Inception_v3_config as Inception_v3_config
from models.Inception_v3_config import g

class Inception_v3:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'Inception_v3'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=Inception_v3_config.weight_decay)
    
    def _conv_layer(self, x, is_training, k_size, no_filters, stride, padding):
        conv = tf.layers.conv2d(x, no_filters, k_size, stride, padding, activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, training=is_training)
        conv = tf.nn.relu(conv)
        return conv
    
    def _inception_block_traditional(self, x, is_training, kernel_1x1, kernel_3x3, kernel_5x5, kernel_pool):
        conv1x1 = self._conv_layer(x, is_training, (1, 1), kernel_1x1, 1, 'same')
        
        conv3x3 = self._conv_layer(x, is_training, (1, 1), kernel_3x3[0], 1, 'same')
        conv3x3 = self._conv_layer(conv3x3, is_training, (3, 3), kernel_3x3[1], 1, 'same')
        conv3x3 = self._conv_layer(conv3x3, is_training, (3, 3), kernel_3x3[2], 1, 'same')

        conv5x5 = self._conv_layer(x, is_training, (1, 1), kernel_5x5[0], 1, 'same')
        conv5x5 = self._conv_layer(conv5x5, is_training, (5, 5), kernel_5x5[1], 1, 'same')
        
        conv_avgpool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=1, padding='same')
        conv_avgpool = self._conv_layer(conv_avgpool, is_training, (1, 1), kernel_pool, 1, 'same')
            
        concat = tf.concat([conv1x1, conv3x3, conv5x5, conv_avgpool], axis=-1)
        return concat
    
    def _inception_block_factorized(self, x, is_training, kernel_1x1, kernel_7x7_1, kernel_7x7_2, kernel_pool):
        conv1x1 = self._conv_layer(x, is_training, (1, 1), kernel_1x1, 1, 'same')
        
        conv7x7_1 = self._conv_layer(x, is_training, (1, 1), kernel_7x7_1[0], 1, 'same')
        conv7x7_1 = self._conv_layer(conv7x7_1, is_training, (1, 7), kernel_7x7_1[1], 1, 'same')
        conv7x7_1 = self._conv_layer(conv7x7_1, is_training, (7, 1), kernel_7x7_1[2], 1, 'same')

        conv7x7_2 = self._conv_layer(x, is_training, (1, 1), kernel_7x7_2[0], 1, 'same')
        conv7x7_2 = self._conv_layer(conv7x7_2, is_training, (7, 1), kernel_7x7_2[1], 1, 'same')
        conv7x7_2 = self._conv_layer(conv7x7_2, is_training, (1, 7), kernel_7x7_2[2], 1, 'same')
        conv7x7_2 = self._conv_layer(conv7x7_2, is_training, (7, 1), kernel_7x7_2[3], 1, 'same')
        conv7x7_2 = self._conv_layer(conv7x7_2, is_training, (1, 7), kernel_7x7_2[4], 1, 'same')
        
        conv_avgpool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=1, padding='same')
        conv_avgpool = self._conv_layer(conv_avgpool, is_training, (1, 1), kernel_pool, 1, 'same')
            
        concat = tf.concat([conv1x1, conv7x7_1, conv7x7_2, conv_avgpool], axis=-1)
        return concat
    
    def _inception_block_factorized_v2(self, x, is_training, kernel_1x1, kernel_3x3_1, kernel_3x3_2, kernel_pool):
        conv1x1 = self._conv_layer(x, is_training, (1, 1), kernel_1x1, 1, 'same')
        
        conv3x3_1 = self._conv_layer(x, is_training, (1, 1), kernel_3x3_1[0], 1, 'same')
        conv3x3_1 = self._conv_layer(conv3x3_1, is_training, (1, 3), kernel_3x3_1[1], 1, 'same')
        conv3x3_1 = self._conv_layer(conv3x3_1, is_training, (3, 1), kernel_3x3_1[2], 1, 'same')

        conv3x3_2 = self._conv_layer(x, is_training, (1, 1), kernel_3x3_2[0], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 3), kernel_3x3_2[1], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (1, 3), kernel_3x3_2[2], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 1), kernel_3x3_2[3], 1, 'same')
        
        conv_avgpool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=1, padding='same')
        conv_avgpool = self._conv_layer(conv_avgpool, is_training, (1, 1), kernel_pool, 1, 'same')
            
        concat = tf.concat([conv1x1, conv3x3_1, conv3x3_2, conv_avgpool], axis=-1)
        return concat
    
    def _inception_block_s2(self, x, is_training, kernel_3x3_only, kernel_3x3):
        conv3x3_1 = self._conv_layer(x, is_training, (3, 3), kernel_3x3_only, 2, 'valid')
        
        conv3x3_2 = self._conv_layer(x, is_training, (1, 1), kernel_3x3[0], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 3), kernel_3x3[1], 1, 'same')
        conv3x3_2 = self._conv_layer(conv3x3_2, is_training, (3, 3), kernel_3x3[2], 2, 'valid')

        conv_maxpool = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=2, padding='valid')
            
        concat = tf.concat([conv3x3_1, conv3x3_2, conv_maxpool], axis=-1)
        return concat
    
    def _inception_block_s2_v2(self, x, is_training, kernel_3x3, kernel_7x7):
        conv3x3 = self._conv_layer(x, is_training, (1, 1), kernel_3x3[0], 1, 'same')
        conv3x3 = self._conv_layer(conv3x3, is_training, (3, 3), kernel_3x3[1], 2, 'valid')

        conv7x7 = self._conv_layer(x, is_training, (1, 1), kernel_7x7[0], 1, 'same')
        conv7x7 = self._conv_layer(conv7x7, is_training, (1, 7), kernel_7x7[1], 1, 'same')
        conv7x7 = self._conv_layer(conv7x7, is_training, (7, 1), kernel_7x7[2], 1, 'same')
        conv7x7 = self._conv_layer(conv7x7, is_training, (3, 3), kernel_7x7[3], 2, 'valid')

        maxpool = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=2, padding='valid')
            
        concat = tf.concat([conv3x3, conv7x7, maxpool], axis=-1)
        return concat


    def __call__(self, x, is_training=None):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv_stem"):
                print("---- Conv Stem ----")
                conv2d_1a = self._conv_layer(x, is_training, (3, 3), 32, 2, 'valid')
                g(conv2d_1a)
                # [None x 32 x 32 x 64]
                conv2d_2a = self._conv_layer(conv2d_1a, is_training, (3, 3), 32, 1, 'valid')
                g(conv2d_2a)
                
                conv2d_2b = self._conv_layer(conv2d_2a, is_training, (3, 3), 64, 1, 'same')
                g(conv2d_2b)
                
                pool_2 = tf.layers.max_pooling2d(conv2d_2b, pool_size=(3, 3), strides=2, padding='valid')
                g(pool_2)
                # [None x 16 x 16 x 64]
                
                conv2d_3a = self._conv_layer(pool_2, is_training, (1, 1), 80, 1, 'valid')
                conv2d_3b = self._conv_layer(conv2d_3a, is_training, (3, 3), 192, 1, 'valid')
                g(conv2d_3b)
                
                pool_3 = tf.layers.max_pooling2d(conv2d_3b, pool_size=(3, 3), strides=2, padding='valid')
                g(pool_3)
                # [None x 16 x 16 x 64]
            
            with tf.variable_scope("inception_4"):
                print("---- Inception : 4 ----")
                inception_4a = self._inception_block_traditional(pool_3, is_training, 64, [64, 96, 96], [48, 64], 32)
                inception_4b = self._inception_block_traditional(inception_4a, is_training, 64, [64, 96, 96], [48, 64], 64)
                inception_4c = self._inception_block_traditional(inception_4b, is_training, 64, [64, 96, 96], [48, 64], 64)
                g(inception_4c)
                
            with tf.variable_scope("inception_5"):
                print("---- Inception : 5 ----")
                inception_5a = self._inception_block_s2(inception_4c, is_training, 384, [64, 96, 96])
                inception_5b = self._inception_block_factorized(inception_5a, is_training, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192)
                inception_5c = self._inception_block_factorized(inception_5b, is_training, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192)
                inception_5d = self._inception_block_factorized(inception_5c, is_training, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192)
                inception_5e = self._inception_block_factorized(inception_5d, is_training, 192, [192, 192, 192], [192, 192, 192, 192, 192], 192)
                g(inception_5e)
                
            with tf.variable_scope("inception_6"):
                print("---- Inception : 6 ----")
                # Can't use stride=2, as we already have 2x2 image here, for 64x64 input image
                #inception_6a = self._inception_block_s2_v2(inception_5e, is_training, [192, 320], [192, 192, 192, 192])
                inception_6a = self._inception_block_factorized_v2(inception_5e, is_training, 320, [384, 384, 384], [448, 384, 384, 384], 192)
                inception_6b = self._inception_block_factorized_v2(inception_6a, is_training, 320, [384, 384, 384], [448, 384, 384, 384], 192)
                inception_6c = self._inception_block_factorized_v2(inception_6b, is_training, 320, [384, 384, 384], [448, 384, 384, 384], 192)
                g(inception_6c)

                kernel_size = inception_6c.get_shape().as_list()[1:3]
                pool5 = tf.layers.average_pooling2d(inception_6c, pool_size=kernel_size, padding='valid', strides=1)
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
