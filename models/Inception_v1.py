# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.Inception_v1_config as Inception_v1_config
from models.Inception_v1_config import g

class Inception_v1:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'Inception_v1'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=Inception_v1_config.weight_decay)
            
    def local_response_normalization(self, x, local_k_size=5, alpha=0.0001, beta=0.75, k = 1.0):
        # taken from : https://github.com/antspy/inception_v1.pytorch/blob/737c61270f76321dabed8e9d2b7c5fd89bfa17ea/inception_v1.py#L40
        # and : https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
        
        div = tf.square(x)
        div = tf.layers.average_pooling2d(div, pool_size=(local_k_size, local_k_size), strides=1, padding='same')
        div = tf.math.pow(k + alpha * div, beta)
        x = x / div
        return x
    
    def _inception_block(self, x, kernel_1x1, kernel_3x3, kernel_5x5, kernel_mx):
        conv1 = tf.layers.conv2d(x, kernel_1x1, (1, 1), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv1 = tf.nn.relu(conv1)
        
        conv3_1 = tf.layers.conv2d(x, kernel_3x3[0], (1, 1), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_3 = tf.layers.conv2d(conv3_1, kernel_3x3[1], (3, 3), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv3_3 = tf.nn.relu(conv3_3)
        
        conv5_1 = tf.layers.conv2d(x, kernel_5x5[0], (1, 1), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv5_1 = tf.nn.relu(conv5_1)
        conv5_5 = tf.layers.conv2d(conv5_1, kernel_5x5[1], (5, 5), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv5_5 = tf.nn.relu(conv5_5)
        
        mxpool = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=1, padding='same')
        mxpool_conv =  tf.layers.conv2d(mxpool, kernel_mx, (1, 1), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        mxpool_conv = tf.nn.relu(mxpool_conv)
        
        concat = tf.concat([conv1, conv3_3, conv5_5, mxpool_conv], axis=-1)
        return concat
    
    def __call__(self, x, is_training=None):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 64, (7, 7), 2, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.nn.relu(conv1)
                g(conv1)
                # [None x 32 x 32 x 64]
                
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=2, padding='same')
                g(pool1)
                # [None x 16 x 16 x 64]
                lrn1 = self.local_response_normalization(pool1)
                g(lrn1)
                # [None x 16 x 16 x 64]
            
            with tf.variable_scope("conv2"):
                conv2 = tf.layers.conv2d(lrn1, 64, (1, 1), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv2 = tf.nn.relu(conv2)
                g(conv2)
                # [None x 16 x 16 x 192]

                conv3 = tf.layers.conv2d(conv2, 192, (3, 3), 1, 'same', activation=None, use_bias=True, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv3 = tf.nn.relu(conv3)
                g(conv3)
                # [None x 16 x 16 x 192]
                
                lrn2 = self.local_response_normalization(conv3)
                g(lrn2)
                # [None x 16 x 16 x 192]

                pool2 = tf.layers.max_pooling2d(lrn2, pool_size=(3, 3), strides=2, padding='same')
                g(pool2)
                # [None x 8 x 8 x 192]            
                
            
            with tf.variable_scope("inception_3"):
                inception_3a = self._inception_block(pool2, 64, [96, 128], [16, 32], 32)
                inception_3b = self._inception_block(inception_3a, 128, [128, 192], [32, 96], 64)
                g(inception_3b)
                
                pool3 = tf.layers.max_pooling2d(inception_3b, pool_size=(3, 3), strides=2, padding='same')
                g(pool3)
                # [None x 4 x 4 x 480]            


            with tf.variable_scope("inception_4"):
                inception_4a = self._inception_block(pool3, 192, [64, 208], [16, 48], 64)
                inception_4b = self._inception_block(inception_4a, 160, [112, 224], [24, 64], 64)
                inception_4c = self._inception_block(inception_4b, 128, [128, 256], [24, 64], 64)
                inception_4d = self._inception_block(inception_4c, 112, [144, 288], [32, 64], 64)
                inception_4e = self._inception_block(inception_4d, 256, [160, 320], [32, 128], 128)
                g(inception_4e)
                
                pool4 = tf.layers.max_pooling2d(inception_4e, pool_size=(3, 3), strides=2, padding='same')
                g(pool4)
                # [None x 2 x 2 x 832]     
                
            
            with tf.variable_scope("inception_5"):
                inception_5a = self._inception_block(pool4, 256, [160, 320], [32, 128], 128)
                inception_5b = self._inception_block(inception_5a, 384, [192, 384], [48, 128], 128)
                g(inception_5b)

                pool5 = tf.layers.average_pooling2d(inception_5b, pool_size=(2, 2), strides=1)
                g(pool5)
                # [None x 1 x 1 x 1024]            

            with tf.variable_scope("tail"):
                dropout = tf.layers.dropout(tf.squeeze(pool5, axis=[1, 2]), rate=0.4)
                # [None x 1024]
                
                fc_logits = tf.layers.dense(dropout, self.num_classes, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='fc_layer')
                fc_op = tf.nn.softmax(fc_logits, name='softmax_op')
                g(fc_op)
                
            return fc_logits, fc_op
