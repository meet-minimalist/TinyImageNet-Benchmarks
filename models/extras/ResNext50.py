# -*- coding: utf-8 -*-
"""
Created on Sat Mar 08 20:01:01 2020

@author: Meet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import models.ResNext50_config as ResNext50_config
from models.ResNext50_config import g

class ResNext50:
    def __init__(self, input_dims=(64, 64), num_classes=200):
        self.model_name = 'ResNext50'
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(scale=ResNext50_config.weight_decay)
    
    def _residual_branch(self, ip, filter_num, stride, cardinality, is_training):
        conv = tf.layers.conv2d(ip, filter_num // 2, (1, 1), stride, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, momentum=0.9, training=is_training)
        conv = tf.nn.relu(conv)
        
        '''
        conv_splits = tf.split(conv, cardinality, axis=-1)
        conv_splits_op = []
        for conv_split in conv_splits:
            conv_cardinity = tf.layers.conv2d(conv_split, filter_num // (2 * cardinality) , (3, 3), 1, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            conv_cardinity = tf.layers.batch_normalization(conv_cardinity, momentum=0.9, training=is_training)
            conv_cardinity = tf.nn.relu(conv_cardinity)
            conv_splits_op.append(conv_cardinity)
        conv_cardinity_concat = tf.concat(conv_splits_op, axis=-1)
        '''

        with tf.variable_scope("group_conv"):
            # Note: Below line of code have been taken from: https://stackoverflow.com/a/49041584 as above commented code didnt work.
            bottleneck_depth = conv.get_shape().as_list()[-1]
            group_size = bottleneck_depth // cardinality  # 128 // 32 ==> 4
            w = tf.get_variable(name='depthwise_filter', shape=[3, 3, bottleneck_depth, group_size])        # [3 x 3 x 128 x 4]
            conv = tf.nn.depthwise_conv2d_native(conv, w, strides=(1, 1, 1, 1), padding='SAME')
            depthwise_shape = conv.get_shape().as_list()  # [B x H x W x 128*4]
            conv = tf.reshape(conv, [-1, depthwise_shape[1], depthwise_shape[2], cardinality, group_size, group_size])
            # [B x H x W x 32 x 4 x 4]  ==> [B, H, W, cardinality, 4_filters_applied_to_each_input_channel, 4_outputs_corresponding_to_each_4_inputs_in_group]
            conv = tf.reduce_sum(conv, axis=4)            # [B x H x W x 32 x 4] 
            # ==> this will sum all the 4 outputs specific to each input channel. similar to what happens when a filter is applied to a 3d tensor to generate a 2d output for that specific filter.
            conv = tf.reshape(conv, [-1, depthwise_shape[1], depthwise_shape[2], bottleneck_depth])
            
            '''
            # For batch norm
            conv_bn_groups = tf.split(conv, num_or_size_splits=cardinality, axis=3)     # this will give the 32 groups, each having 4 channels. It is required to compute batch_norm on group not on whole 128 channels.
            # 32 list of tensors, each having shape [b x h x w x 1 x 4]
            conv_bn_output = []
            for conv_bn_group in conv_bn_groups:
                conv_bn_group = tf.layers.batch_normalization(conv_bn_group, momentum=0.9, training=is_training)
                conv_bn_output.append(conv_bn_group)
            conv_bn_output = tf.nn.relu(tf.concat(conv_bn_output, axis=3))
            # [B x H x W x 32 x 4]
            conv_bn_output = tf.reshape(conv_bn_output, [-1, depthwise_shape[1], depthwise_shape[2], bottleneck_depth])
            # [B x H x W x 128]
            '''
        conv = tf.layers.conv2d(conv, filter_num, (1, 1), 1, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
        conv = tf.layers.batch_normalization(conv, momentum=0.9, training=is_training)
        return conv
        
    def _identity_branch(self, ip, filter_num, stride, is_training):
        if stride == 2 or ip.get_shape().as_list()[-1] != filter_num:
            ip = tf.layers.conv2d(ip, filter_num, (1, 1), stride, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            ip = tf.layers.batch_normalization(ip, momentum=0.9, training=is_training)
        return ip
    
    def _res_block(self, ip, filter_num, stride, cardinality, is_training, name):
        with tf.variable_scope(name):
            identity_branch = self._identity_branch(ip, filter_num, stride, is_training)
            res_branch = self._residual_branch(ip, filter_num, stride, cardinality, is_training)
            return tf.nn.relu(identity_branch + res_branch)
        
    def __call__(self, x, is_training):
        # x : [None x 64 x 64 x 3]

        with tf.variable_scope(self.model_name):
            with tf.variable_scope("conv1"):
                conv1 = tf.layers.conv2d(x, 64, (7, 7), 2, 'same', activation=None, use_bias=False, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                conv1 = tf.layers.batch_normalization(conv1, momentum=0.9, training=is_training)
                conv1 = tf.nn.relu(conv1)
                g(conv1)
                # [None x 32 x 32 x 64]
                
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=2, padding='same')
                g(pool1)
                # [None x 16 x 16 x 64]
            
            with tf.variable_scope("block_1"):
                block_1 = self._res_block(pool1, 256, 1, 32, is_training, 'rep_1')
                block_1 = self._res_block(block_1, 256, 1, 32, is_training, 'rep_2')
                block_1 = self._res_block(block_1, 256, 1, 32, is_training, 'rep_3')
                g(block_1)
            
            with tf.variable_scope("block_2"):
                block_2 = self._res_block(block_1, 512, 2, 32, is_training, 'rep_1')
                block_2 = self._res_block(block_2, 512, 1, 32, is_training, 'rep_2')
                block_2 = self._res_block(block_2, 512, 1, 32, is_training, 'rep_3')
                block_2 = self._res_block(block_2, 512, 1, 32, is_training, 'rep_4')
                g(block_2)
            
            with tf.variable_scope("block_3"):
                block_3 = self._res_block(block_2, 1024, 2, 32, is_training, 'rep_1')
                block_3 = self._res_block(block_3, 1024, 1, 32, is_training, 'rep_2')
                block_3 = self._res_block(block_3, 1024, 1, 32, is_training, 'rep_3')
                block_3 = self._res_block(block_3, 1024, 1, 32, is_training, 'rep_4')
                block_3 = self._res_block(block_3, 1024, 1, 32, is_training, 'rep_5')
                block_3 = self._res_block(block_3, 1024, 1, 32, is_training, 'rep_6')
                g(block_3)
            
            with tf.variable_scope("block_4"):
                block_4 = self._res_block(block_3, 2048, 2, 32, is_training, 'rep_1')
                block_4 = self._res_block(block_4, 2048, 1, 32, is_training, 'rep_2')
                block_4 = self._res_block(block_4, 2048, 1, 32, is_training, 'rep_3')
                g(block_4)
            
            with tf.variable_scope("tail"):
                gap = tf.reduce_mean(block_4, axis=[1, 2], name='global_avg_pool')
                g(gap)
                fc_logits = tf.layers.dense(gap, self.num_classes, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, name='fc_layer')
                fc_op = tf.nn.softmax(fc_logits, name='softmax_op')
                g(fc_op)
                
            return fc_logits, fc_op
