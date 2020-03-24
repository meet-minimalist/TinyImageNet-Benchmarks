# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 23:17:03 2020

@author: Meet
"""

# For Inception_ResNet_v2
minLR = 1.18e-6     # 715
maxLR = 5.17e-5     # 1003
step_factor = 2

weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
