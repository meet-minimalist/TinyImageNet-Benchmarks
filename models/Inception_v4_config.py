# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 22:45:05 2020

@author: Meet
"""

# For inception_v4
minLR = 7.48e-7     # 680
maxLR = 2.25e-4     # 1115
step_factor = 2

weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
