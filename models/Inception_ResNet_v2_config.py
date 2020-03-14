# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For Inception_ResNet_v2
minLR = 3.65e-6     # 801
maxLR = 6.59e-4     # 1197
step_factor = 2

weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
