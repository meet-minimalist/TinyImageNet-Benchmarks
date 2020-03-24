# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:57:01 2020

@author: Meet
"""

# For VGG
minLR = 1.88e-5     # 926
maxLR = 2.88e-4     # 1134

step_factor = 2

# weight_decay = 1e-5   # with regularization model was not getting trained, so removed the regularization.
weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
