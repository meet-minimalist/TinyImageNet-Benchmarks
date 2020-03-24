# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 21:58:01 2020

@author: Meet
"""

# For inception_v3
minLR = 5.71e-6     # 835
maxLR = 0.01421     # 1431
step_factor = 2

weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
