# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 00:02:01 2020

@author: Meet
"""

# For MobileNet_v1
minLR = 1.288e-3     # 1248
maxLR = 0.048        # 1524
step_factor = 2

weight_decay = 0.0001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
