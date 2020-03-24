# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 00:21:01 2020

@author: Meet
"""

# For MobileNet_v2
minLR = 6.6e-6       # 847
maxLR = 2.012e-3     # 1282
step_factor = 2

weight_decay = 0.00004

def g(x):
    print(x.name, " ", x.get_shape().as_list())
