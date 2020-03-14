# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For MobileNet_v3
minLR = 2.600e-4     # 1126
maxLR = 3.929e-3     # 1333
step_factor = 2

weight_decay = 1e-5

def g(x):
    print(x.name, " ", x.get_shape().as_list())
