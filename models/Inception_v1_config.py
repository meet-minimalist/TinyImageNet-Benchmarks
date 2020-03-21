# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For inception_v1
minLR = 4.36e-5     # 992
maxLR = 2.40e-4     # 1120
step_factor = 2

weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
