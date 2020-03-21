# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For MobileNet_v3_small / MobileNet_v3_large
#minLR = 2.600e-4     # 1126
#maxLR = 8.520e-3     # 1392

minLR = 1.145e-5     # 888
maxLR = 8.520e-3     # 1392
step_factor = 2

#weight_decay = 1e-5
weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
