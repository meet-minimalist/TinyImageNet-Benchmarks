# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For resnet34_v2
minLR = 6.603e-6     # 846
maxLR = 0.02106     # 1461
step_factor = 2

weight_decay = 0.0001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
