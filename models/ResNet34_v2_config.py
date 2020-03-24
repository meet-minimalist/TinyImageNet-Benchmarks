# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 20:42:01 2020

@author: Meet
"""

# For resnet34_v2
minLR = 6.603e-6     # 846
maxLR = 0.02106     # 1461
step_factor = 2

weight_decay = 0.00001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
