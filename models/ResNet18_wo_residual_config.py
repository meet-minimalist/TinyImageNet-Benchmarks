# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:37:43 2020

@author: Meet
"""

# For resnet18_wo_residual
minLR = 1.76e-5     # 921
maxLR = 8.52e-3     # 1392
step_factor = 2

weight_decay = 0.0001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
