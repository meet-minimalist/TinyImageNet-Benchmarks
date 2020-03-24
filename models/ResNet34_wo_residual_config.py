# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:37:43 2020

@author: Meet
"""

# For resnet34_wo_residual
minLR = 2.37e-6     # 768
maxLR = 1.65e-3     # 1267
step_factor = 2

weight_decay = 0.0001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
