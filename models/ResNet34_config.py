# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 00:58:01 2020

@author: Meet
"""

# For resnet34
minLR = 2.37e-6     # 768
maxLR = 8.52e-3     # 1392
step_factor = 2

weight_decay = 0.0001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
