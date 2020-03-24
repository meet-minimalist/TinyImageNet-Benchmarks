# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:05:01 2020

@author: Meet
"""

# For VGG
minLR = 1.76e-5     # 921
maxLR = 5.35e-4     # 1181

step_factor = 2

weight_decay = 5e-4

def g(x):
    print(x.name, " ", x.get_shape().as_list())
