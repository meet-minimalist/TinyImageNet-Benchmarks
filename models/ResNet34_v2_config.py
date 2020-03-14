# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For resnet18_v2
minLR = 1.76e-5     # 921
#maxLR = 1.61e-3     # 1240
maxLR = 0.04222     # 1514
step_factor = 2

weight_decay = 0.0001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
