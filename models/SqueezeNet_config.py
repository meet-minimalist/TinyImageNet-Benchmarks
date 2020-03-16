# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For squeezenet
minLR = 5.04e-7     # 650
maxLR = 6.30e-5     # 1018
step_factor = 2

weight_decay = 0.0  # no regularization mentioned in paper

def g(x):
    print(x.name, " ", x.get_shape().as_list())
