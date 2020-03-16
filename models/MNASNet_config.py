# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For MNASNet
minLR = 1.101e-5       # 885
maxLR = 0.0304         # 1489
step_factor = 2

weight_decay = 0.00001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
