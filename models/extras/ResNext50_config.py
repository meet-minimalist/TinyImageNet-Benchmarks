# -*- coding: utf-8 -*-
"""
Created on Sat Mar 08 20:01:01 2020

@author: Meet
"""

# For resnext50
minLR = 1.72e-6     # 744
maxLR = 4.25e-5     # 988
step_factor = 2

weight_decay = 0.0001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
