# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

minLR = 7.23e-6     # 853
maxLR = 0.01167     # 1416
step_factor = 2

weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
