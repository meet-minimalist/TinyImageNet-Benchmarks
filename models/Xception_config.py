# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For Xception
minLR = 5.49e-6     # 832
maxLR = 3.90e-4     # 1157
step_factor = 2

weight_decay = 1e-5

def g(x):
    print(x.name, " ", x.get_shape().as_list())
