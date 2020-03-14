# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For VGG
minLR = 1.76e-5     # 921
maxLR = 5.35e-4     # 1181

step_factor = 2

weight_decay = (5e-4) / 2
# dividing by 2 to 
#weight_decay = 0.0

def g(x):
    print(x.name, " ", x.get_shape().as_list())
