# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 22:11:43 2020

@author: Meet
"""

# For NASNetA (6 @ 768)
minLR = 5.86e-6     # 837
maxLR = 8.97e-3     # 1396
step_factor = 2

weight_decay = 1e-4

penultimate_filters = 768
num_repeated_blocks = 6
filters_multiplier = 2
num_reduction_cells = 2

def g(x):
    print(x.name, " ", x.get_shape().as_list())
