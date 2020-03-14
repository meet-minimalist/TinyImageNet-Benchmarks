# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

# For NASNetA (6 @ 4032)
minLR = 3.65e-6     # 801
maxLR = 6.59e-4     # 1197
step_factor = 2

weight_decay = 0.0

penultimate_filters = 4032
num_repeated_blocks = 6
filters_multiplier = 2
num_reduction_cells = 2

def g(x):
    print(x.name, " ", x.get_shape().as_list())
