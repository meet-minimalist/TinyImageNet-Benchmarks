# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:20:43 2019

@author: Meet
"""

'''
Model name			width_coeff, depth_coeff, default_res, dropout_rate

EfficientNetB0		1.0, 		 1.0, 		  224, 		   0.2

EfficientNetB1		1.0, 		 1.1, 		  240,  	   0.2

EfficientNetB2		1.1, 		 1.2, 		  260,  	   0.3
		
EfficientNetB3		1.2, 		 1.4, 		  300, 		   0.3
		
EfficientNetB4		1.4, 		 1.8, 		  380, 		   0.4
		
EfficientNetB5		1.6, 		 2.2, 		  456, 		   0.4

EfficientNetB6		1.8, 		 2.6, 		  528,   	   0.5
		
EfficientNetB7		2.0, 		 3.1, 		  600, 		   0.5

EfficientNetL2 		4.3, 		 5.3, 		  800, 		   0.5
'''

# For EfficientNet-B0
#minLR = 2.31e-6       # 766
#maxLR = 1.94e-4       # 1104
#minLR = 1.76e-3       # 1272
#maxLR = 9.58e-3       # 1401


# For EfficientNet-B2
minLR = 7.23e-4        # 1204
maxLR = 0.02531        # 1475
step_factor = 2

depth_divisor = 8

'''
# For EfficientNetB0 
depth_coefficient = 1.0         # depth constant
width_coefficient = 1.0         # width constant
resolution_coefficient = 1.0    # resolution constant
dropout = 0.2
'''
# For EfficientNetB2
depth_coefficient = 1.2         # depth constant
width_coefficient = 1.1         # width constant
resolution_coefficient = 1.15   # resolution constant       # we wont use gamma as we have to upscale the images from 64 x 64 to 74 x 74
dropout = 0.3

weight_decay = 0.00001

def g(x):
    print(x.name, " ", x.get_shape().as_list())
