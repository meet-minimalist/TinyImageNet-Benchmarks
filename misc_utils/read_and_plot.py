# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:19:17 2020

@author: Meet
"""


import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt
os.makedirs('./plots/', exist_ok=True)

workbook = xlrd.open_workbook('../paper/Benchmark Results.xls')

worksheet = workbook.sheet_by_index(0)

model_result = dict()


for i in range(6, 31):
    model_name = worksheet.cell(i, 2).value

    if model_name in ['ResNet50', 'ResNext50', 'ShuffleNet']:
        continue

    train_t1 = worksheet.cell(i, 5).value
    train_t5 = worksheet.cell(i, 6).value
    test_t1 = worksheet.cell(i, 7).value
    test_t5 = worksheet.cell(i, 8).value
    
    model_result[model_name] = [train_t1, train_t5, test_t1, test_t5]
    
    
######## VGG #########
x = np.arange(2) / 2
cat = 'Train Top-1 Acc.', 'Test Top-1 Acc.'
train_acc_list = []
test_acc_list = []
model_names = ['VGG16', 'VGG16_bn']

for model_name, model_acc in model_result.items():
    if model_name in model_names:
        train_acc_list.append(model_acc[0])
        test_acc_list.append(model_acc[2])


plt.figure(figsize=(10,8)) 
plt.title('Accuracy of VGG16 models', fontsize=20)
b1 = plt.bar(x - 0.1, train_acc_list, width=0.2, color='#76BB86', align='center')
b2 = plt.bar(x + 0.1, test_acc_list, width=0.2, color='#428953', align='center')
plt.legend([b1, b2], cat, fontsize=10)

for i, val in enumerate(x):
    plt.text(val - 0.13, train_acc_list[i] * 1.01, str(round(train_acc_list[i], 2)), fontsize=12)
    plt.text(val + 0.07, test_acc_list[i] * 1.01, str(round(test_acc_list[i], 2)), fontsize=12)

plt.xticks(x, model_names, rotation=60)
plt.tight_layout()
plt.plot()
plt.savefig('./plots/Vgg16.png')
##########################

######## ResNets with and without residuals #########
x = np.arange(4) / 2
cat = 'Train Top-1 Acc.', 'Test Top-1 Acc.'
train_acc_list = []
test_acc_list = []
model_names = ['ResNet18_wo_residual', 'ResNet34_wo_residual', 'ResNet18', 'ResNet34']

for model_name, model_acc in model_result.items():
    if model_name in model_names:
        train_acc_list.append(model_acc[0])
        test_acc_list.append(model_acc[2])

plt.figure(figsize=(10,8)) 
plt.title('Accuracy of ResNet models', fontsize=20)

b1 = plt.bar(x - 0.1, train_acc_list, width=0.2, color='#00BFC4', align='center')
b2 = plt.bar(x + 0.1, test_acc_list, width=0.2, color='#00EEF0', align='center')
plt.legend([b1, b2], cat, fontsize=10)

for i, val in enumerate(x):
    plt.text(val - 0.16, train_acc_list[i] * 1.01, str(round(train_acc_list[i], 2)), fontsize=12)
    plt.text(val + 0.04, test_acc_list[i] * 1.01, str(round(test_acc_list[i], 2)), fontsize=12)

plt.xticks(x, model_names, rotation=60)
plt.tight_layout()
plt.plot()
plt.savefig('./plots/ResNets_w_wo_residual.png')
###############################################

######## ResNets #########
x = np.arange(4) / 2
cat = 'Train Top-1 Acc.', 'Test Top-1 Acc.'
train_acc_list = []
test_acc_list = []
model_names = ['ResNet18', 'ResNet34', 'ResNet18_v2', 'ResNet34_v2']

for model_name, model_acc in model_result.items():
    if model_name in model_names:
        train_acc_list.append(model_acc[0])
        test_acc_list.append(model_acc[2])

plt.figure(figsize=(10,8)) 
plt.title('Accuracy of ResNet models', fontsize=20)

b1 = plt.bar(x - 0.1, train_acc_list, width=0.2, color='#FF7676', align='center')
b2 = plt.bar(x + 0.1, test_acc_list, width=0.2, color='#CE2929', align='center')
plt.legend([b1, b2], cat, fontsize=10)

for i, val in enumerate(x):
    plt.text(val - 0.16, train_acc_list[i] * 1.01, str(round(train_acc_list[i], 2)), fontsize=12)
    plt.text(val + 0.04, test_acc_list[i] * 1.01, str(round(test_acc_list[i], 2)), fontsize=12)

plt.xticks(x, model_names, rotation=60)
plt.tight_layout()
plt.plot()
plt.savefig('./plots/ResNets.png')
##########################


######## Inceptions #########
x = np.arange(6) / 2
cat = 'Train Top-1 Acc.', 'Test Top-1 Acc.'
train_acc_list = []
test_acc_list = []
model_names = ['Inception_v1', 'Inception_v2', 'Inception_v3', 'Inception_v4', 'Inception_ResNet_v2', 'Xception']

for model_name, model_acc in model_result.items():
    if model_name in model_names:
        train_acc_list.append(model_acc[0])
        test_acc_list.append(model_acc[2])

plt.figure(figsize=(10,8)) 
plt.title('Accuracy of Inception models', fontsize=20)

b1 = plt.bar(x - 0.1, train_acc_list, width=0.2, color='#708FF0', align='center')
b2 = plt.bar(x + 0.1, test_acc_list, width=0.2, color='#5675D6', align='center')
plt.legend([b1, b2], cat, fontsize=10)

for i, val in enumerate(x):
    plt.text(val - 0.20, train_acc_list[i] * 1.01, str(round(train_acc_list[i], 2)), fontsize=12)
    plt.text(val + 0.01, test_acc_list[i] * 1.01, str(round(test_acc_list[i], 2)), fontsize=12)

plt.xticks(x, model_names, rotation=60)
plt.tight_layout()
plt.plot()
plt.savefig('./plots/Inceptions.png')
##########################



######## Lightweight #########
x = np.arange(8) / 2
cat = 'Train Top-1 Acc.', 'Test Top-1 Acc.'
train_acc_list = []
test_acc_list = []
model_names = ['MobileNet_v1', 'MobileNet_v2', 'MobileNet_v3_small', 'MobileNet_v3_large', 'SqueezeNet', 'NASNet', 'MNASNet', 'EfficientNet']

for model_name, model_acc in model_result.items():
    if model_name in model_names:
        train_acc_list.append(model_acc[0])
        test_acc_list.append(model_acc[2])

plt.figure(figsize=(10,8)) 
plt.title('Accuracy of Lightweight models', fontsize=20)

b1 = plt.bar(x - 0.1, train_acc_list, width=0.2, color='#FEBE62', align='center')
b2 = plt.bar(x + 0.1, test_acc_list, width=0.2, color='#FF8B07', align='center')
plt.legend([b1, b2], cat, fontsize=10)

for i, val in enumerate(x):
    plt.text(val - 0.2, train_acc_list[i] * 1.01, str(round(train_acc_list[i], 2)), fontsize=10)
    plt.text(val + 0.01, test_acc_list[i] * 1.01, str(round(test_acc_list[i], 2)), fontsize=10)

plt.xticks(x, model_names, rotation=60)
plt.tight_layout()
plt.plot()
plt.savefig('./plots/Light_weight.png')
##########################



######## All models #########
x = np.arange(22) / 2
cat = 'Train Top-1 Acc.', 'Test Top-1 Acc.'
train_acc_list = []
test_acc_list = []
model_names = list(model_result.keys())

for model_name, model_acc in model_result.items():
    if model_name in model_names:
        train_acc_list.append(model_acc[0])
        test_acc_list.append(model_acc[2])
        
plt.figure(figsize=(20,8)) 
plt.title('Accuracy of All models', fontsize=20)

b1 = plt.bar(x - 0.1, train_acc_list, width=0.2, color='#BEF771', align='center')
b2 = plt.bar(x + 0.1, test_acc_list, width=0.2, color='#A3DD57', align='center')
#b1 = plt.bar(x - 0.1, train_acc_list, width=0.2, color='#AAFFCC', align='center')
#b2 = plt.bar(x + 0.1, test_acc_list, width=0.2, color='#77E599', align='center')
plt.legend([b1, b2], cat, fontsize=10)

for i, val in enumerate(x):
    plt.text(val - 0.2, train_acc_list[i] * 1.01, str(round(train_acc_list[i], 2)), fontsize=10)
    plt.text(val + 0.01, test_acc_list[i] * 1.01, str(round(test_acc_list[i], 2)), fontsize=10)

plt.xticks(x, model_names, rotation=60)
plt.tight_layout()
plt.plot()
plt.savefig('./plots/all_models.png')
##########################
