# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:28:24 2020

@author: Home
"""
import glob
from tqdm import tqdm
import numpy as np
np.random.seed(42)

#dataset_path = "C:/TinyImageNet/"
dataset_path = "M:/Datasets/tiny-imagenet-200/"


train_folder_list = glob.glob(dataset_path + "train/*")

class_to_id_dict = dict()
id_to_class_dict = dict()
for i, f in enumerate(train_folder_list):
    f_name = f.split('\\')[-1]
    class_to_id_dict[f_name] = i
    id_to_class_dict[i] = f_name

train_file_list = glob.glob(dataset_path + "train/*/images/*.JPEG")
np.random.shuffle(train_file_list)

train_writer = open('train.txt', 'w')
for f in tqdm(train_file_list):
    string = f.replace('/','\\') + " " + str(class_to_id_dict[f.split('\\')[1]])
    train_writer.write(string + "\n")
train_writer.close()


valid_file_list = glob.glob(dataset_path + "val/images/*.JPEG")

with open(dataset_path + "val/val_annotations.txt", 'r') as f:
    val_lines = f.readlines()

val_writer = open('val.txt', 'w')
for l in tqdm(val_lines):
    file_path = dataset_path + "val/images/" + l.split('\t')[0]
    string = file_path + " " + str(class_to_id_dict[l.split('\t')[1]])
    val_writer.write(string + "\n")
val_writer.close()

