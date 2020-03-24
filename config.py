# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 23:03:43 2020

@author: Meet
"""

import glob

def get_model(model_name, x, is_training):
    if model_name == 'vgg16':
        from models.VGG16 import VGG16
        import models.VGG16_config as config
        model = VGG16()
    elif model_name == 'vgg16_bn':
        from models.VGG16_bn import VGG16_bn
        import models.VGG16_bn_config as config
        model = VGG16_bn()
    elif model_name == 'resnet18':
        from models.ResNet18 import ResNet18
        import models.ResNet18_config as config
        model = ResNet18()
    elif model_name == 'resnet34':
        from models.ResNet34 import ResNet34
        import models.ResNet34_config as config
        model = ResNet34()
    elif model_name == 'resnet18_wo_residual':
        from models.ResNet18_wo_residual import ResNet18_wo_res
        import models.ResNet18_wo_residual_config as config
        model = ResNet18_wo_res()
    elif model_name == 'resnet34_wo_residual':
        from models.ResNet34_wo_residual import ResNet34_wo_res
        import models.ResNet34_wo_residual_config as config
        model = ResNet34_wo_res()
    elif model_name == 'resnet18_v2':
        from models.ResNet18_v2 import ResNet18_v2
        import models.ResNet18_v2_config as config
        model = ResNet18_v2()
    elif model_name == 'resnet34_v2':
        from models.ResNet34_v2 import ResNet34_v2
        import models.ResNet34_v2_config as config
        model = ResNet34_v2()
    elif model_name == 'inception_v1':
        from models.Inception_v1 import Inception_v1
        import models.Inception_v1_config as config
        model = Inception_v1()
    elif model_name == 'inception_v2':
        from models.Inception_v2 import Inception_v2
        import models.Inception_v2_config as config
        model = Inception_v2()
    elif model_name == 'inception_v3':
        from models.Inception_v3 import Inception_v3
        import models.Inception_v3_config as config
        model = Inception_v3()
    elif model_name == 'inception_v4':
        from models.Inception_v4 import Inception_v4
        import models.Inception_v4_config as config
        model = Inception_v4()
    elif model_name == 'inception_resnet_v2':
        from models.Inception_ResNet_v2 import Inception_ResNet_v2
        import models.Inception_ResNet_v2_config as config
        model = Inception_ResNet_v2()
    elif model_name == 'mobilenet_v1':
        from models.MobileNet_v1 import MobileNet_v1
        import models.MobileNet_v1_config as config
        model = MobileNet_v1()
    elif model_name == 'mobilenet_v2':
        from models.MobileNet_v2 import MobileNet_v2
        import models.MobileNet_v2_config as config
        model = MobileNet_v2()
    elif model_name == 'mobilenet_v3_large':
        from models.MobileNet_v3 import MobileNet_v3
        import models.MobileNet_v3_config as config
        model = MobileNet_v3()
    elif model_name == 'mobilenet_v3_small':
        from models.MobileNet_v3 import MobileNet_v3
        import models.MobileNet_v3_config as config
        model = MobileNet_v3(mode='small')
    elif model_name == 'squeezenet':
        from models.SqueezeNet import SqueezeNet
        import models.SqueezeNet_config as config
        model = SqueezeNet()
    elif model_name == 'efficientnet':
        from models.EfficientNet import EfficientNet
        import models.EfficientNet_config as config
        model = EfficientNet()
    elif model_name == 'nasnet':
        from models.NASNet import NASNet
        import models.NASNet_config as config
        model = NASNet()
    elif model_name == 'mnasnet':
        from models.MNASNet import MNASNet
        import models.MNASNet_config as config
        model = MNASNet()
    elif model_name == 'xception':
        from models.Xception import Xception
        import models.Xception_config as config
        model = Xception()
    else:
        print("Error: Please specify correct model name.")
        exit(0)
    
    logits, outputs = model(x, is_training)
    return logits, outputs, config.minLR, config.maxLR, config.step_factor, config.weight_decay


start_lr = 1e-10
decay_steps = 20
decay_rate = 1.3
total_steps = 2000

epochs = 200
batch_size = 256
num_classes = 200
input_dims = [64, 64]

summary_path = "./summaries"
train_using_tfrecords = True
prepare_tfrecords = False

dataset_path = "M:/Datasets/tiny-imagenet-200/"

if train_using_tfrecords:
    import tensorflow as tf
    train_tfrecord_list = glob.glob("./data/train*.tfrecords")
    test_tfrecord_list = glob.glob("./data/test*.tfrecords")
    train_img_cnt = 0
    for tfrecord in train_tfrecord_list:        
        train_img_cnt += sum(1 for _ in tf.io.tf_record_iterator(tfrecord))
    test_img_cnt = 0
    for tfrecord in test_tfrecord_list:        
        test_img_cnt += sum(1 for _ in tf.io.tf_record_iterator(tfrecord))
    
if prepare_tfrecords:
    with open(dataset_path + "train.txt") as f:
        train_file_lines = f.readlines()
    
    with open(dataset_path + "val.txt") as f:
        test_file_lines = f.readlines()
    
    train_img_cnt = len(train_file_lines)
    test_img_cnt = len(test_file_lines)
    
    train_tfrecords_num_splits = 100
    test_tfrecords_num_splits = 10


train_folder_list = glob.glob(dataset_path + "train/*")

class_to_id_dict = dict()
id_to_class_dict = dict()
for i, f in enumerate(train_folder_list):
    f_name = f.split('\\')[-1]
    class_to_id_dict[f_name] = i
    id_to_class_dict[i] = f_name
    
def g(x):
    print(x.name, " ", x.get_shape().as_list())
