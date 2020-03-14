# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:25:37 2019

@author: Meet
"""


from TinyImageNetClassifier import Classifier
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = Classifier('xception')

model.dry_run_clr()

#model.plot_clr_graph()

#model.train()

#restore_path = "./summaries/vgg/2020_01_13_20_05_27_879414_training_summary/best_checkpoint/vgg_eps76-test_loss_2.57-test_top_1_acc_42.46.ckpt"
#model.train(resume=True, resume_from_eps=80, resume_from_gstep=31280, restore_ckpt=restore_path)

#ckpt_path = "./summaries/vgg_bn/2020_01_16_08_55_03_523669_training_summary/best_checkpoint/vgg_bn_eps64-test_loss_2.18-test_top_1_acc_52.90.ckpt"
#model.eval_on_dataset(ckpt_path, train_dataset=False)
#model.eval_on_dataset_v2(ckpt_path, dataset_path=config.dataset_path, training_set=False)
        