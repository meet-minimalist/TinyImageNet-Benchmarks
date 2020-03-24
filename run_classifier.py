# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 23:03:43 2020

@author: Meet
"""

import os
import argparse
import config
from TinyImageNetClassifier import Classifier
import tensorflow as tf 

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='True', help='Use gpu for processing. Options: True and False. Default: True')
parser.add_argument("--model", type=str, help='Use gpu for processing. \n Options \
							1. vgg16, 2. vgg16_bn, \
							3. resnet18_wo_residual, 4. resnet34_wo_residual, \
							5. resnet18, 6. resnet34, \
							7. resnet18_v2, 8. resnet34_v2, \
							9. inception_v1, 10. inception_v2, 11. inception_v3, 12. inception_v4, 13. inception_resnet_v2, 14. xception, \
							15. mobilnet_v1, 16. mobilenet_v2, 17. mobilenet_v3_small, 18. mobilenet_v3_large, \
							19. squeezenet, \
							20. nasnet, 21. mnasnet, \
							22. efficientnet ')
parser.add_argument("--mode", type=str, default='training', help='Mode of training: dry_run, training and eval. Default: training')
parser.add_argument("--ckpt_path", type=str, help="Checkpoint folder for evaluation purposes. (Latest checkpoint will be selected automatically.")
parser.add_argument("--eval_dataset", type=str, help="Evaluation to be done on which dataset : train or test, Default: train")

args = parser.parse_args()


if __name__ == "__main__":
	if args.gpu.lower() == 'false':
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	model = Classifier(args.model)
	if args.mode.lower() == 'dry_run':
		model.dry_run_clr()
		model.plot_clr_graph()

	elif args.mode.lower() == 'training':
		model.train()
	
		#restore_path = "./summaries/efficientnet/2020_03_20_20_34_38_697290_training_summary/best_checkpoint/efficientnet_eps90-test_loss_3.02-test_top_1_acc_35.37.ckpt"
		#model.train(resume=True, resume_from_eps=90, resume_from_gstep=35190, restore_ckpt=restore_path)


	elif args.mode.lower() == 'eval':
		ckpt_path = tf.train.latest_checkpoint(args.ckpt_path)
		print("Restoring from ckpt: ", ckpt_path)
		
		if args.eval_dataset.lower() == 'train':
			model.eval_on_dataset(ckpt_path, train_dataset=True)
		else:
			model.eval_on_dataset(ckpt_path, train_dataset=False)
			