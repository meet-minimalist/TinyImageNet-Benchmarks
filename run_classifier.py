# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:25:37 2019

@author: Meet
"""

import os
import argparse
from TinyImageNetClassifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='True', help='Use gpu for processing. Options: True and False. Default: True')
parser.add_argument("--model", type=str, help='Use gpu for processing. \n Options \
							1. vgg16, 2. vgg16_bn, \
							3. resnet18, 4. resnet34, \
							5. resnet18_v2, 6. resnet34_v2, \
							7. inception_v1, 8. inception_v2, 9. inception_v3, 10. inception_v4, 11. inception_resnet_v2, 12. xception, \
							13. mobilnet_v1, 14. mobilenet_v2, 15. mobilenet_v3_small, 16. mobilenet_v3_large, \
							17. squeezenet, \
							18. nasnet, 19. mnasnet, \
							20. efficientnet ')
parser.add_argument("--mode", type=str, default='training', help='Mode of training: dry_run or training. Default: training')


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
	
	#restore_path = "./summaries/vgg/2020_01_13_20_05_27_879414_training_summary/best_checkpoint/vgg_eps76-test_loss_2.57-test_top_1_acc_42.46.ckpt"
	#model.train(resume=True, resume_from_eps=80, resume_from_gstep=31280, restore_ckpt=restore_path)

	#ckpt_path = "./summaries/vgg_bn/2020_01_16_08_55_03_523669_training_summary/best_checkpoint/vgg_bn_eps64-test_loss_2.18-test_top_1_acc_52.90.ckpt"
	#model.eval_on_dataset(ckpt_path, train_dataset=False)
	#model.eval_on_dataset_v2(ckpt_path, dataset_path=config.dataset_path, training_set=False)
			