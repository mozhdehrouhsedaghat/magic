"""Train the model"""
from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms
import math
import numpy as np
import os
import torchvision.models as torch_models
import pdb
import model.net as models
import glob
from magic import MAGIC_Class
from torchsummary import summary

random.seed(0)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', default=20000, type=int, help='epochs')
	parser.add_argument('--setting_id', default=0, type=int,
						help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
	parser.add_argument('--bs', default=5, type=int, help='batch size')
	parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')
	parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
	parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
	parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
	parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
	parser.add_argument('--main_loss_multiplier', type=float, default=1.0,
						help='coefficient for the main loss in optimization')
	parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')
	parser.add_argument('--gpu', default='0', help='index of gpus to use')
	parser.add_argument('--pre_w', default='', help='pretrained weights')
	parser.add_argument('--save_prefix', default='', help='file saving prefix')
	parser.add_argument('--mode', default='', help='image generation mode. use syn for synthesis, location for location control')
	parser.add_argument('--target_prefix', default='1')
	parser.add_argument('--file_name', default=None)

	args = parser.parse_args()
	print(args)

	args.gpu = args.gpu.split(',')
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
	os.environ['QT_QPA_PLATFORM'] = 'offscreen'

	torch.backends.cudnn.benchmark = True
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print("loading torchvision model for inversion with the name: {}".format(args.arch_name))
	pretrain = False
	if(args.pre_w == ''):
		pretrain = True
	net = torch_models.__dict__[args.arch_name](pretrained=pretrain)
	#model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
	#model.to(device)
	net = torch.nn.DataParallel(net, device_ids=range(len(args.gpu))).to(device)
	if(args.pre_w != ''):
		print('loading weights of the adversarially robust model')
		wDict=torch.load("./" + args.pre_w, device)["model"]
		for key in list(wDict.keys()):
			if ".model" in key:
				new_key = key.replace(".model","")
				wDict[new_key] = wDict.pop(key)
		net.load_state_dict(wDict, strict = False)

	net.eval()
	summary(net,(3,24,24))
	# introduce discriminator
	scale = 12
	nfc = 32
	min_nfc = min(32 * pow(2, math.floor(scale / 4)), 128)
	netD = models.myWDiscriminator2(nfc, min_nfc).to(device)# use myWDiscriminator if you want PatchGAN with smaller patches
	netD.apply(models.weights_init)
	netD = torch.nn.DataParallel(netD, device_ids=range(len(args.gpu))).to(device)
	netD.train()
	print('netD',netD)
	summary(netD, (3, 224, 224))
    
    
    
	vae =  models.VAE()
	vae.apply(models.weights_init)
	vae = torch.nn.DataParallel(vae, device_ids=range(len(args.gpu))).to(device)
	vae.train()
	print('vae',vae)
	summary(vae, (3, 224, 224))

	args.iterations = 2000
	args.start_noise = True
	args.resolution = 224
	bs = args.bs

	parameters = dict()
	parameters["resolution"] = 224
	parameters["start_noise"] = True
	parameters["store_best_images"] = args.store_best_images
	parameters["pre_w"] = args.pre_w
	parameters["save_prefix"] = args.save_prefix
	parameters["mode"] = args.mode
	parameters["target_prefix"] = args.target_prefix

	coefficients = dict()
	if(args.mode == 'synthesis'):
		coefficients["tv_l1"] = 0.001
		coefficients["tv_l2"] = 0.001
	else:
		coefficients["tv_l1"] = args.tv_l1
		coefficients["tv_l2"] = args.tv_l2
	coefficients["l2"] = args.l2
	coefficients["lr"] = args.lr
	coefficients["main_loss_multiplier"] = args.main_loss_multiplier

	network_output_function = lambda x: x

	criterion = nn.CrossEntropyLoss()
	vae_criterion = nn.BCEWithLogitsLoss()

	DeepInversionEngine = MAGIC_Class(net_teacher=net,
											net_D=netD,
											parameters=parameters,
											setting_id=args.setting_id,
											bs=bs,
											criterion=criterion,
											coefficients=coefficients,
											network_output_function=network_output_function,
											gpu_number=len(args.gpu),
											vae = vae,
											vae_criterion = vae_criterion)

	folder_path = './input_images'
	files = []
	if(args.file_name ==  None):
		for i_file in glob.glob(folder_path + '/*.jpg'):
			files.append(i_file)
		for i_file in glob.glob(folder_path + '/*.png'):
			files.append(i_file)
		for i_file in glob.glob(folder_path + '/*.JPEG'):
			files.append(i_file)
	else:
		files.append(folder_path + '/' + args.file_name +'.jpg')    


	for i in files:
		DeepInversionEngine.get_images(i_sample=i)

if __name__ == '__main__':
	main()
