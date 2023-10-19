import os
import sys
import math

import torch
from torch import nn
import functools

import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import weight_norm

from models.cnn import CNN, CNN13
from models.lenet import LeNet5
from models.resnet_cifar import ResNet18 as resnet18_cifar
from models.resnet_tinyimagenet import resnet18 as resnet18_tiny
from models.wide_resnet import *


def get_model(args, bn_args=None, name=None):

	if args.dataset in ['mnist']:
		if args.model.lower() == "cnn":
			return CNN(num_classes=args.num_classes)
		elif args.model.lower() == "lenet5":
			return LeNet5(num_classes=args.num_classes)
		else: 
			raise ValueError("unknown model : {}".format(args.model))

	if args.dataset in ["svhn", "cifar10", "cifar100", 'cifar10few', 'cifar100few']:
		if args.model.lower() == "cnn13":
			return CNN13(num_classes=args.num_classes)
		elif args.model.lower() == "resnet18":
			return resnet18_cifar(num_classes=args.num_classes)
		
		elif args.model.lower() == "wrn161":
			return wrn_16_1(num_classes=args.num_classes)
		elif args.model.lower() == "wrn162":
			return wrn_16_2(num_classes=args.num_classes)
		elif args.model.lower() == "wrn401":
			return wrn_40_1(num_classes=args.num_classes)
		elif args.model.lower() == "wrn402":
			return wrn_40_2(num_classes=args.num_classes)
		else:
			raise ValueError("unknown model : {}".format(args.model))


	elif args.dataset in ["tiny-imagenet", "vggface2_subset", "mini-imagenet"]:
		if args.model.lower() == "resnet18":
			return resnet18_tiny(num_classes=args.num_classes)
		# elif args.model.lower() == 'resnet34':
		# 	return Resnet_64(base_encoder=resnet34, num_classes=args.num_classes, bn_args=bn_args)
		# elif args.model.lower() == 'resnet50':
		# 	return Resnet_64(base_encoder=resnet50, num_classes=args.num_classes, bn_args=bn_args)
		else:
			raise ValueError("unknown model : {}".format(args.model))