import os
import sys
import copy

import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import pickle as pkl

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import cv2
from tqdm import tqdm
from torchvision import datasets, transforms


import math
import torch.nn.functional as F

import typing
from backdoor.trigger import MaskedTrigger, Trigger, cifar_triggers, mnist_triggers
from backdoor.trigger import WaterMarkTrigger
from backdoor.trigger import SteganographyTrigger
from backdoor.trigger import SinusoidalTrigger
from backdoor.base import TriggerPastedTestDataset, MultiTriggerPastedTestDataset
from backdoor.dirty_label import DirtyLabelPoisonedDataset, MultiTriggerPoisonedDataset

from test import howto, semantic, chameleon



def logical_and(logi_list):
	result = logi_list[0]
	for logi in logi_list[1:]:
		result = np.logical_and(result, logi)
	return result


def count_true(res):
	return np.sum(res.astype(np.float32), axis=-1)



def test_model_acc(args, model, test_ds):
	test_loader = torch.utils.data.dataloader.DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
	_correct = 0
	_sum = 0
	model.eval()

	with torch.no_grad():
		for idx, (test_x, test_label) in enumerate(test_loader):

			predict_y = model(test_x.float().cuda()).detach()
			predict_ys = np.argmax(predict_y.cpu().numpy(), axis=-1)
			test_label = test_label.numpy()
			num_samples = test_label.shape[0]

			_correct += count_true(np.equal(predict_ys, test_label))

			_sum += num_samples

	return _correct / _sum

def test_model_asr(args, model, test_ds, target_class):
	test_loader = torch.utils.data.dataloader.DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
	_correct = 0
	_sum = 0
	model.eval()

	with torch.no_grad():
		for idx, (test_x, test_label) in enumerate(test_loader):

			predict_y = model(test_x.float().cuda()).detach()
			predict_ys = np.argmax(predict_y.cpu().numpy(), axis=-1)
			test_label = test_label.numpy()
			num_samples = test_label.shape[0]

			_correct += count_true(np.equal(predict_ys, target_class))

			_sum += num_samples

	return _correct / _sum


def get_poisoned_img(args, batch):
	if args.backdoor_method in ['howto', 'neurotoxin', 'DBA', 'ff']:
		synthesizer = howto.PatternSynthesizer(args.dataset, args.trigger_name)
		new_batch = synthesizer.make_backdoor_batch(batch)
	elif args.backdoor_method == 'chameleon':
		new_batch = copy.deepcopy(batch)
		for i in range(len(batch)):
			new_batch[i] = chameleon.add_pixel_pattern(batch[i])
	return new_batch


def test_model_acc_and_asr_per_batch(args, model, test_ds):
	'''
	used in backdoor methods poisoning per batch
	'''
	test_loader = torch.utils.data.dataloader.DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)
	target_class = args.target_class

	_correct = 0
	_sum = 0

	_success = 0
	_valid = 0.00001

	model.eval()

	with torch.no_grad():
		for idx, (test_x, test_label) in enumerate(test_loader):
			
			test_x_t = get_poisoned_img(args, test_x)
			predict_y = model(test_x.float().cuda()).detach()
			predict_y_t = model(test_x_t.float().cuda()).detach()

			predict_ys = np.argmax(predict_y.cpu().numpy(), axis=-1)
			predict_ys_t = np.argmax(predict_y_t.cpu().numpy(), axis=-1)

			test_label = test_label.numpy()
			num_samples = test_label.shape[0]

			_correct += count_true(
				np.equal(predict_ys, test_label)
			)

			_sum += num_samples

			_success += count_true(logical_and([
				np.not_equal(test_label, target_class),
				np.equal(predict_ys, test_label),
				np.equal(predict_ys_t, target_class)
			]))

			_valid += count_true(logical_and([
				np.not_equal(test_label, target_class),
				np.equal(predict_ys, test_label)
			]))

	return _correct / _sum, _success / _valid

def test_model_acc_and_asr(args, model, poisoned_test_ds):
	'''
	used in backdoor methods poisoning whole dataset
	'''
	test_loader = torch.utils.data.dataloader.DataLoader(poisoned_test_ds, batch_size=args.test_batch_size, shuffle=False)
	target_class = poisoned_test_ds.target_class
	_correct = 0
	_sum = 0

	_success = 0
	_valid = 0.00001

	model.eval()

	with torch.no_grad():
		for idx, (test_x, test_x_t, test_label) in enumerate(test_loader):

			predict_y = model(test_x.float().cuda()).detach()
			predict_y_t = model(test_x_t.float().cuda()).detach()

			predict_ys = np.argmax(predict_y.cpu().numpy(), axis=-1)
			predict_ys_t = np.argmax(predict_y_t.cpu().numpy(), axis=-1)

			test_label = test_label.numpy()
			num_samples = test_label.shape[0]

			_correct += count_true(
				np.equal(predict_ys, test_label)
			)

			_sum += num_samples

			_success += count_true(logical_and([
				np.not_equal(test_label, target_class),
				np.equal(predict_ys, test_label),
				np.equal(predict_ys_t, target_class)
			]))

			_valid += count_true(logical_and([
				np.not_equal(test_label, target_class),
				np.equal(predict_ys, test_label)
			]))

	return _correct / _sum, _success / _valid


def infer_trigger_name_from_path(input_path):
	
	ip_splits = input_path.split("_")

	if "steganography" in ip_splits:
		ind = ip_splits.index("steganography")
		return "_".join(ip_splits[ind:ind+4])

	elif "watermark" in ip_splits:
		ind = ip_splits.index('watermark')
		return "_".join(ip_splits[ind:ind+4])

	elif "sinusoidal" in ip_splits:
		ind = ip_splits.index("sinusoidal")
		return "_".join(ip_splits[ind:ind+4])

	elif "tri1" in ip_splits:
		ind = ip_splits.index("tri1")
		return "_".join(ip_splits[ind:ind+5])

	elif "tri2" in ip_splits:
		ind = ip_splits.index("tri2")
		return "_".join(ip_splits[ind:ind+5])

	else:
		raise ValueError("uknown trigger")


def infer_trigger_from_path_internal(ip_splits, train_ds, img_size):

	if "steganography" in ip_splits:
		ind = ip_splits.index("steganography")
		info = ip_splits[ind+1]
		nb_bits = int(ip_splits[ind+2])
		target_class = int(ip_splits[ind+3][1:])
		trigger = SteganographyTrigger(info, nb_bits, img_size=img_size)
		ip_splits = ip_splits[:ind] + ip_splits[ind+4:]

	elif "watermark" in ip_splits:
		ind = ip_splits.index('watermark')
		data_ind = int(ip_splits[ind+1])
		opacity = float(ip_splits[ind+2])
		target_class = int(ip_splits[ind+3][1:])
		trigger = WaterMarkTrigger(np.array(train_ds.data[data_ind]), opacity=opacity)
		ip_splits = ip_splits[:ind] + ip_splits[ind+4:]

	elif "sinusoidal" in ip_splits:
		ind = ip_splits.index("sinusoidal")
		sin_delta = int(ip_splits[ind+1])
		sin_freq = int(ip_splits[ind+2])
		target_class = int(ip_splits[ind+3][1:])
		trigger = SinusoidalTrigger(sin_delta, sin_freq)
		ip_splits = ip_splits[:ind] + ip_splits[ind+4:]

	else:
		find=False
		for tri_name in ["tri1", "tri2", "tri3", "tri4", "trisq33"]:
			if tri_name in ip_splits:
				ind = ip_splits.index(tri_name)
				size = ip_splits[ind+1]
				target_class = int(ip_splits[ind+2][1:])
				offset_to_right = int(ip_splits[ind+3])
				offset_to_bottom = int(ip_splits[ind+4])
				trigger = Trigger(tri_name+"_"+size, cifar_triggers[tri_name+"_"+size], offset_to_right=offset_to_right, offset_to_bottom=offset_to_bottom)
				ip_splits = ip_splits[:ind] + ip_splits[ind+5:]
				find=True
				break
		if not find:
			raise ValueError("uknown trigger")

	return ip_splits, trigger, target_class



def infer_trigger_from_path(input_path, train_ds, img_size):
	ip_splits = input_path.split("_")
	#print(ip_splits)

	if "trojansq" in ip_splits:
		ind = ip_splits.index("trojansq")
		target_class = int(ip_splits[ind+2][1:])

		while not os.path.exists(os.path.join(input_path, "trojan_trigger.pkl")):
			input_path = os.path.split(input_path)[0]
		
		mask_np, trigger_np, patch_np = pkl.load(open(os.path.join(input_path, "trojan_trigger.pkl"), "rb"))
		trigger = MaskedTrigger(trigger_np, mask_np)

	else:
		_, trigger, target_class = infer_trigger_from_path_internal(ip_splits, train_ds, img_size)

	return trigger, target_class



def infer_model_from_path(input_path):
	for model in ["_cnn_", "_lenet5_", "_resnet18_", "_resnet34_", "_resnet50_", "_cnn13_", "_wrn161_", "_wrn162_", "_wrn401_", "_wrn402_"]:
		if model in input_path:
			return model[1:-1]
	raise ValueError("Not found model from input path")




def pack_images(images, col=None, channel_last=False):
	if isinstance(images, (list, tuple) ):
		images = np.stack(images, 0)
	if channel_last:
		images = images.transpose(0,3,1,2) # make it channel first
	assert len(images.shape)==4
	assert isinstance(images, np.ndarray)
	
	N,C,H,W = images.shape

	# print("pack_images : ", N,C,H,W)
	if col is None:
		col = int(math.ceil(math.sqrt(N)))
	row = int(math.ceil(N / col))
	pack = np.zeros( (C, H*row, W*col), dtype=images.dtype )
	for idx, img in enumerate(images):
		h = (idx//col) * H
		w = (idx% col) * W
		pack[:, h:h+H, w:w+W] = img
	return pack



def test_generators(args, generators, nz, epoch, output_dir, tb_writer=None, stacked_output=False, norm_trans_inv=None):

	if isinstance(generators, list):
		generators = {"{}".format(ind+1) : pg for ind, pg in enumerate(generators)}
	assert isinstance(generators, dict)


	assert output_dir is not None

	for _, pg in generators.items():
		pg.eval()

	with torch.no_grad():

		z = torch.randn( (100, nz), dtype=torch.float32).cuda()
		gene_dict = { name:pg(z) for name,pg in generators.items() }

		if norm_trans_inv is not None:
			gene_dict = {name:norm_trans_inv(pert) for name,pert in gene_dict.items()}

		gene_pert_dict = {name:pert.detach().cpu().numpy() for name,pert in gene_dict.items()}

		if tb_writer is not None:
			if stacked_output:
				tb_writer.plot("Train/gene_max", epoch, {name:np.max(gene_pert) for name, gene_pert in gene_pert_dict.items()})
				tb_writer.plot("Train/gene_min", epoch, {name:np.min(gene_pert) for name, gene_pert in gene_pert_dict.items()})
				tb_writer.plot("Train/gene_mean", epoch, {name:np.mean(gene_pert) for name, gene_pert in gene_pert_dict.items()})
			else:
				for name, gene_pert in gene_pert_dict.items():
					tb_writer.plot("Train/gene_max_{}".format(name), epoch, np.max(gene_pert))
					tb_writer.plot("Train/gene_min_{}".format(name), epoch, np.min(gene_pert))
					tb_writer.plot("Train/gene_mean_{}".format(name), epoch, np.mean(gene_pert))

		for name, gene_pert in gene_pert_dict.items():
			tb_writer.image(f'gene_{name}', epoch, pack_images(np.clip(gene_pert, 0.0, 1.0)))
			# gene_pert_dict = {name:pack_images(np.clip(gene_pert, 0.0, 1.0)) for name,gene_pert in gene_pert_dict.items()}
			# gene_pert_dict = {name:(gene_pert.transpose([1, 2, 0]) * 255.0).astype(np.uint8)[:, :, ::-1] for name,gene_pert in gene_pert_dict.items()}
			# for name, gene_pert in gene_pert_dict.items():
			# 	cv2.imwrite(os.path.join(output_dir, "images", "gene_pert_{}_e{}.jpg".format(name, epoch)), gene_pert)




def get_image_prior_losses_l1(inputs_jit):
	# COMPUTE total variation regularization loss
	diff1 = torch.mean(torch.abs(inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]))
	diff2 = torch.mean(torch.abs(inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]))
	diff3 = torch.mean(torch.abs(inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]))
	diff4 = torch.mean(torch.abs(inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]))
	return diff1 + diff2 + diff3 + diff4



def get_image_prior_losses_l2(inputs_jit):
	# COMPUTE total variation regularization loss
	diff1 = torch.norm(inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:])
	diff2 = torch.norm(inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :])
	diff3 = torch.norm(inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:])
	diff4 = torch.norm(inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:])
	return diff1 + diff2 + diff3 + diff4


def get_poison_dataset(args, dataset, output_dir):

	if args.backdoor_method == 'DBA':

		trigger_list = []
		trigger_ids = range(len(args.multi_trigger_offset))
		for i in trigger_ids:
			local_trigger = Trigger(args.trigger_name, cifar_triggers[args.trigger_name] if args.img_channels == 3 else mnist_triggers[args.trigger_name],
					offset_to_right=args.multi_trigger_offset[i][0],
					offset_to_bottom=args.multi_trigger_offset[i][1])
			trigger_list.append(local_trigger)
		
		train_ds = MultiTriggerPoisonedDataset(dataset, trigger_list, target_class=args.target_class,
							num_poison_images=args.num_poison_images, 
							sample_save_path=os.path.join(output_dir, 'poisoned_samples.png')
							)
		
	elif args.backdoor_method == 'semantic':
		if args.trigger_name == 'green-car':
			backdoor_index = semantic.green_cars_indices
		elif args.trigger_name == 'racing-stripe':
			backdoor_index = semantic.racing_stripe_indices
		elif args.trigger_name == 'wall':
			backdoor_index = semantic.vertical_stripe_indices
		
		transform = transforms.Compose([transforms.ToTensor()])
		sample = transform(dataset.data[np.random.choice(backdoor_index, size=1)].squeeze())
		save_image(sample, os.path.join(output_dir, 'poisoned_samples.png'))
		
		poison_size = args.num_poison_images - len(backdoor_index)
		poison_indices = np.random.choice([i for i in range(5000) if i not in backdoor_index], size=poison_size, replace=False)
		for k in poison_indices:
			dataset.data[k] = dataset.data[np.random.choice(backdoor_index, size=1)]
			dataset.targets[k] = args.target_class
		for i in backdoor_index:
			dataset.targets[i] = args.target_class
		train_ds = dataset

	else:
		trigger = Trigger(args.trigger_name, cifar_triggers[args.trigger_name] if args.img_channels == 3 else mnist_triggers[args.trigger_name],
					offset_to_right=args.trigger_offset,
					offset_to_bottom=args.trigger_offset)
		train_ds = DirtyLabelPoisonedDataset(dataset, trigger, target_class=args.target_class,
							num_poison_images=args.num_poison_images, 
							sample_save_path=os.path.join(output_dir, 'poisoned_samples.png'))
	
	return train_ds


def get_backdoored_test_ds(args, test_ds, adv_index=-1):
	if args.backdoor_method == 'DBA':
		trigger_list = []
		if adv_index == -1:
			trigger_ids = range(len(args.multi_trigger_offset))
		else:
			trigger_ids = [adv_index]

		for i in trigger_ids:
			local_trigger = Trigger(args.trigger_name, cifar_triggers[args.trigger_name] if args.img_channels == 3 else mnist_triggers[args.trigger_name],
					offset_to_right=args.multi_trigger_offset[i][0],
					offset_to_bottom=args.multi_trigger_offset[i][1])
			trigger_list.append(local_trigger)
		backdoored_test_ds = MultiTriggerPastedTestDataset(test_ds, trigger_list, target_class=args.target_class)
	
	else:
		trigger = Trigger(args.trigger_name, cifar_triggers[args.trigger_name] if args.img_channels == 3 else mnist_triggers[args.trigger_name],
					offset_to_right=args.trigger_offset,
					offset_to_bottom=args.trigger_offset)
		backdoored_test_ds = TriggerPastedTestDataset(test_ds, trigger, target_class=args.target_class)
	return backdoored_test_ds



