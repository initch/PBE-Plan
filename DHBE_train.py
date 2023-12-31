import os
import sys
import random
import argparse
import pickle as pkl
import numpy as np
import cv2

import logging

import torch
import torch.nn.functional as F
import torch.optim as optim

import gan
from models.utils import get_model
from dataloader import get_dataset, get_norm_trans, get_norm_trans_inv
import my_utils as utils

from backdoor.base import SimpleSubset, TriggerPastedTestDataset, MultiTriggerPastedTestDataset

import vis
from vis import TbWriter, create_logger

import copy

from test.semantic import GreenCarTest, RacingStripeTest, WallTest
from test.edge_case import EdgeCaseTest


logger = create_logger()

def train(args, teacher,
	student, generator, pert_generator,
	norm_trans, norm_trans_inv, 
	optimizer, epoch, tb_writer=None, 
	):
	teacher.eval()
	student.train()
	generator.train()
	pert_generator.train()

	criterion = torch.nn.CrossEntropyLoss().cuda()

	optimizer_S, optimizer_G, optimizer_Gp = optimizer

	loss_oh_pre = 0.0
	loss_ie_pre = 0.0
	for i in range( args.epoch_iters):

		# ------- Step 1 - update sample generator
		z = torch.randn((args.batch_size, args.nz)).cuda()
		z2 = torch.randn((args.batch_size, args.nz)).cuda()
		optimizer_G.zero_grad()
		generator.train()

		fake_data = generator(z)
		fake_data_2 = generator(z2)
		t_logit = teacher(fake_data)
		t_logit_2 = teacher(fake_data_2)
		# t2_logit = teacher2(fake_data)
		# s_logit = student(fake_data)
		s_logit = student(fake_data)


		loss_G1 = - F.l1_loss(s_logit, t_logit)

		# one-hot loss
		pred = t_logit.data.max(1)[1]
		loss_one_hot = F.cross_entropy(t_logit, pred)

		# information entropy loss
		mean_softmax_T = torch.nn.functional.softmax(t_logit, dim=1).mean(dim=0)
		loss_information_entropy = (mean_softmax_T * torch.log(mean_softmax_T)).sum()

		# diversity loss
		softmax_o_T1 = torch.nn.functional.softmax(t_logit, dim=1)
		softmax_o_T2 = torch.nn.functional.softmax(t_logit_2, dim=1)
		lz = torch.norm(fake_data - fake_data_2) / torch.norm(softmax_o_T2 - softmax_o_T1)
		loss_diversity_seeking = 1 / (lz + 1 * 1e-20)

		# real loss
		loss_real = torch.exp(loss_one_hot - loss_oh_pre) + torch.exp(loss_information_entropy - loss_ie_pre) + loss_diversity_seeking

		# loss_tvl1 = utils.get_image_prior_losses_l1(fake_data)
		loss_tvl2 = utils.get_image_prior_losses_l2(fake_data)

		loss_G = loss_G1 + args.loss_weight_tvl2 * loss_tvl2 #+ args.loss_weight_real * loss_real
		loss_G.backward()
		optimizer_G.step()

		if i < 1:
			loss_oh_pre = loss_one_hot.detach()
			loss_ie_pre = loss_information_entropy.detach()

		# ------- Step 2 - update trigger generator
		z = torch.randn((args.batch_size, args.nz)).cuda()
		z2 = torch.randn((args.batch_size, args.nz2)).cuda()
		z3 = torch.randn((args.batch_size, args.nz2)).cuda()

		optimizer_Gp.zero_grad()
		pert_generator.train()

		fake_data = generator(z).detach()
		pert = pert_generator(z2)
		pert = pert_generator.random_pad(pert)
		fake_img = norm_trans_inv(fake_data)
		patched_img = fake_img + pert
		patched_data = norm_trans(patched_img)

		pert3 = pert_generator(z3)
		pert3 = pert_generator.random_pad(pert3)
		patched_img3 = fake_img + pert3
		patched_data3 = norm_trans(patched_img3)

		t_logit = teacher(fake_data)
		
		s_logit_pert3 = student(patched_data3)

		am1, am2, am3, am4, s_logit = student(fake_data, latent=True)
		am5, am6, am7, am8, s_logit_pert = student(patched_data, latent=True)

		# Adversarial AMA loss
		loss_am1 = - F.mse_loss(am1.detach(), am5)
		loss_am2 = - F.mse_loss(am2.detach(), am6)
		loss_am3 = - F.mse_loss(am3.detach(), am7)
		loss_am4 = - F.mse_loss(am4.detach(), am8)
		loss_adv_ama =  loss_am1 * 500 + loss_am2 * 500 + loss_am3 * 500 + loss_am4 * 1000

		loss_mislead = - F.l1_loss(s_logit.detach(), s_logit_pert)
		loss_consistency = F.l1_loss(s_logit_pert3, s_logit_pert)
		# loss_teacher = - F.smooth_l1_loss(t_logit.detach(), t_logit_pert) + 0.1 * F.cross_entropy(t_logit_pert3, t_logit_pert.data.max(1)[1])

		loss_Gp = loss_mislead  + args.loss_weight_consistency * loss_consistency + args.loss_weight_adv_ama * loss_adv_ama
		

		loss_Gp.backward()
		optimizer_Gp.step()

		# ------- Step 3 - update student
		for k in range(args.inner_iters):
			z = torch.randn((args.batch_size, args.nz)).cuda()
			z2 = torch.randn((args.batch_size, args.nz2)).cuda()

			optimizer_S.zero_grad()

			fake_data = generator(z).detach()
			pert = pert_generator(z2)
			pert = pert_generator.random_pad(pert).detach()

			fake_img = norm_trans_inv(fake_data)
			patched_img = fake_img + pert
			patched_data = norm_trans(patched_img)

			t_logit = teacher(fake_data)
			# s_logit = student(fake_data)
			# s_logit_pert = student(patched_data)
			
			am1, am2, am3, am4, s_logit = student(fake_data, latent=True)
			am5, am6, am7, am8, s_logit_pert = student(patched_data, latent=True)
			loss_am1 = F.mse_loss(am1.detach(), am5)
			loss_am2 = F.mse_loss(am2.detach(), am6)
			loss_am3 = F.mse_loss(am3.detach(), am7)
			loss_am4 = F.mse_loss(am4.detach(), am8)
			loss_ama =  loss_am1 * 500 + loss_am2 * 500 + loss_am3 * 500 + loss_am4 * 1000
			

			loss_S1 = F.l1_loss(s_logit, t_logit.detach()) #KD蒸馏
			# loss_S1 = F.smooth_l1_loss(s_logit, t_logit.detach())
			loss_S2 = F.l1_loss(s_logit_pert, s_logit.detach()) #+ F.smooth_l1_loss(fake_data, patched_data)
			
			#+ 0.1 * F.l1_loss(s_logit_pert, t2_logit.detach())

			loss_S = loss_S1 + loss_S2 * args.loss_weight_d1 + args.loss_weight_ama * loss_ama
			
			loss_S.backward()
			optimizer_S.step()

		if i % args.log_interval == 0:
			logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.4f} S_loss: {:.4f}'.format(
				epoch, i, args.epoch_iters, 100*float(i)/float(args.epoch_iters), loss_G.item(), loss_S.item()))

			if tb_writer is not None:
				tb_writer.plot('Train/Loss_S', (epoch-1)*args.epoch_iters+i, loss_S.item())
				tb_writer.plot('Train/Loss_S1', (epoch-1)*args.epoch_iters+i, loss_S1.item())
				tb_writer.plot('Train/Loss_S2', (epoch-1)*args.epoch_iters+i, loss_S2.item())
				tb_writer.plot('Train/Loss_S_AMA', (epoch-1)*args.epoch_iters+i, loss_ama.item())
				tb_writer.plot('Train/Loss_G', (epoch-1)*args.epoch_iters+i, loss_G.item())
				tb_writer.plot('Train/Loss_G_l1', (epoch-1)*args.epoch_iters+i, loss_G1.item())
				tb_writer.plot('Train/Loss_G_tvl2', (epoch-1)*args.epoch_iters+i, loss_tvl2.item())
				tb_writer.plot('Train/Loss_G_real', (epoch-1)*args.epoch_iters+i, loss_real.item())
				tb_writer.plot('Train/Loss_Gp', (epoch-1)*args.epoch_iters+i, loss_Gp.item())
				tb_writer.plot('Train/Loss_Gp_mislead', (epoch-1)*args.epoch_iters+i, loss_mislead.item())
				tb_writer.plot('Train/Loss_Gp_consistency', (epoch-1)*args.epoch_iters+i, loss_consistency.item())
				tb_writer.plot('Train/Loss_Gp_AMA', (epoch-1)*args.epoch_iters+i, loss_adv_ama.item())
			

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='DHBE CIFAR')
	parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 256)')
	parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
	parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 500)')
	parser.add_argument('--epoch_iters', type=int, default=50)
	parser.add_argument('--inner_iters', type=int, default=5)

	parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
	parser.add_argument('--lr_G', type=float, default=0.001, help='learning rate (default: 0.1)')
	parser.add_argument('--lr_decay', type=float, default=0.1)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

	parser.add_argument('--input_dir', type=str, default="howto_cifar10_resnet18_tri1_3x3_t0_scale_3")
	parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'vggface2_subset', 'mini-imagenet', 'tiny-imagenet'], help='dataset name')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
	parser.add_argument('--nz', type=int, default=256)

	parser.add_argument('--loss_weight_tvl2', type=float, default=0.0001)
	parser.add_argument('--lr_Gp', type=float, default=1e-3, help='learning rate (default: 0.1)')
	parser.add_argument('--loss_weight_d1', type=float, default=0.1)
	parser.add_argument('--loss_weight_ama', type=float, default=0)
	parser.add_argument('--loss_weight_real', type=float, default=0)
	parser.add_argument('--loss_weight_consistency', type=float, default=0)
	parser.add_argument('--loss_weight_adv_ama', type=float, default=0)
	parser.add_argument('-ps', '--patch_size', type=int, default=5)
	parser.add_argument('--nz2', type=int, default=256)
	parser.add_argument('--layerwise_ratio', type=float, nargs='+')

	parser.add_argument('--backdoor_method', type=str, default='howto')
	# for Badnets
	parser.add_argument('--trigger_offset', type=int, default=0)
	parser.add_argument('--trigger_name', type=str, default='tri1_3x3')
	# parser.add_argument('--multi_trigger_offset', type=list, default=[[0,0], [0,3], [5,0], [5,3]])
	parser.add_argument('--target_class', type=int, default=0)

	parser.add_argument('--vis_generator', action='store_true', default=True)
	parser.add_argument('--adjlr',type=int, default=0)
	parser.add_argument('--note',type=str, default='')
	
	args = parser.parse_args()
	args.num_classes = {"cifar10":10, "cifar100":100, "mnist":10, "vggface2_subset":100, "svhn":10, "mini-imagenet":100, "tiny-imagenet":200}.get(args.dataset, 10)
	args.img_size = {"cifar10":32, "cifar100":32, "mnist":28, "vggface2_subset":64, "svhn":32, "mini-imagenet":64, "tiny-imagenet":64}.get(args.dataset, 32)
	args.img_channels = {"cifar10":3, "cifar100":3, "mnist":1, "vggface2_subset":3, "svhn":3, "mini-imagenet":3, "tiny-imagenet":3}.get(args.dataset, 3)
	
	
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.enabled = False

	
	print(args)

	if args.note == '':
		param_string = "e_{}_inner_{}_ps_{}_wama_{}_wd1_{}_wtvl2_{}_wreal_{}_wadv_{}_wcon_{}_lrs_{}_lrg_{}_lrgp_{}"
		param_string = param_string.format(args.epochs, args.inner_iters, args.patch_size, args.loss_weight_ama, args.loss_weight_d1, args.loss_weight_tvl2, args.loss_weight_real, args.loss_weight_adv_ama, args.loss_weight_consistency, args.lr_S, args.lr_G, args.lr_Gp)
	else:
		param_string = "{}_e_{}_ps_{}_wama_{}_wd1_{}_wtvl2_{}_wreal_{}_wadv_{}_wcon_{}_lrs_{}_lrg_{}_lrgp_{}"
		param_string = param_string.format(args.note, args.epochs, args.patch_size, args.loss_weight_ama, args.loss_weight_d1, args.loss_weight_tvl2, args.loss_weight_real, args.loss_weight_adv_ama, args.loss_weight_consistency, args.lr_S, args.lr_G, args.lr_Gp)

	output_dir = os.path.join('logs/'+args.input_dir, __file__.split('.')[0] + "_{}_results".format(param_string))

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	os.makedirs(os.path.join(output_dir, "student"), exist_ok=True)
	os.makedirs(os.path.join(output_dir, "generator"), exist_ok=True)
	# os.makedirs(os.path.join(output_dir, "report"), exist_ok=True)
	# if args.vis_generator:
	# 	os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
	#if args.vis_sensitivity:
	#	os.makedirs(os.path.join(output_dir, "sensitivity"), exist_ok=True)
	#if args.save_checkpoint:
	#	os.makedirs(os.path.join(output_dir, "train_stats"), exist_ok=True)
	

	norm_trans = get_norm_trans(args)
	norm_trans_inv = get_norm_trans_inv(args)
	train_ds, test_ds = get_dataset(args)
	args.model = utils.infer_model_from_path(args.input_dir)

	if args.backdoor_method == 'Badnets':
		trigger, target_class = utils.infer_trigger_from_path(args.input_dir, train_ds, args.img_size)
		poisoned_test_ds = TriggerPastedTestDataset(test_ds, trigger, target_class=target_class)
	# elif args.backdoor_method == 'DBA':
	# 	poisoned_test_ds = utils.get_backdoored_test_ds(args, test_ds)
	elif args.backdoor_method == 'edge-case':
		tester = EdgeCaseTest()
		poisoned_test_ds = tester.get_poisoned_dataset()
	elif args.backdoor_method == 'semantic':
		if args.trigger_name == 'green-car':
			tester = GreenCarTest(train_ds, args.target_class)
		elif args.trigger_name == 'racing-stripe':
			tester = RacingStripeTest(train_ds, args.target_class)
		elif args.trigger_name == 'wall':
			tester = WallTest(train_ds, args.target_class)
		poisoned_test_ds = tester.get_backdoored_test_dataset()

	# ckpt_path = os.path.join(args.input_dir, "teacher", "{}-{}.pt".format(args.dataset, args.model))
	teacher = get_model(args)

	# torch.save(teacher.state_dict(), 'model.pt')
	# exit()
	student = get_model(args)
	generator = gan.GeneratorB(nz=args.nz, nc=args.img_channels, img_size=args.img_size)

	pert_generator = gan.PatchGeneratorPreBN(nz=args.nz2, nc=args.img_channels, patch_size=args.patch_size, out_size=args.img_size)
	
	ckpt_path = f'logs/{args.input_dir}/teacher/{args.dataset}-{args.model}.pt'
	
	# teacher.load_state_dict(torch.load('/home/bei_chen/DHBE-main/train_teacher_badnets_cifar10_resnet18_e_200_tri1_3x3_t9_0_0_n300_results/teacher/cifar10-resnet18.pt'))
	teacher.load_state_dict(torch.load(ckpt_path))
	logger.info("Teacher restored from %s"%(ckpt_path))

	# student.load_state_dict(torch.load(ckpt_path))
	# print("Student restored from %s"%(ckpt_path))

	# old_state = torch.load(ckpt_path, map_location='cpu')
	# new_state = student.cpu().state_dict()
	# # BCU: random initialization
	# for key in old_state.keys():
	# 	if key.find('bn') != -1 or key.find('shortcut.1') != -1:
	# 		continue
	# 	if key.endswith('.weight') or key.endswith('.bias'):
	# 		p = args.layerwise_ratio[0]
	# 		if key.startswith('layer1'):
	# 			p = args.layerwise_ratio[1]
	# 		elif key.startswith('layer2'):
	# 			p = args.layerwise_ratio[2]
	# 		elif key.startswith('layer3'):
	# 			p = args.layerwise_ratio[3]
	# 		elif key.startswith('layer4'):
	# 			p = args.layerwise_ratio[4]
	# 		elif key.startswith('linear'):
	# 			p = args.layerwise_ratio[5]

	# 		# if key.startswith('fc'):
	# 		#     p = 1
	# 		# elif key.find('shortcut') != -1:
	# 		#     p = 1
	# 		#     # p = 1 - (num_layers - 3) * 0.01
	# 		#     print(key, p)
	# 		# else:
	# 		#     p = num_layers * 0.01
	# 		#     print(key, p)
	# 		#     num_layers += 1
	# 		mask_one = torch.ones(old_state[key].shape) * (1 - p)
	# 		mask = torch.bernoulli(mask_one)
	# 		# masked_weight = old_state[key] * mask * (1/(1-p)) + new_state[key] * (1 - mask)
	# 		masked_weight = old_state[key] * mask + new_state[key] * (1 - mask)     # 1 copy, 0 random
	# 		old_state[key] = masked_weight
	
	# student.load_state_dict(old_state,strict=False)

	teacher = teacher.cuda()
	student = student.cuda()

	generator = generator.cuda()
	pert_generator = pert_generator.cuda()

	teacher.eval()

	optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
	optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G )
	optimizer_Gp = optim.Adam( pert_generator.parameters(), lr=args.lr_Gp )

	lr_decay_steps = [0.4, 0.8]
	lr_decay_steps = [int(e * args.epochs) for e in lr_decay_steps]
	# lr_decay_steps = [100, 200]
	
	# lr_decay_steps_list = [
	# 	[400, 1200, 1600], # adjusting learning rate 0: 1111 epoch 2000
	# 	[200, 600, 1000],  # adj 1:  1111 epoch 1200
	# 	[200, 600, 800],  # adj 2: epoch 1000
	# 	[100, 300, 500],  # adj 3:xxxx
	# 	[200, 500, 700],  # adj 4: xxxx epoch 800
	# 	[1200, 1600],     # adj 5:  epoch 2000
	# 	[1800, 2400]
	# ]
	# lr_decay_steps = lr_decay_steps_list[args.adjlr]

	scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, lr_decay_steps, args.lr_decay)
	scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, lr_decay_steps, args.lr_decay)
	scheduler_Gp = optim.lr_scheduler.MultiStepLR(optimizer_Gp, lr_decay_steps, args.lr_decay)

	tb_writer = TbWriter(f'runs/{args.input_dir}_DHBE_{param_string}')
	tb_writer.save_params_to_table(vars(args))

	for epoch in range(1, args.epochs + 1):
		# Train
		if epoch == 1:
			if args.backdoor_method == 'Badnets':
				acc, asr = utils.test_model_acc_and_asr(args, student, poisoned_test_ds)
			elif args.backdoor_method in ['edge-case', 'semantic']:
				acc = utils.test_model_acc(args, student, test_ds)
				asr = utils.test_model_asr(args, student, poisoned_test_ds, args.target_class)
			else:
				acc, asr = utils.test_model_acc_and_asr_per_batch(args, student, test_ds)
			logger.warning("Epoch 0 | ACC : {:.4f}, ASR : {:.4f}".format(acc, asr))
			tb_writer.plot("Test/ACC", 0, acc)
			tb_writer.plot("Test/ASR", 0, asr)

		train(args, teacher=teacher,student=student, generator=generator, 
					pert_generator=pert_generator,
					norm_trans=norm_trans, norm_trans_inv=norm_trans_inv,
					optimizer=[optimizer_S, optimizer_G, 
					optimizer_Gp
					], epoch=epoch, 
					tb_writer=tb_writer, 
					# trigger=trigger
					)

		scheduler_S.step()
		scheduler_G.step()
		scheduler_Gp.step()

		# Test
		if args.vis_generator:
			utils.test_generators(args, {'img':generator}, args.nz, epoch, output_dir, tb_writer, norm_trans_inv=norm_trans_inv)
			utils.test_generators(args, {'pert':pert_generator}, args.nz2, epoch, output_dir, tb_writer, norm_trans_inv=lambda x:(x+1.0)/2.0)
		
		if epoch <= 50 or epoch % 10 == 0:
			# student = model.get_base_model_from_am(student_am)
			if args.backdoor_method == 'Badnets':
				acc, asr = utils.test_model_acc_and_asr(args, student, poisoned_test_ds)
			elif args.backdoor_method in ['edge-case', 'semantic']:
				acc = utils.test_model_acc(args, student, test_ds)
				asr = utils.test_model_asr(args, student, poisoned_test_ds, args.target_class)
			else:
				acc, asr = utils.test_model_acc_and_asr_per_batch(args, student, test_ds)
			logger.warning("Epoch {} | ACC : {:.4f}, ASR : {:.4f}".format(epoch, acc, asr))
			tb_writer.plot("Test/ACC", epoch, acc)
			tb_writer.plot("Test/ASR", epoch, asr)


		if epoch % 50 == 0 or epoch == args.epochs or epoch == 100:
			torch.save(student.state_dict(), os.path.join(output_dir, "student/%s-%s_epoch_%d.pt"%(args.dataset, args.model, epoch)))
			torch.save(generator.state_dict(), os.path.join(output_dir, "generator/%s-%s-generator.pt"%(args.dataset, args.model)))
			torch.save(pert_generator.state_dict(), os.path.join(output_dir, "generator/%s-%s-pert_generator.pt"%(args.dataset, args.model)))



if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	main()


