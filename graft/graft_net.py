import os
import sys
import random
import argparse
import pickle as pkl
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torch.optim as optim

import gan
import model
from dataloader import get_dataset, get_norm_trans, get_norm_trans_inv
import my_utils as utils

from dataset_few import CIFAR10Few, CIFAR100Few
from torch.utils.data import DataLoader

from backdoor.base import SimpleSubset, TriggerPastedTestDataset, MultiTriggerPastedTestDataset

import vis

import copy

from test.semantic import GreenCarTest, RacingStripeTest, WallTest
from test.edge_case import EdgeCaseTest



def train(args, teacher,  
	student, pert_generator,
	train_loader,
	norm_trans, norm_trans_inv, 
	optimizer, epoch, plotter=None, 
	):
	teacher.eval()
	student.train()
	pert_generator.train()

	optimizer_S, optimizer_Gp = optimizer

	
	for data in train_loader:
		# ------- Step 1 - update trigger generator
		data = data.cuda()
		size = data.shape[0]
		z = torch.randn((size, args.nz)).cuda()

		optimizer_Gp.zero_grad()
		pert_generator.train()

		pert = pert_generator(z)
		pert = pert_generator.random_pad(pert)
		img = norm_trans_inv(data)
		patched_img = img + pert
		patched_data = norm_trans(patched_img)

		t_logit = teacher(data)
		s_logit = student(data)
		s_logit_pert = student(patched_data)

		loss_Gp = - F.smooth_l1_loss(s_logit.detach(), s_logit_pert)

		loss_Gp.backward()
		optimizer_Gp.step()

		# ------- Step 2 - update student
		for k in range(5):
			z = torch.randn((size, args.nz)).cuda()

			optimizer_S.zero_grad()

			pert = pert_generator(z)
			pert = pert_generator.random_pad(pert).detach()

			img = norm_trans_inv(data)
			patched_img = img + pert
			patched_data = norm_trans(patched_img)

			t_logit = teacher(data)
			s_logit = student(data)
			s_logit_pert = student(patched_data)

			loss_S1 = F.smooth_l1_loss(s_logit, t_logit.detach())
			loss_S2 = F.smooth_l1_loss(s_logit_pert, s_logit.detach())
			
			loss_S = loss_S1 + loss_S2 * args.loss_weight_d1
			
			loss_S.backward()
			optimizer_S.step()
		
	print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KD_loss: {:.6f} BR_loss: {:.6f}'.format(
		epoch, epoch, 4*args.epochs, 100*float(epoch)/float(4*args.epochs), loss_S1.item(), loss_S2.item()))

	if plotter is not None:
		plotter.scalar('Loss_S', (epoch-1)*args.epochs+epoch, loss_S.item())
		plotter.scalar('Loss_S1', (epoch-1)*args.epochs+epoch, loss_S1.item())
		plotter.scalar('Loss_S2', (epoch-1)*args.epochs+epoch, loss_S2.item())
		plotter.scalar('Loss_Gp', (epoch-1)*args.epochs+epoch, loss_Gp.item())


def graft_block(args, student, teacher_dict, block_id):
	student.load_state_dict(teacher_dict)
	if block_id == 1:
		optimizer = optim.SGD(student.layer1.parameters(), lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)
	elif block_id == 2:
		optimizer = optim.SGD(student.layer2.parameters(), lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)
	elif block_id == 3:
		optimizer = optim.SGD(student.layer3.parameters(), lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)
	elif block_id == 4:
		optimizer = optim.SGD(student.layer4.parameters(), lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)

	lr_decay_steps = [0.6]
	lr_decay_steps = [int(e * args.epochs) for e in lr_decay_steps]
	scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_steps, args.lr_decay)

	return optimizer, scheduler_S


def save_block(student, block_list, block_id):
	for params in student.state_dict():
		if f'layer{block_id}' in params:
			block_list.update({params: student.state_dict()[params]})
	return block_list

def graft_net(args, student, teacher_dict, saved_blocks, block_id):
	if block_id == 1:
		optimizer = optim.SGD(student.layer1.parameters(), lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)
	elif block_id == 2:
		# teacher_dict.update({'layer1': saved_blocks['layer1'], 'layer2': saved_blocks['layer2']})
		# student.load_state_dict(teacher_dict)
		# params = [param for name, param in student.named_parameters() if 'layer1' in name or 'layer2' in name]
		optimizer = optim.SGD([{'params':student.layer1.parameters()}, {'params': student.layer2.parameters()}], lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)
	elif block_id == 3:
		# teacher_dict.update({'layer1': saved_blocks['layer1'], 'layer2': saved_blocks['layer2'], 'layer3': saved_blocks['layer3']})
		# student.load_state_dict(teacher_dict)
		optimizer = optim.SGD([{'params':student.layer1.parameters()}, {'params': student.layer2.parameters()}, {'params': student.layer3.parameters()}], lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)
	elif block_id == 4:
		# teacher_dict.update(saved_blocks)
		# student.load_state_dict(teacher_dict)
		optimizer = optim.SGD(student.parameters(), lr=args.lr_block, weight_decay=args.weight_decay, momentum=0.9)

	lr_decay_steps = []
	lr_decay_steps = [int(e * args.epochs) for e in lr_decay_steps]
	scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_steps, args.lr_decay)

	return optimizer, scheduler_S


def update_saved_block(student, block_list, block_id):
	for idx in range(1, block_id+1):
		name = f'layer{idx}'
		block_list[name] = student.state_dict()[name]
	return block_list

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='DHBE CIFAR')
	parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 256)')
	parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
	parser.add_argument('--num_per_class', type=int, default=10)
	parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 500)')
	parser.add_argument('--epoch_iters', type=int, default=5)

	parser.add_argument('--lr_block', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.1)')
	parser.add_argument('--lr_net', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.1)')
	parser.add_argument('--lr_G', type=float, default=0.001, help='learning rate (default: 0.1)')
	parser.add_argument('--lr_decay', type=float, default=0.1)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

	parser.add_argument('--input_dir', type=str, default="train_howto_cifar10_resnet18_tri1_3x3_t0_scale_3")
	parser.add_argument('--dataset', type=str, default='cifar10few', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'vggface2_subset', 'mini-imagenet'], help='dataset name')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
	parser.add_argument('--nz', type=int, default=256)

	parser.add_argument('--loss_weight_tvl2', type=float, default=0.0001)
	parser.add_argument('--lr_Gp', type=float, default=1e-3, help='learning rate (default: 0.1)')
	parser.add_argument('--loss_weight_d1', type=float, default=0.1)
	parser.add_argument('--loss_weight_ama', type=float, default=0)
	parser.add_argument('--loss_weight_real', type=float, default=0)
	parser.add_argument('--loss_weight_feature', type=float, default=0)
	parser.add_argument('--loss_weight_consistency', type=float, default=0)
	parser.add_argument('--loss_weight_adv_ama', type=float, default=0)
	parser.add_argument('-ps', '--patch_size', type=int, default=5)
	parser.add_argument('--nz2', type=int, default=256)

	parser.add_argument('--backdoor_method', type=str, default='howto')
	# for Badnets
	parser.add_argument('--trigger_offset', type=int, default=0)
	parser.add_argument('--trigger_name', type=str, default='tri1_3x3')
	parser.add_argument('--target_class', type=int, default=0)

	parser.add_argument('--vis_generator', action='store_true', default=True)
	parser.add_argument('--adjlr',type=int, default=0)
	
	args = parser.parse_args()
	args.num_classes = {"cifar10":10, "cifar100":100, "mnist":10, "vggface2_subset":100, "svhn":10, "mini-imagenet":100}.get(args.dataset, 10)
	args.img_size = {"cifar10":32, "cifar100":32, "mnist":28, "vggface2_subset":64, "svhn":32, "mini-imagenet":64}.get(args.dataset, 32)
	args.img_channels = {"cifar10":3, "cifar100":3, "mnist":1, "vggface2_subset":3, "svhn":3, "mini-imagenet":3}.get(args.dataset, 3)
	
	
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.enabled = False
	
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	assert 'few' in args.dataset
	print(args)

	param_string = "wo_adjlr_e_{}_npc_{}_ps_{}_wd1_{}_lrb_{}_lrn_{}_lrgp_{}"
	param_string = param_string.format(args.epochs, args.num_per_class, args.patch_size, args.loss_weight_d1, args.lr_block, args.lr_net, args.lr_Gp)

	output_dir = os.path.join('logs/'+args.input_dir, __file__.split('.')[0] + "_{}_results".format(param_string))

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	os.makedirs(os.path.join(output_dir, "student"), exist_ok=True)
	os.makedirs(os.path.join(output_dir, "generator"), exist_ok=True)
	os.makedirs(os.path.join(output_dir, "report"), exist_ok=True)
	if args.vis_generator:
		os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
	#if args.vis_sensitivity:
	#	os.makedirs(os.path.join(output_dir, "sensitivity"), exist_ok=True)
	#if args.save_checkpoint:
	#	os.makedirs(os.path.join(output_dir, "train_stats"), exist_ok=True)

	norm_trans = get_norm_trans(args)
	norm_trans_inv = get_norm_trans_inv(args)
	train_ds, test_ds = get_dataset(args)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
	args.model = utils.infer_model_from_path(args.input_dir)

	if args.backdoor_method == 'Badnets':
		trigger, target_class = utils.infer_trigger_from_path(args.input_dir, train_ds, args.img_size)
		poisoned_test_ds = TriggerPastedTestDataset(test_ds, trigger, target_class=target_class)
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

	#ckpt_path = os.path.join(args.input_dir, "teacher", "{}-{}.pt".format(args.dataset, args.model))
	teacher = model.get_model(args)
	student = model.get_model(args)
	pert_generator = gan.PatchGeneratorPreBN(nz=args.nz2, nc=args.img_channels, patch_size=args.patch_size, out_size=args.img_size)
	
	ckpt_path = f'logs/{args.input_dir}/teacher/cifar10-resnet18.pt'
	
	# teacher.load_state_dict(torch.load('/home/bei_chen/DHBE-main/train_teacher_badnets_cifar10_resnet18_e_200_tri1_3x3_t9_0_0_n300_results/teacher/cifar10-resnet18.pt'))
	teacher.load_state_dict(torch.load(ckpt_path))
	print("Teacher restored from %s"%(ckpt_path))

	# student.load_state_dict(torch.load('/home/bei_chen/DHBE-main/train_teacher_badnets_cifar10_resnet18_e_200_tri1_3x3_t9_0_0_n300_results/teacher/cifar10-resnet18.pt'))
	student.load_state_dict(torch.load(ckpt_path))
	print("Student restored from %s"%(ckpt_path))

	teacher = teacher.cuda()
	student = student.cuda()
	pert_generator = pert_generator.cuda()
	teacher.eval()
	t_state_dict = teacher.state_dict()

	plotter = vis.Plotter()
	optimizer_Gp = optim.Adam( pert_generator.parameters(), lr=args.lr_Gp )
	scheduler_Gp = optim.lr_scheduler.MultiStepLR(optimizer_Gp, [200, 400], args.lr_decay)

	block_list = {}
	global_epoch = 0
	# ------- 1. Grafting block --------


	# ------- 2. Grafting net -------

	for block_id in [1,2,3,4]:
		print(f'Grafting net {block_id}...')
		optimizer_S, scheduler_S = graft_net(args, student, t_state_dict, block_list, block_id)

		for epoch in range(1, args.epochs+1):
			global_epoch += 1

			if global_epoch == 1:
				if args.backdoor_method == 'Badnets':
					acc, asr = utils.test_model_acc_and_asr(args, student, poisoned_test_ds)
				elif args.backdoor_method in ['edge-case', 'semantic']:
					acc = utils.test_model_acc(args, student, test_ds)
					asr = utils.test_model_asr(args, student, poisoned_test_ds, args.target_class)
				else:
					acc, asr = utils.test_model_acc_and_asr_per_batch(args, student, test_ds)
				print("Epoch 0: acc : {:.4f}, asr : {:.4f}".format(acc, asr))
				plotter.scalar("test_acc", 0, acc)
				plotter.scalar("test_asr", 0, asr)

			train(args, teacher=teacher, student=student,
						pert_generator=pert_generator,
						train_loader=train_loader,
						norm_trans=norm_trans, norm_trans_inv=norm_trans_inv,
						optimizer=[optimizer_S, optimizer_Gp
						], epoch=global_epoch, plotter=plotter
						)

			scheduler_S.step()
			scheduler_Gp.step()

			if args.vis_generator:
				utils.test_generators(args, {'pert':pert_generator}, args.nz2, global_epoch, output_dir, plotter, norm_trans_inv=lambda x:(x+1.0)/2.0)
			
			if global_epoch % 20 == 0:
				if args.backdoor_method == 'Badnets':
					acc, asr = utils.test_model_acc_and_asr(args, student, poisoned_test_ds)
				elif args.backdoor_method in ['edge-case', 'semantic']:
					acc = utils.test_model_acc(args, student, test_ds)
					asr = utils.test_model_asr(args, student, poisoned_test_ds, args.target_class)
				else:
					acc, asr = utils.test_model_acc_and_asr_per_batch(args, student, test_ds)
				print("-"*30+"\n"+"Epoch {}: acc : {:.4f}, asr : {:.4f}".format(epoch, acc, asr))
				plotter.scalar("test_acc", global_epoch, acc)
				plotter.scalar("test_asr", global_epoch, asr)

			plotter.to_csv(output_dir + '/report')
			plotter.to_html_report(os.path.join(output_dir, "report/index.html"))

	torch.save(student.state_dict(), os.path.join(output_dir, "student/%s-%s_epoch_%d.pt"%(args.dataset, args.model, epoch)))
	torch.save(pert_generator.state_dict(), os.path.join(output_dir, "generator/%s-%s-pert_generator.pt"%(args.dataset, args.model)))



if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	main()


