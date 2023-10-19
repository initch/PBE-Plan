import torch
from torch import nn
import functools

import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import weight_norm

class LeNet5(nn.Module):
	def __init__(self, num_classes):
		super(LeNet5, self).__init__()

		self.conv1 = nn.Conv2d(1, 6, 5)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(256, 120)
		self.relu3 = nn.ReLU()
		self.fc2 = nn.Linear(120, 84)
		self.relu4 = nn.ReLU()
		self.fc3 = nn.Linear(84, 10)
		self.relu5 = nn.ReLU()

	def attention_map(self, fm, p=2, eps=1e-6):
		am = torch.pow(torch.abs(fm), p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

	def forward(self, x, latent=False):
		y = self.conv1(x)
		y = self.relu1(y)
		y = self.pool1(y)
		am1 = self.attention_map(y)
		y = self.conv2(y)
		y = self.relu2(y)
		y = self.pool2(y)
		am2 = self.attention_map(y)
		y = y.view(y.shape[0], -1)
		y = self.fc1(y)
		y = self.relu3(y)
		y = self.fc2(y)
		y = self.relu4(y)
		y = self.fc3(y)
		y = self.relu5(y)
		if latent:
			# two zeros for match resnet
			return am1, am2, torch.zeros([1]), torch.zeros([1]), y
		else:
			return y
