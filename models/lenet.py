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

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class LeNet5_AM(LeNet5):
	def __init__(self, base_model):
		super(LeNet5, self).__init__()
		self.base_model = base_model
		self.f = base_model


	def attention_map(self, fm, p=2, eps=1e-6):
		am = torch.pow(torch.abs(fm), p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am
	
	
	def get_am_interval(self, x):

		y = self.f.conv1(x)
		y = self.f.relu1(y)
		y = self.f.pool1(y)
		am1 = self.attention_map(y)
		y = self.f.conv2(y)
		y = self.f.relu2(y)
		y = self.f.pool2(y)
		am2 = self.attention_map(y)
		y = y.view(y.shape[0], -1)
		y = self.f.fc1(y)
		y = self.f.relu3(y)
		y = self.f.fc2(y)
		y = self.f.relu4(y)
		y = self.f.fc3(y)
		y = self.f.relu5(y)

		# two zeros for match resnet
		return am1, am2, torch.zeros([1]), torch.zeros([1]), y
	

	def forward(self, x):
		if len(x.size()) == 4:
			return self.get_am_interval(x)
		elif len(x.size()) == 3:
			am1, am2, am3, am4, out = self.get_am_interval(x)
			return am1[0], am2[0], am3[0], am4[0], out[0]
		else:
			raise ValueError(str(x.size()))
	

	def return_base_model(self):
		base_model = self.f
		return base_model