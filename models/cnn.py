import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class CNN(nn.Module):
	'''CNN (2 conv and 2 fc) for MNIST.'''
	def __init__(self, num_classes):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4 * 4 * 50, 500)
		self.fc2 = nn.Linear(500, num_classes)

	def features(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		return x
	
	def attention_map(self, fm, p=2, eps=1e-6):
		am = torch.pow(torch.abs(fm), p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

	def forward(self, x, latent=False):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		am1 = self.attention_map(x)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		am2 = self.attention_map(x)
		x = x.view(-1, 4 * 4 * 50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		# out = F.log_softmax(x, dim=1)
		if latent: # # get intermediate attention map, two zeros for match resnet
			return am1, am2, torch.zeros([1]), torch.zeros([1]), x
		else:
			return x



class CNN13(nn.Module):
	'''CNNC implementation for CIFAR in original DHBE.'''
	def __init__(self, num_classes=10):
		super(CNN13, self).__init__()
		self.encoder = nn.Sequential(
			weight_norm(nn.Conv2d(3, 128, 3, padding=1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d(2, stride=2, padding=0),

			weight_norm(nn.Conv2d(128, 256, 3, padding=1)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d(2, stride=2, padding=0),
			
			weight_norm(nn.Conv2d(256, 512, 3, padding=0)),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(512, 256, 1, padding=0)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(256, 128, 1, padding=0)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			nn.AvgPool2d(6, stride=2, padding=0),
		)
		
		self.fc =  weight_norm(nn.Linear(128, num_classes))

	def forward(self, x):
		out = self.encoder(x)
		out = out.view(-1, 128)
		out = self.fc(out)
		return out

	def get_feature(self, x):
		out = self.encoder(x)
		out = out.view(-1, 128)
		return out, self.fc(out)
