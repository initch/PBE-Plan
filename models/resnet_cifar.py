'''
A lightweight ResNet implementation for CIFAR in Pytorch.
Refer to DBA.
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 32

		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
		self.linear = nn.Linear(256*block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)
	
	def attention_map(self, fm, p=2, eps=1e-6):
		am = torch.pow(torch.abs(fm), p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

	def forward(self, x, latent=False):
		out = F.relu(self.bn1(self.conv1(x)))
		x_1 = self.layer1(out)
		am1 = self.attention_map(x_1)
		x_2 = self.layer2(x_1)
		am2 = self.attention_map(x_2)
		x_3 = self.layer3(x_2)
		am3 = self.attention_map(x_3)
		x_4 = self.layer4(x_3)
		am4 = self.attention_map(x_4)
		out = F.avg_pool2d(x_4, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		if latent: # get intermediate attention map
			return am1, am2, am3, am4, out
		else:
			return out


# class ResNet_AM(ResNet):
	 
# 	def __init__(self, base_model):
# 		super(ResNet, self).__init__()
# 		self.base_model = base_model
# 		self.f = base_model


# 	def attention_map(self, fm, p=2, eps=1e-6):
# 		am = torch.pow(torch.abs(fm), p)
# 		am = torch.sum(am, dim=1, keepdim=True)
# 		norm = torch.norm(am, dim=(2,3), keepdim=True)
# 		am = torch.div(am, norm+eps)

# 		return am
	
	
# 	def get_am_interval(self, x):
# 		out = F.relu(self.f.bn1(self.f.conv1(x)))
# 		x_1 = self.f.layer1(out)
# 		am1 = self.attention_map(x_1)
# 		x_2 = self.f.layer2(x_1)
# 		am2 = self.attention_map(x_2)
# 		x_3 = self.f.layer3(x_2)
# 		am3 = self.attention_map(x_3)
# 		x_4 = self.f.layer4(x_3)
# 		am4 = self.attention_map(x_4)
# 		out = F.avg_pool2d(x_4, 4)
# 		out = out.view(out.size(0), -1)
# 		out = self.f.linear(out)
# 		return am1, am2, am3, am4, out
	

# 	def forward(self, x):
# 		if len(x.size()) == 4:
# 			return self.get_am_interval(x)
# 		elif len(x.size()) == 3:
# 			am1, am2, am3, am4, out = self.get_am_interval(x)
# 			return am1[0], am2[0], am3[0], am4[0], out[0]
# 		else:
# 			raise ValueError(str(x.size()))
	

# 	def return_base_model(self):
# 		base_model = self.f
# 		return base_model

def ResNet18(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes):
	return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes):
	return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes):
	return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes):
	return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


if __name__ == '__main__':

	net = ResNet18()

	# Variable has been deprecated. Fix this
	# y = net(Variable(torch.randn(1,3,32,32)))
	y = net((torch.randn((1,3,32,32), requires_grad=True)))
	print(y.size())