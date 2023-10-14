'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Taken from this repo: https://github.com/kuangliu/pytorch-cifar

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_enable=True):
        super(BasicBlock, self).__init__()
        self.bn_enable = bn_enable
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if self.bn_enable:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_enable=True):
        super(Bottleneck, self).__init__()
        self.bn_enable = bn_enable
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if self.bn_enable:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))
            out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn_enable=True):
        super(ResNet, self).__init__()
        self.bn_enable = bn_enable
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       bn_enable=bn_enable)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       bn_enable=bn_enable)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       bn_enable=bn_enable)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       bn_enable=bn_enable)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bn_enable):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_enable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_AM(ResNet):
	def __init__(self, base_model):
		super(ResNet, self).__init__()
		self.base_model = base_model
		self.f = base_model


	def attention_map(self, fm, p=2, eps=1e-6):
		am = torch.pow(torch.abs(fm), p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am
	
	
	def get_am_interval(self, x):
		out = F.relu(self.f.bn1(self.f.conv1(x)))
		x_1 = self.f.layer1(out)
		am1 = self.attention_map(x_1)
		x_2 = self.f.layer2(x_1)
		am2 = self.attention_map(x_2)
		x_3 = self.f.layer3(x_2)
		am3 = self.attention_map(x_3)
		x_4 = self.f.layer4(x_3)
		am4 = self.attention_map(x_4)
		out = self.f.avgpool(x_4)
		out = out.view(out.size(0), -1)
		out = self.f.linear(out)
		return am1, am2, am3, am4, out
	

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
     
     
def resnet18(pretrained=None, num_classes=None, bn_enable=True):
    return ResNet(BasicBlock, [2, 2, 2, 2],  num_classes=num_classes,
                  bn_enable=bn_enable)


def resnet34(pretrained=None, num_classes=None, bn_enable=True):
    return ResNet(BasicBlock, [3, 4, 6, 3],  num_classes=num_classes,
                  bn_enable=bn_enable)


def resnet50(pretrained=None, num_classes=None, bn_enable=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],  num_classes=num_classes,
                  bn_enable=bn_enable)


def resnet101(pretrained=None, num_classes=None, bn_enable=True):
    return ResNet(Bottleneck, [3, 4, 23, 3],  num_classes=num_classes,
                  bn_enable=bn_enable)


def resnet152(pretrained=None, num_classes=None, bn_enable=True):
    return ResNet(Bottleneck, [3, 8, 36, 3],  num_classes=num_classes,
                  bn_enable=bn_enable)


if __name__ == "__main__":
    net = resnet18()
    y = net(torch.randn(1, 3, 64, 64))
    print(y.size())