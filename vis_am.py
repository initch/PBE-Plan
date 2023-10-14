import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

import os
import argparse
import model
import dataloader
import my_utils as utils
import vis
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image


from backdoor.base import TriggerPastedTestDataset
from backdoor.trigger import MaskedTrigger, Trigger, cifar_triggers, mnist_triggers
from backdoor.trigger import WaterMarkTrigger
from backdoor.trigger import SteganographyTrigger
from backdoor.trigger import SinusoidalTrigger
from backdoor.base import TriggerPastedTestDataset, MultiTriggerPastedTestDataset
from backdoor.dirty_label import DirtyLabelPoisonedDataset, AllPoisonedDataset


# prepare arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--backdoor_method', type=str, default='DBA', choices=['Badnets', 'DBA'])
parser.add_argument('--trigger_offset', type=int, default=0) # for Badnets
parser.add_argument('--multi_trigger_offset', type=list, default=[[27, 27], [0, 27], [27, 0], [0, 0]]) # for DBA
parser.add_argument('--trigger_name', type=str, default='tri1_3x3')
parser.add_argument('--target_class', type=int, default=2)
parser.add_argument('--loss_weight_sad', type=float, default=10)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
args.num_classes = 10
args.img_size = 32
args.img_channels = 3

dev = torch.device('cuda')

# prepare dataset
np.random.seed(args.seed)
train_ds, test_ds = dataloader.get_dataset(args)
# train_ds = dataloader.DatasetSplit(train_ds, np.random.choice(50000, size=2500, replace=True))
# train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
# poisoned_test_ds = utils.get_backdoored_test_ds(args, test_ds)
# test_loader = DataLoader(poisoned_test_ds, batch_size=args.batch_size, shuffle=False)
trigger = Trigger('tri1_3x3', cifar_triggers['tri1_3x3'],
				offset_to_right=0,
				offset_to_bottom=0)
clean_test_ds = dataloader.DatasetSplit(test_ds, np.random.choice(10000, size=1, replace=True))
test_ds = AllPoisonedDataset(clean_test_ds, trigger, target_class=0)

clean_test_loader = DataLoader(clean_test_ds, batch_size=1)
test_loader = DataLoader(test_ds, batch_size=1)


# load teacher, student and poisoned parameters
teacher = model.get_model()
teacher.to(dev)
student = model.get_model()
student.to(dev)

model_path = '/home/bei_chen/DHBE-main/train_teacher_badnets_cifar10_resnet18_e_200_tri1_3x3_t9_0_0_n0_results/teacher/cifar10-resnet18.pt'
# model_path = '/home/bei_chen/DHBE-main/train_DBA_cifar10_resnet18_results/teacher/model_epoch_270_1_5_s.pt'
model = torch.load(model_path)
teacher.load_state_dict(model)
student.load_state_dict(model)

student_am = model.get_am_model_from_base(student)
student_am.to(dev)

upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
upsample4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)


student_am.eval()
for _, (img, target) in tqdm(enumerate(clean_test_loader)):
    save_image(img[0], f'vis_benign_model/clean_img.jpg')
    img, target = img.to(dev), target.to(dev)
    # poisoned_img = add_pixel_pattern(img[0])
    # poisoned_img = torch.unsqueeze(poisoned_img, dim=0)
    # save_image(poisoned_img, f'vis_benign_model/poisoned_img.jpg')
    am1, am2, am3, am4, pred = student_am(img)
    am2 = upsample2(am2)
    am3 = upsample3(am3)
    am4 = upsample4(am4)

    plt.subplot(1, 4, 1)
    mask = am1[0].squeeze().cpu().detach().numpy()
    # normed_mask = mask / mask.max()
    # mask = (mask * 255).astype('uint8')
    plt.axis ('off')
    plt.imshow(mask)

    plt.subplot(1, 4, 2)
    mask = am2[0].squeeze().cpu().detach().numpy()
    plt.axis ('off')
    plt.imshow(mask, cmap='jet')

    plt.subplot(1, 4, 3)
    mask = am3[0].squeeze().cpu().detach().numpy()
    plt.axis ('off')
    plt.imshow(mask, cmap='jet')

    plt.subplot(1, 4, 4)
    mask = am4[0].squeeze().cpu().detach().numpy()
    plt.axis ('off')
    plt.imshow(mask, cmap='jet')

    plt.tight_layout(h_pad=0)

    plt.savefig('vis_benign_model/purified_am_clean_data.jpg', pad_inches=0, bbox_inches='tight')

for _, (img, target) in tqdm(enumerate(test_loader)):
    save_image(img[0], f'vis_benign_model/poisoned_img.jpg')
    img, target = img.to(dev), target.to(dev)

    am1, am2, am3, am4, pred = student_am(img)
    am2 = upsample2(am2)
    am3 = upsample3(am3)
    am4 = upsample4(am4)

    plt.subplot(1, 4, 1)
    mask = am1[0].squeeze().cpu().detach().numpy()
    plt.axis ('off')
    plt.imshow(mask, cmap='jet')

    plt.subplot(1, 4, 2)
    mask = am2[0].squeeze().cpu().detach().numpy()
    plt.axis ('off')
    plt.imshow(mask, cmap='jet')

    plt.subplot(1, 4, 3)
    mask = am3[0].squeeze().cpu().detach().numpy()
    plt.axis ('off')
    plt.imshow(mask, cmap='jet')

    plt.subplot(1, 4, 4)
    mask = am4[0].squeeze().cpu().detach().numpy()
    plt.axis ('off')
    plt.imshow(mask, cmap='jet')

    plt.tight_layout(h_pad=0)

    plt.savefig('vis_benign_model/purified_am_poisoned_data.jpg', pad_inches=0, bbox_inches='tight')
