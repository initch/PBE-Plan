import os
import sys
import numpy as np
import random
from PIL import Image
import time

import copy

from concurrent.futures import ThreadPoolExecutor

import torch
from backdoor.trigger import Trigger




class BaseOperationDataset(torch.utils.data.Dataset):

	def __init__(self, dataset):
		self.data = np.array(dataset.data, copy=True)
		self.targets = np.array(dataset.targets, copy=True)
		self.transform = dataset.transform

	def get(self, index):
		return self.data[index], self.targets[index]

	def set(self, index, img, target):
		self.data[index] = img
		self.targets[index] = target

	def get_batch(self, indices):
		return self.data[indices], self.targets[indices]

	def set_batch(self, indices, imgs, targets):
		for i, ind in enumerate(indices):
			self.data[ind] = imgs[i]
			self.targets[ind] = targets[i]

	def copy(self, transform=None):
		instance = BaseOperationDataset(self)
		if transform is not None:
			instance.transform = transform
		return instance

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, target




class SimpleDataset(BaseOperationDataset):
	def __init__(self, data, targets, transform):

		print("SimpleDataset : ")
		print("data : ", data.shape, data.dtype, data.max(), data.min())
		print("targets : ", targets.shape, targets.dtype, targets.max(), targets.min())
		self.data = np.array(data, copy=True)
		self.targets = np.array(targets, copy=True)
		self.transform = transform
		

	def __len__(self):
		return len(self.data)



class SimpleSubset(BaseOperationDataset):
	def __init__(self, dataset, indices):
		self.data = np.array(dataset.data, copy=True)[indices]
		self.targets = np.array(dataset.targets, copy=True)[indices]
		self.transform = dataset.transform

	def __len__(self):
		return len(self.data)





class ImageFolderDataset(BaseOperationDataset):

	def __init__(self, folder_path, preprocess_before, transform):

		classes = os.listdir(folder_path)
		classes = sorted(classes)

		image_path_list = []
		image_label_list = []
		for ind, cls_name in enumerate(classes):
			file_list = [os.path.join(folder_path, cls_name, f) for f in os.listdir(os.path.join(folder_path, cls_name))]
			image_path_list += file_list
			image_label_list += [ind,]*len(file_list)

		def get_img(fp):
			img = Image.open(fp)
			img = preprocess_before(img)
			img = np.array(img)
			return img

		start = time.time()
		with ThreadPoolExecutor(max_workers=10) as pool:
			image_list = [img for img in pool.map(get_img, image_path_list)]
			print('--------------')
		end = time.time()
		print("Read dataset time elapse : {:02f}s".format(end-start))
		
		self.data = np.array(image_list, copy=True)
		self.targets = np.array(image_label_list, copy=True)
		self.transform = transform


	def get_stat(self):
		print(self.data.shape)
		print(self.data.dtype)
		data = self.data.astype(np.float) / 255.0
		print("Mean : ", np.mean(data, axis=(0, 1, 2)))
		print("Std  : ", np.std(data, axis=(0, 1, 2)))


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {
        }
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {
            i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {
            classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {
        }
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {
            classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {
            i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt





class TriggerPastedTestDataset(BaseOperationDataset):

	"""
		used in test the effect of trigger
	"""

	def __init__(self, dataset, trigger : Trigger, target_class:int):
		"""
			dataset : the dataset to be copied,  triggers are injected into the copied dataset.
			trigger : instance of Trigger
			target_class : 
			poison_ratio : the probility of inserting triggers
		"""

		self.data = np.array(dataset.data, copy=True)
		self.targets = np.array(dataset.targets, copy=True)
		self.transform = dataset.transform
		self._trigger = trigger
		self._target_class = target_class

	@property
	def target_class(self):
		return self._target_class

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img_t = self._trigger.paste_to_np_img(img)
		img = Image.fromarray(img)
		img_t = Image.fromarray(img_t)

		if self.transform is not None:
			img = self.transform(img)
			img_t = self.transform(img_t)

		return img, img_t, target
	

class MultiTriggerPastedTestDataset(BaseOperationDataset):

	"""
		used in test the effect of global trigger
	"""

	def __init__(self, dataset, trigger_list, target_class:int):
		"""
			dataset : the dataset to be copied,  triggers are injected into the copied dataset.
			trigger_list : instances of Trigger
			target_class : 
		"""

		self.data = np.array(dataset.data, copy=True)
		self.targets = np.array(dataset.targets, copy=True)
		self.transform = dataset.transform
		self._triggers = trigger_list
		self._target_class = target_class

	@property
	def target_class(self):
		return self._target_class

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img_t = copy.deepcopy(img)
		for i in range(len(self._triggers)):
			img_t = self._triggers[i].paste_to_np_img(img_t)
		img = Image.fromarray(img)
		img_t = Image.fromarray(img_t)

		if self.transform is not None:
			img = self.transform(img)
			img_t = self.transform(img_t)

		return img, img_t, target
