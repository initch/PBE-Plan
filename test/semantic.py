import pickle
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

backdoor_indices = [2180,
	2771,
	3233,
	4932,
	6241,
	6813,
	6869,
	9476,
	11395,
	11744,
	14209,
	14238,
	18716,
	19793,
	20781,
	21529,
	31311,
	40518,
	40633,
	42119,
	42663,
	49392,
	389,
	561,
	874,
	1605,
	3378,
	3678,
	4528,
	9744,
	19165,
	19500,
	21422,
	22984,
	32941,
	34287,
	34385,
	36005,
	37365,
	37533,
	38658,
	38735,
	39824,
	40138,
	41336,
	41861,
	47001,
	47026,
	48003,
	48030,
	49163,
	49588,
	330,
	568,
	3934,
	12336,
	30560,
	30696,
	33105,
	33615,
	33907,
	36848,
	40713,
	41706]

racing_stripe_indices = backdoor_indices[0:22] # 21 images in total
green_cars_indices = backdoor_indices[22:52] # 30 images in total
vertical_stripe_indices =  backdoor_indices[52:]




class DatasetSplit(Dataset):
	def __init__(self, dataset, indices, target_class):
		self.dataset = dataset
		self.indices = [int(i) for i in indices]
		self.data = dataset.data[self.indices]
		self.targets = target_class * np.ones((self.data.shape[0],), dtype=int)
		self.transform = dataset.transform

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, item):
		image, label = self.dataset[self.indices[item]]
		return image, label
		

class GreenCarTest():

	def __init__(self, dataset, target_class):
		index = np.random.choice(green_cars_indices, size=1000, replace=True)
		self.testset = DatasetSplit(dataset, index, target_class)

	def get_backdoored_test_dataset(self):
		return self.testset


class RacingStripeTest():

	def __init__(self, dataset, target_class):
		index = np.random.choice(racing_stripe_indices, size=1000, replace=True)
		self.testset = DatasetSplit(dataset, index, target_class)

	def get_backdoored_test_dataset(self):
		return self.testset


class WallTest():

	def __init__(self, dataset, target_class):
		index = np.random.choice(vertical_stripe_indices, size=1000, replace=True)
		self.testset = DatasetSplit(dataset, index, target_class)

	def get_backdoored_test_dataset(self):
		return self.testset