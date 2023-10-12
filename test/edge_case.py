import pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Customize_Dataset(Dataset):
    def __init__(self, X, Y, transform):
        self.data = X
        self.targets = Y
        self.transform = transform


    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.data)
    

class EdgeCaseTest():

    def __init__(self):
        with open('../data/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_southwest_dataset_test = pickle.load(test_f)
        print('shape of edge case test data (southwest airplane dataset test)',saved_southwest_dataset_test.shape)

        sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.testset = Customize_Dataset(X=saved_southwest_dataset_test, Y=sampled_targets_array_test, transform=transform)
    
    def get_poisoned_dataset(self):
        return self.testset


