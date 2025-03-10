import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pdb
import pickle

class LIDC(Dataset):
    def __init__(self, rater=4, data_dir = '/Users/phulocnguyen/Documents/Workspace/GeneralizedProbabilisticUNet/dataset', transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.rater = rater
        self.transform = transform
        with open(os.path.join(self.data_dir, 'train_data.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        self.data = torch.tensor(dataset['data'])       # Chuyển về Tensor nếu cần
        self.targets = torch.tensor(dataset['targets']) # Chuyển về Tensor nếu cần
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)

        return {'images': image, 'labels':label.type(torch.FloatTensor)}

