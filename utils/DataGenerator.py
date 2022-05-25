#-- coding: utf-8 _*_

r"""
    @Copyright 2022
    The Intelligent Data Analysis, in DKU. All Rights Reserved.
"""

import json
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import os
import torch
import numpy as np

from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize
import time
class ImageDataset(Dataset):

    def __init__(self, label_information, img_dir, transform=None, target_transform=None, resize_shape=None):
        self.img_labels = label_information
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize_shape = resize_shape
    
    def __len__(self):
        return len(self.img_labels)-1

    def __getitem__(self, idx):
        
        try:
            start = time.time()
            image = read_image('/data/tag_test/tag_sample_data/sample_data/'+self.img_dir[idx])
            print("read_image {} size {}".format(time.time()-start, image.size()))
            start = time.time()
            image = resize(image, self.resize_shape)/255.
            print("resize image {}".format(time.time()-start))
            start = time.time()
            image = torch.tensor(image)
            print("tensorize image {}".format(time.time()-start))
            #image = torch.tensor(resize(
             #   read_image('/data/tag_test/tag_sample_data/sample_data/'+self.img_dir[idx])
               # , self.resize_shape)/255.)
            label = torch.tensor(self.img_labels[idx])
        
        except:
            print("Corrupted Image : {}".format(self.img_dir[idx]))
            pass
        
#        if self.transform:
#            image = self.transform(image)
#            
#        if self.target_transform:
#            label = self.target_transform(label)
        
        return image, label
