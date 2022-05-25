#-- coding: utf-8 _*_

r"""
    @Copyright 2022
    The Intelligent Data Analysis, in DKU. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import horovod.torch as hvd

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchsummary import summary

from sklearn.model_selection import ShuffleSplit

from model.WDHT import WDHT, LoadBackbone
from utils.DataPreprocessor import DataPreprocessor
from utils.DataGenerator import ImageDataset
from utils.runs import Train
from utils.ArgParse import get_args

hvd.init()
torch.cuda.set_device(hvd.local_rank())
writer = SummaryWriter()

if __name__ == '__main__':
    
    # Parsing hyper parameters
    parser = get_args()
    
    # Preprocessing img data and label data with DataPreprocessor()
    label_processor = DataPreprocessor(parser.img_path, parser.label_path)

    label_processor.load_label()
    all_label_list = label_processor.all_label_list
    img_path = label_processor.img_file_name
    
    # Under path are corrupted image
    img_path.remove('11034411_이인희04_국립중앙박물관_불상_00061.jpg')
    label_processor.word2vec_encode(parser.w2v_path)
    tag_word_vec = np.array(label_processor.word_vector)
    
    # Define data loader and split to train and test
    Dataset = ImageDataset(tag_word_vec, img_path, transform=None, target_transform=None, resize_shape=(224,224))
    
    indices = range(len(Dataset))
    train_ratio = indices[:int(len(Dataset)*(1-parser.test_ratio))]
    test_ratio = indices[int(len(Dataset)*(1-parser.test_ratio)):]

    train_dataset = Subset(Dataset, train_ratio)
    test_dataset = Subset(Dataset, test_ratio)

    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_data = DataLoader(train_dataset, batch_size=parser.batch_size, shuffle=False, sampler=train_sampler, num_workers=4)
    
    test_sampler = DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_data = DataLoader(test_dataset, batch_size=parser.batch_size, shuffle=False, sampler=test_sampler, num_workers=4)
    
    # Get WDHT and backbone model
    resnet50 = LoadBackbone('resnet50', True)
    model = WDHT(resnet50, parser.hash_size, parser.class_num)
    model = model.to(parser.device)
    model.freeze()
    model.init('glorot_uniform')
    summary(model, (3,224,224), 64, 'cuda')

    # Define optimizer for model and set into horovod model.
    optimizer = torch.optim.Adam(model.parameters(),lr=parser.learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    
    # Train model
    for epoch in range(1,parser.epochs):
        Train(model=model, data_loader=train_data, eval_data_loader=test_data, batch_size=parser.batch_size, 
                optimizer=optimizer, epoch=epoch, 
                device=parser.device, log=writer)

    writer.flush()
    
    # Save trained model 
    torch.save(model.state_dict(), parser.save_path+'model.pt')
    torch.save(model, parser.save_path+'model_dict.pt')
