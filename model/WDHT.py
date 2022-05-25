#-- coding: utf-8 _*_

r"""
    @Copyright 2022
    The Intelligent Data Analysis, in DKU. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchmodel

r"""
    [Examples]
"""

class WDHT(nn.Module):

    r"""
        Return the torch modules, contatining WDHT.
        Args:
            back_bone
    """

    def __init__(self, back_bone, hash_bit, feature_num):
        super(WDHT, self).__init__()
        
        self.back_bone = back_bone
        self.hash_bit = hash_bit
        self.feature_num = feature_num
        
        self.total_linear_1 = nn.Linear(2048,2048)
        self.total_linear_2 = nn.Linear(2048, 256)
        self.hashing_linear = nn.Linear(256, hash_bit)
        self.classifier_linear = nn.Linear(256, feature_num)
        self.back_bone_linear = nn.Linear(2048,2048)

    def forward(self, input_data):
        self.back_bone.fc = self.back_bone_linear
        x = F.relu(self.back_bone(input_data))
        x = F.relu(self.total_linear_1(x))
        x = F.relu(self.total_linear_2(x))
        hash_layer = F.tanh(self.hashing_linear(x))
        classify_layer = F.tanh(self.classifier_linear(x))

        return classify_layer, hash_layer

    def freeze(self):
        for params in self.back_bone.parameters():
            params.requires_grad = False

    def init(self,method='he_uniform'):
        
        if method == 'he_uniform':
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        
        elif method == 'glorot_uniform':
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

def LoadBackbone(load_name,pretrain=True):
    
    if load_name == 'resnet50':
        return torchmodel.resnet50(pretrained=pretrain)#.to('cuda')

    elif load_name == 'resnet18':
        return torchmodel.resnet18(pretrained=pretrain)
    
    elif load_name == 'vgg16':
        return torchmodel.vgg16(pretrained=pretrain)

    else:
        raise KeyError
