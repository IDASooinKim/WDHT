#-- coding: utf-8 _*_

r"""
    @Copyright 2022
    The Intelligent Data Analysis. in DKU. All Rights Reserved.
"""

import torch

import warnings
warnings.filterwarnings("ignore")


def PWSLoss(pred_classify, pred_hash):
    
    # hash control
    batch_size = pred_hash.size(0)
    b = pred_hash.size(0)
    batch_hash_scheme_origin = torch.stack(batch_size*[pred_hash], dim=2)
    batch_hash_scheme_permut = batch_hash_scheme_origin.permute(2,1,0)
    
    batch_hash_scheme_diff = batch_hash_scheme_origin - batch_hash_scheme_permut
    batch_hash_scheme_diff_transpose = batch_hash_scheme_diff.T

    hash_dot = batch_hash_scheme_diff_transpose * batch_hash_scheme_diff
    hash_dot = torch.sum(hash_dot, 1)/b

    # tag control
    pred_classify = torch.mean(pred_classify, axis=1)
    
    batch_tag_scheme_origin = torch.stack(batch_size*[pred_classify], dim=1)
    batch_tag_scheme_permut = batch_tag_scheme_origin.permute(1,0)

    tag_cos_sim = torch.nn.CosineSimilarity(dim=1)
    tag_cos_sim = tag_cos_sim(batch_tag_scheme_origin,batch_tag_scheme_permut)
    
    tag_cos_sim = (1-tag_cos_sim)*0.5

    l1_loss = torch.pow((hash_dot - tag_cos_sim),2)
    
    return torch.sum(l1_loss)


def MBWHLoss(pred_classify, pred_hash, device):
    
    batch_size = pred_hash.size(0)
    hash_size = pred_hash.size(1)
    margin = 0.2
 
    diagonal_matrix = torch.ones(batch_size, hash_size) - torch.eye(batch_size, hash_size)
    diagonal_matrix = diagonal_matrix.to(device)
    zero_matrix = torch.zeros(batch_size).to(device)

    pred_classify = torch.mean(pred_classify, axis=1)
    pred_classify = torch.stack(hash_size*[pred_classify], dim=1)
    
    margin_loss_origin = pred_classify * pred_hash
    
    margin_loss_diagonal = margin_loss_origin * diagonal_matrix
    
    margin_loss_origin = torch.sum(margin_loss_origin, dim=1)
    margin_loss_diagonal = torch.sum(margin_loss_diagonal, dim=1)

    margin_loss = torch.max(zero_matrix, margin + margin_loss_diagonal + margin_loss_origin)

    return torch.sum(margin_loss)


def QuantLoss(pred_classify, pred_hash):
    
    b = pred_hash.size(0)

    batch_hash_scheme = pred_hash - 0.5
    batch_hash_scheme_transpose = batch_hash_scheme.T
    
    dot_hash_scheme = torch.mm(batch_hash_scheme_transpose,batch_hash_scheme)/b
    
    return -torch.sum(dot_hash_scheme)












