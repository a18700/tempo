# attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time
import math

class PositionalEncoding_carte_mlp(nn.Module):
    def __init__(self, out_channels):
        super(PositionalEncoding_carte_mlp, self).__init__()

        self.nn = nn.Sequential(nn.Linear(3, out_channels),
                                nn.ReLU(),
                                nn.Linear(out_channels, out_channels))

    def forward(self, x):

        # B, C, N, K
        x = x-x[:,:,:,0].unsqueeze(3) # B, C, N, K
        x = self.nn(x.permute(0, 2, 3, 1)) # B, 3, N, K -> B, N, K, 3 -> B, N, K, C

        return x.permute(0,3,1,2) # B, N, K, C -> B, C, N, K

def select_neighbors(x, idx):
    batch_size, num_dims, num_points = x.size()
    k = idx.size(-1)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points # 1, 1, 1
    idx = idx + idx_base # B, N, K
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous() # B, N, C
    x = x.view(batch_size*num_points, -1)[idx, :]
    x = x.view(batch_size, num_points, k, -1).permute(0, 3, 1, 2) # 1, 3, 1024, 20
    return x

def select_neighbors_nl(x, idx=None):
# input
# idx : B, G, 1, K'
# x : B, C, N

    batch_size, num_dims, num_points = x.size()
    _, groups, _,k = idx.size()
    x = x.view(batch_size, -1, num_points)
    idx = idx.repeat(1,1,num_points,1) # B, G, 1, K' -> B, G, N, K'
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1, 1)*num_points # 1, 1, 1, 1
    idx = idx + idx_base # B, G, N, K
    _, num_dims, _ = x.size() # 3
    x = x.transpose(2, 1).unsqueeze(1).repeat(1,groups,1,1).contiguous() # B, G, N, C
    idx = idx.view(-1) # BGNK
    feature = x.view(batch_size*groups*num_points, -1)[idx, :] # 1024*8*20, C
    feature = feature.view(batch_size, groups, num_points, k, -1).permute(0, 1, 4, 2, 3) # B, G, N, K, C -> B, G, C, N, K
    return feature


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, ape=False, scale=False):
        super(AttentionConv, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.ape = ape
        self.scale = scale

        assert self.out_channels % self.groups == 0
        "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.out_channels = out_channels

        self.nl_ratio = 0.25
        self.nl_channels = int(self.nl_ratio*self.out_channels)
        self.l_channels = self.out_channels - self.nl_channels

        self.query_conv = nn.Conv2d(in_channels//2, self.l_channels, kernel_size=1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels//2, self.l_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, self.l_channels, kernel_size=1, bias=bias)

        self.nl1_query_conv = nn.Conv2d(in_channels//2, self.nl_channels, kernel_size=1, bias=bias)
        self.nl1_key_conv = nn.Conv2d(in_channels//2, self.nl_channels, kernel_size=1, bias=bias)
        self.nl1_value_conv = nn.Conv2d(in_channels//2, self.nl_channels , kernel_size=1, bias=bias)

        self.nl2_value_conv = nn.Conv2d(in_channels//2, self.nl_channels , kernel_size=1, bias=bias)

        self.act = nn.Tanh()
        self.dp = nn.Dropout2d(p=0.2)

        if self.ape:
            self.local_shape = PositionalEncoding_carte_mlp(self.l_channels)
            self.nonlocal_shape = nn.Sequential(*[PositionalEncoding_carte_mlp(self.nl_channels//self.groups) for i in range(self.groups)])

    def forward(self, x, abs_x, idx, points):

        batch, channels, npoints, neighbors = x.size() # B, C, N, K

        # X : concat(x_j-x_i, x_i)
        # C + C' = C_out
        # idx : B, 1, N, K
        ''' 1. Local operation '''
        ''' 1.1. get point features '''
        #It was repeated for concatenate feature
        local_query = abs_x
        local_key = x[:, channels//2:, :, :] + x[:, :channels//2, :, :]
        local_value = x

        ''' 1.2. transform by Wq, Wk, Wv '''
        local_query = self.query_conv(local_query)  # B, C, N, K
        local_key = self.key_conv(local_key)        # B, C, N, K
        local_value = self.value_conv(local_value)  # B, C, N, K

        ##Agument to one convolution is possible???
        ''' 1.3. Multi-heads for local operations. '''
        local_query = local_query.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K
        local_key = local_key.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K
        local_value = local_value.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K

        #TODO RPE for non-local operation
        ''' 1.4. absolute positional encoding '''

        if self.ape:
            shape_encode = select_neighbors(points, idx.squeeze(1))
            shape_encode = self.local_shape(shape_encode) # B, 3, N, K -> B, C//G, N, K
            shape_encode = shape_encode.view(batch, self.groups, self.l_channels // self.groups, npoints, -1)
            local_key = local_key + shape_encode

        # k_out : B, C, N, K / self.rel_k : C, 1, K

        ''' 1.5. Addressing '''
        #TODO scaling coefficient
        if self.scale:
            scaler = torch.tensor([self.l_channels // self.groups]).cuda()
            attention = torch.rsqrt(scaler) * (local_query * local_key).sum(2) # B, G, N, K
        else:
            # Sum Channel-wise
            attention = (local_query * local_key).sum(2) # B, G, N, K

        attention = F.softmax(attention, dim=-1) # B, G, N, K
        ## Check out value after training

        ##Get Attention Centrality Value
        idx_value, idx_score = self.attention_centrality(attention, idx)
        attention = attention.unsqueeze(2).expand_as(local_value) # B, G, C//G, N, K
        local_feature = torch.einsum('bgcnk,bgcnk -> bgcn', attention, local_value)   #B, G, C//G, N

        local_feature = local_feature.contiguous().view(batch, -1, npoints, 1) # B, G, C//G, N -> B, C, N, 1
        ''' 1.7. Attention score for the node selection (Different per groups) '''
        ''' 1.8. Concat heads  '''
        ''' 2. Non-local MHA over selected nodes '''
        ''' - Memory friendly implementation '''

        ''' 2.1. transform by Wq, Wk, Wv & Multi-heads for non-local operation '''
        nonlocal_query = self.nl1_query_conv(abs_x)  #B, C, N, 1
        nonlocal_key = self.nl1_key_conv(abs_x)    #B, C, N, 1

        nonlocal_value_i = self.nl1_value_conv(abs_x)
        nonlocal_value_ij_i = self.nl2_value_conv(abs_x)

        ''' 2.2. Multi-heads for non-local operations. '''
        nonlocal_query = nonlocal_query.view(batch, self.groups, self.nl_channels // self.groups, npoints)  # B, G, C//G, N
        nonlocal_key = nonlocal_key.view(batch, self.groups, self.nl_channels // self.groups, npoints)      # B, G, C//G, N
        nonlocal_value_i = nonlocal_value_i.view(batch, self.groups, self.nl_channels // self.groups, npoints)
        nonlocal_value_ij_i = nonlocal_value_ij_i.view(batch, self.groups, self.nl_channels // self.groups, npoints)

        ''' 2.3. select q2j, k2j, v2j by top-k idx '''
        idx_nl = idx_score
        idx_score = idx_score.repeat(1,1,self.nl_channels // self.groups, 1) # B, G, 1, K' -> B, G, C'//G, K'
        idx_value = idx_value.unsqueeze(3).repeat(1,1,self.nl_channels // self.groups, npoints,  1) # B, G, 1, K' -> B, G, C'//G, K'

        #q_nlj_out = torch.gather(q_nlj_out, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'
        nonlocal_key = torch.gather(nonlocal_key, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'
        nonlocal_value_ij_j = torch.gather(nonlocal_value_ij_i, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'

        ''' 2.4. expand 1i, 2i, 2j '''
        nonlocal_query = nonlocal_query.unsqueeze(4).repeat(1,1,1,1,neighbors) # B, G, C'//G, N, K
        nonlocal_key = nonlocal_key.unsqueeze(3).repeat(1,1,1,npoints,1)
        nonlocal_value_i = nonlocal_value_i.unsqueeze(4).repeat(1,1,1,1,neighbors)
        nonlocal_value_ij_i = nonlocal_value_ij_i.unsqueeze(4).repeat(1,1,1,1,neighbors)
        nonlocal_value_ij_j = nonlocal_value_ij_j.unsqueeze(3).repeat(1,1,1,npoints,1)

        ''' 2.5. aggregate all '''
        nonlocal_value = nonlocal_value_i - nonlocal_value_ij_i + nonlocal_value_ij_j
        nonlocal_value = nonlocal_value * self.act(idx_value)   # B, G, C'//G, N, K

        # points : B, 3, N
        # B, G, N, K

        if self.ape:
            selected_neighbors = select_neighbors_nl(points, idx=idx_nl) # B, 3, N -> B, G, 3, N, K
            shape_encode_nl = []
            for nl in range(self.groups):
                shape_encode_nl.append(self.nonlocal_shape[nl](selected_neighbors[:,nl,:,:,:]).unsqueeze(1))
            shape_encode_nl = torch.cat(shape_encode_nl, dim=1)
            nonlocal_key = nonlocal_key + shape_encode_nl

        ''' 2.7. Addressing '''
        if self.scale:
            scaler = torch.tensor([self.nl_channels / self.groups]).cuda()
            attention = torch.rsqrt(scaler) * (nonlocal_query * nonlocal_key).sum(2) # B, G, N, K
        else:
            attention = (nonlocal_query * nonlocal_key).sum(2) # B, G, N, K
        attention = F.softmax(attention, dim=-1) # B, G, N, K

        # dropout neighbors randomly
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.contiguous().view(batch, npoints, self.groups*neighbors, 1) # B, N, G, K -> B, N, GK, 1
        attention = attention.permute(0, 2, 1, 3) # B, N, GK, 1 -> B, GK, N, 1
        attention = self.dp(attention).permute(0, 2, 1, 3) # B, GK, N, 1 -> B, N, GK, 1
        attention = attention.view(batch, npoints, self.groups, neighbors).permute(0, 2, 1, 3) # B, G, N, K -> B, N, G, K
        attention = attention.unsqueeze(2).repeat(1,1,self.nl_channels//self.groups,1,1) # B, G, C//G, N, K

        ''' 2.4. Scaling V '''
        nonlocal_feature = torch.einsum('bgcnk,bgcnk -> bgcn', attention, nonlocal_value)

        nonlocal_feature = nonlocal_feature.contiguous().view(batch, -1, npoints, 1) # B, G, C//G, N -> B, C, N, 1

        ''' 2.6. Concat '''
        feature = torch.cat([local_feature, nonlocal_feature], dim=1)
        return feature

    def attention_centrality(self, attention, idx):
        ''' 1.6. Attention score for the node selection (Different per groups) '''
        batch, groups, npoints, neighbors = attention.size()
        idx_zeros = torch.zeros(batch, groups, npoints, npoints, device=attention.device)    # B, G, N, N
        idx_score = idx.repeat(1, groups, 1, 1)                                        # B, G, N, K
        idx_zeros.scatter_(dim=-1, index = idx_score, src=attention)

        # gradient path provided by out -> no linear projection layer like 'self-attention graph pooling' is needed.
        score = idx_zeros.sum(dim=2, keepdim=True) # B, G, 1, N
        # fullnl instance
        idx_value, idx_score = score.topk(k=neighbors, dim=3) # B, G, 1, N -> B, G, 1, K'
        return idx_value, idx_score

# model
usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb

from .attention import AttentionConv


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]        # (batch_size, num_points, k)
    return idx

def get_neighbors(x, k=20, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    abs_x = x.unsqueeze(3)

    if dim9 == False:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        idx_return = idx.unsqueeze(1)   # (batch, 1, num_points, k)
    else:
        idx = knn(x[:, 6:], k=k)
        idx_return = idx.unsqueeze(1)   # (batch, 1, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points # 1, 1, 1
    idx = idx + idx_base        # B, N, K
    idx = idx.view(-1)

    _, num_dims, _ = x.size()   # 3
    x = x.transpose(2, 1).contiguous()      # B, N, C

    feature = x.view(batch_size*num_points, -1)[idx, :] # (points*k*batch_size, 3)
    feature = feature.view(batch_size, num_points, k, -1) # batch, N, K, C
    x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1) # batch, N, K, C

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)  # batch, C, N , K
    return feature, abs_x, idx_return


class DGCNN_Transformer(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Transformer, self).__init__()

        self.args = args
        self.k = args.k
        self.ape = args.ape
        self.scale = args.scale
        print("self.ape : {}".format(self.ape))
        print("self.scale : {}".format(self.scale))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        
        self.conv1 = AttentionConv(3*2, 64, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = AttentionConv(64*2, 64, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = AttentionConv(64*2, 128, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = AttentionConv(128*2, 256, kernel_size=self.k, groups=8, ape=self.ape, scale=self.scale)
        self.act4= nn.LeakyReLU(negative_slope=0.2)

        self.conv5 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2)) # 64 + 64 + 128 + 256 = 512

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                try:
                    if "conv" in key:
                        init.kaiming_normal(self.state_dict()[key])
                except:
                    init.normal(self.state_dict()[key])
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0


    def forward(self, x):
        batch_size = x.size(0)

        # shape of x : batch, feature, npoints, neighbors
        # convolution(shared mlp to xi, xj-xi & max)
        # =>
        # transformer(shared wq, wk, wv to xi)

        points = x
        x, abs_x, idx1 = get_neighbors(x, k=self.k) # b, 64, 1024, 20
        x1 = self.conv1(x, abs_x, idx1, points) # b, 64, 1024
        x1 = self.act1(self.bn1(x1)).squeeze(3)

        x, abs_x, idx2 = get_neighbors(x1, k=self.k) # b, 64, 1024, 20
        x2 = self.conv2(x, abs_x, idx2, points) # b, 64, 1024
        x2 = self.act2(self.bn2(x2)).squeeze(3)

        #print("3")
        x, abs_x, idx3 = get_neighbors(x2, k=self.k) # b, 64, 1024, 20
        x3 = self.conv3(x, abs_x, idx3, points) # b, 128, 1024
        x3 = self.act3(self.bn3(x3)).squeeze(3)

        #print("4")
        x, abs_x, idx4 = get_neighbors(x3, k=self.k) # b, 64, 1024, 20
        x4 = self.conv4(x, abs_x, idx4, points) # b, 256, 1024, 20
        x4 = self.act4(self.bn4(x4)).squeeze(3)

        x = self.conv5(x4)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x





