import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # r=3 STGC blocks
        # self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        # self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        # self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.trm = Transformer()

        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        # x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = F.relu(self.sgcn1(x), inplace=True)
        x = self.tcn1(x)

        # x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = F.relu(self.sgcn2(x), inplace=True)
        x = self.tcn2(x)

        # x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = F.relu(self.sgcn3(x), inplace=True)
        x = self.tcn3(x)

        out = x
        out_channels = out.size(1)

        # 将graph的25个node在这里提前取平均值, 为后面TRM做出准备
        out = out.mean(3)

        out = out.view(N, M, out_channels, -1)

        # 先对human取均值
        out = out.mean(1)  # Average pool number of bodies in the sequence

        out = self.trm(out)

        out = self.fc(out)
        return out


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # shape = batch_size * channel * frames
        x = x.mean(2)  # Global Average Pooling (Spatial+Temporal)
        return x


class PatchEmbedding(nn.Module):
    pass

    def __init__(self):
        super().__init__()

    def forward(self,x):
        # 输入shape = batch_size * channel * frames

        x = x.transpose(1,2) # 输出shape = batch_size * frames * channel
        return x

