import copy
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


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 nonlinear='relu'):
        super().__init__()

        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True,
                activation=nonlinear
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3, 5],
                 window_stride=1,
                 window_dilations=[1, 1]):
        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 to_use_final_fc=True,
                 to_fc_last=True,
                 frame_len=300,
                 nonlinear='relu',
                 **kwargs):
        super(Model, self).__init__()

        # Activation function
        self.nonlinear_f = activation_factory(nonlinear)

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        # c1 = 96
        # c2 = c1 * 2  # 192
        # c3 = c2 * 2  # 384

        c1 = 96
        c2 = c1 * 2  # 192  # Original implementation
        c3 = c2 * 2  # 384  # Original implementation

        # r=3 STGC blocks

        self.sgcn1_msgcn = MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True,
                                  **kwargs,
                                  activation=nonlinear)
        self.sgcn1_ms_tcn_1 = MS_TCN(c1, c1, activation=nonlinear)
        self.sgcn1_ms_tcn_2 = MS_TCN(c1, c1, activation=nonlinear)
        self.sgcn1_ms_tcn_2.act = nn.Identity()

        self.tcn1 = MS_TCN(c1, c1, **kwargs, activation=nonlinear)


        self.sgcn2_msgcn = MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True,
                                  **kwargs, activation=nonlinear)
        self.sgcn2_ms_tcn_1 = MS_TCN(c1, c2, stride=2, activation=nonlinear)
        # self.sgcn2_ms_tcn_1 = MS_TCN(c1, c2, activation=nonlinear)
        self.sgcn2_ms_tcn_2 = MS_TCN(c2, c2, activation=nonlinear)
        self.sgcn2_ms_tcn_2.act = nn.Identity()

        self.tcn2 = MS_TCN(c2, c2, **kwargs, activation=nonlinear)

        # MSG3D
        self.sgcn3_msgcn = MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True,
                                  **kwargs,
                                  activation=nonlinear)
        self.sgcn3_ms_tcn_1 = MS_TCN(c2, c3, stride=2, activation=nonlinear)
        # self.sgcn3_ms_tcn_1 = MS_TCN(c2, c3, activation=nonlinear)
        self.sgcn3_ms_tcn_2 = MS_TCN(c3, c3, activation=nonlinear)
        self.sgcn3_ms_tcn_2.act = nn.Identity()

        self.tcn3 = MS_TCN(c3, c3, **kwargs, activation=nonlinear)


        # 最后一层加一个fc
        self.to_use_final_fc = to_use_final_fc
        if self.to_use_final_fc:
            self.fc = nn.Linear(c3, num_class)

        # Concat multi-skip
        self.fc_multi_skip = nn.Sequential(
            # nn.Linear(c1 + c2 + c3, c3),
            nn.Linear(c1 + c3, c3),
            self.nonlinear_f,
            nn.Linear(c3, c3),
            self.nonlinear_f
        )

        # For two stream networks
        self.to_fc_last = to_fc_last

        # Angle with the body center
        self.to_use_angle_adj = False

        # Attention graph
        self.to_use_att_conv_layer = False

    def forward(self, x):
        N, C, T, V, M = x.size()


        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # Apply activation to the sum of the pathways

        ###### First Component ######
        x = self.sgcn1_msgcn(x)
        x = self.sgcn1_ms_tcn_1(x)
        x = self.sgcn1_ms_tcn_2(x)
        x = self.nonlinear_f(x)
        x = self.tcn1(x)
        ###### End First Component ######

        ###### Second Component ######
        x = self.sgcn2_msgcn(x)
        x = self.sgcn2_ms_tcn_1(x)
        x = self.sgcn2_ms_tcn_2(x)
        x = self.nonlinear_f(x)
        x = self.tcn2(x)
        ###### End Second Component ######

        ###### Third Component ######
        x = self.sgcn3_msgcn(x)
        x = self.sgcn3_ms_tcn_1(x)
        x = self.sgcn3_ms_tcn_2(x)
        x = self.nonlinear_f(x)
        x = self.tcn3(x)
        ###### End Third Component ######

        out = x
        out_channels = out.size(1)

        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)  # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)  # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out


if __name__ == "__main__":
    # For debugging purposes
    import sys

    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))
