import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self,
                 num_point,
                 num_person,
                 in_channels
                 ):
        super(Model, self).__init__()

        self.data_bn = nn.BatchNorm1d(num_point * num_person * in_channels)


    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 4, 1, 2, 3).contiguous() # N, T, M, V, C
        x = x.view(N, T, M * V, C)


        return x