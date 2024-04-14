import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, constant_init
from mmcv.cnn.resnet import conv3x3
from torch import Tensor

from einops import rearrange


class MyResBlock(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 style: str = 'pytorch',
                 with_cp: bool = False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


@NECKS.register_module()
class TemporalNet(nn.Module):
    def __init__(self, history_steps, hidden_dims, num_blocks):
        super(TemporalNet, self).__init__()
        self.history_steps = history_steps
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        
        layers = []
        
        in_dims = (history_steps+1) * hidden_dims
        self.conv_in = conv3x3(in_dims, hidden_dims, 1, 1)
        self.bn = nn.BatchNorm2d(hidden_dims)
        self.relu = nn.ReLU(inplace=True)        

        for _ in range(self.num_blocks):
            layers.append(MyResBlock(hidden_dims, hidden_dims))
        self.res_layer = nn.Sequential(*layers) 
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
    

    def forward(self, history_feats, curr_feat):
        input_feats = torch.cat([history_feats, curr_feat.unsqueeze(1)], dim=1)
        input_feats = rearrange(input_feats, 'b t c h w -> b (t c) h w') 

        out = self.conv_in(input_feats)
        out = self.bn(out)
        out = self.relu(out)
        out = self.res_layer(out)
        if curr_feat.dim() == 3:
            out = out.squeeze(0)

        return out

