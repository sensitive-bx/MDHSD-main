# '''VGG for CIFAR10. FC layers are removed.
# (c) YANG, Wei
# '''
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from .util import adaptive_channel
#
#
# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
#
#
# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
# }
#
#
# class VGG(nn.Module):
#
#     def __init__(self, cfg, batch_norm=False, num_classes=1000):
#         super(VGG, self).__init__()
#         self.block0 = self._make_layers(cfg[0], batch_norm, 3)
#         self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
#         self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
#         self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
#         self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])
#
#         self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
#         # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.classifier = nn.Linear(512, num_classes)
#         self._initialize_weights()
#
#     def get_feat_modules(self):
#         feat_m = nn.ModuleList([])
#         feat_m.append(self.block0)
#         feat_m.append(self.pool0)
#         feat_m.append(self.block1)
#         feat_m.append(self.pool1)
#         feat_m.append(self.block2)
#         feat_m.append(self.pool2)
#         feat_m.append(self.block3)
#         feat_m.append(self.pool3)
#         feat_m.append(self.block4)
#         feat_m.append(self.pool4)
#         return feat_m
#
#     def get_bn_before_relu(self):
#         bn1 = self.block1[-1]
#         bn2 = self.block2[-1]
#         bn3 = self.block3[-1]
#         bn4 = self.block4[-1]
#         return [bn1, bn2, bn3, bn4]
#
#     def forward(self, x, is_feat=False, preact=False,feat_s=None,feat_t=None):
#         if not feat_s is None:
#             # new_feat = adaptive_channel(student_feat=feat_t,teacher_feat=feat_s)
#             new_feat = feat_s
#             x = self.pool4(new_feat)
#             x = x.view(x.size(0), -1)
#             x = self.classifier(x)
#             return x
#         else:
#             h = x.shape[2]
#             x = F.relu(self.block0(x))
#             f0 = x
#             x = self.pool0(x)
#             x = self.block1(x)
#             f1_pre = x
#             x = F.relu(x)
#             f1 = x
#             x = self.pool1(x)
#             x = self.block2(x)
#             f2_pre = x
#             x = F.relu(x)
#             f2 = x
#             x = self.pool2(x)
#             x = self.block3(x)
#             f3_pre = x
#             x = F.relu(x)
#             f3 = x
#             if h == 64:
#                 x = self.pool3(x)
#             x = self.block4(x)
#             f4_pre = x
#             x = F.relu(x)
#             f4 = x
#             x = self.pool4(x)
#             x = x.view(x.size(0), -1)
#             f5 = x
#             x = self.classifier(x)
#
#             if is_feat:
#                 if preact:
#                     return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
#                 else:
#                     return [f0, f1, f2, f3, f4, f5], x
#             else:
#                 return x
#
#     def forward4(self, feat_s=None):
#         x = self.pool4(feat_s)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#
#
#     @staticmethod
#     def _make_layers(cfg, batch_norm=False, in_channels=3):
#         layers = []
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#                 if batch_norm:
#                     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#                 else:
#                     layers += [conv2d, nn.ReLU(inplace=True)]
#                 in_channels = v
#         layers = layers[:-1]
#         return nn.Sequential(*layers)
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#
# cfg = {
#     'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
#     'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
#     'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
#     'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
#     'S': [[64], [128], [256], [512], [512]],
# }
#
#
# def vgg8(**kwargs):
#     """VGG 8-layer model (configuration "S")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['S'], **kwargs)
#     return model
#
#
# def vgg8_bn(**kwargs):
#     """VGG 8-layer model (configuration "S")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['S'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg11(**kwargs):
#     """VGG 11-layer model (configuration "A")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['A'], **kwargs)
#     return model
#
#
# def vgg11_bn(**kwargs):
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     model = VGG(cfg['A'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg13(**kwargs):
#     """VGG 13-layer model (configuration "B")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['B'], **kwargs)
#     return model
#
#
# def vgg13_bn(**kwargs):
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     model = VGG(cfg['B'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg16(**kwargs):
#     """VGG 16-layer model (configuration "D")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['D'], **kwargs)
#     return model
#
#
# def vgg16_bn(**kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     model = VGG(cfg['D'], batch_norm=True, **kwargs)
#     return model
#
#
# def vgg19(**kwargs):
#     """VGG 19-layer model (configuration "E")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(cfg['E'], **kwargs)
#     return model
#
#
# def vgg19_bn(**kwargs):
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     model = VGG(cfg['E'], batch_norm=True, **kwargs)
#     return model
#
#
# if __name__ == '__main__':
#     import torch
#
#     x = torch.randn(2, 3, 32, 32)
#     net = vgg19_bn(num_classes=100)
#     feats, logit = net(x, is_feat=True, preact=True)
#
#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)
#
#     for m in net.get_bn_before_relu():
#         if isinstance(m, nn.BatchNorm2d):
#             print('pass')
#         else:
#             print('warning')




'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/vgg.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import sys
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast

cifar10_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
}

cifar100_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
}



class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()


    def forward(self, x: torch.Tensor, is_feat=True, preact=False, feat_s=None, feat_t=None) -> torch.Tensor:
        if not feat_s is None:
            new_feat = feat_s
            # x = self.pool4(new_feat) #***
            layer_num = len(self.features)
            # print("layer_num:", layer_num)

            for i, layer in enumerate(self.features):
                # print("layer-----:", i)
                if i == (layer_num - 1):
                    x = layer(new_feat)
                    # print("do_pool--------------:", x.shape)
                else:
                    pass

            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        else:
            feat = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                feat.append(x)
                # x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            if is_feat:
                return feat, x
            else:
                return x


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)




cfg = {
    'S': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['S']), **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['S'], batch_norm=True), **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model

def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model

def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model

def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model

def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model

def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model

def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


# if __name__ == '__main__':
#     import torch
#
#     x = torch.randn(2, 3, 32, 32)
#     net = vgg19_bn(num_classes=100)
#     feats, logit = net(x, is_feat=True, preact=True)
#
#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)
#
#     for m in net.get_bn_before_relu():
#         if isinstance(m, nn.BatchNorm2d):
#             print('pass')
#         else:
#             print('warning')

