# """
# MobileNetV2 implementation used in
# <Knowledge Distillation via Route Constrained Optimization>
# """
#
# import torch
# import torch.nn as nn
# import math
#
# __all__ = ['mobilenetv2_T_w', 'mobile_half']
#
# BN = None
#
#
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.blockname = None
#
#         self.stride = stride
#         assert stride in [1, 2]
#
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         self.conv = nn.Sequential(
#             # pw
#             nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(inp * expand_ratio),
#             nn.ReLU(inplace=True),
#             # dw
#             nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
#             nn.BatchNorm2d(inp * expand_ratio),
#             nn.ReLU(inplace=True),
#             # pw-linear
#             nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         )
#         self.names = ['0', '1', '2', '3', '4', '5', '6', '7']
#
#     def forward(self, x):
#         t = x
#         if self.use_res_connect:
#             return t + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class MobileNetV2(nn.Module):
#     """mobilenetV2"""
#     def __init__(self, T,
#                  feature_dim,
#                  input_size=32,
#                  width_mult=1.,
#                  remove_avg=False):
#         super(MobileNetV2, self).__init__()
#         self.remove_avg = remove_avg
#
#         # setting of inverted residual blocks
#         self.interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [T, 24, 2, 1],
#             [T, 32, 3, 2],
#             [T, 64, 4, 2],
#             [T, 96, 3, 1],
#             [T, 160, 3, 2],
#             [T, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(32 * width_mult)
#         self.conv1 = conv_bn(3, input_channel, 2)
#
#         # building inverted residual blocks
#         self.blocks = nn.ModuleList([])
#         for t, c, n, s in self.interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             layers = []
#             strides = [s] + [1] * (n - 1)
#             for stride in strides:
#                 layers.append(
#                     InvertedResidual(input_channel, output_channel, stride, t)
#                 )
#                 input_channel = output_channel
#             self.blocks.append(nn.Sequential(*layers))
#
#         self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
#         self.conv2 = conv_1x1_bn(input_channel, self.last_channel)
#
#         # building classifier
#         self.classifier = nn.Sequential(
#             # nn.Dropout(0.5),
#             nn.Linear(self.last_channel, feature_dim),
#         )
#
#         H = input_size // (32//2)
#         self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
#
#         self._initialize_weights()
#         print(T, width_mult)
#
#     def get_bn_before_relu(self):
#         bn1 = self.blocks[1][-1].conv[-1]
#         bn2 = self.blocks[2][-1].conv[-1]
#         bn3 = self.blocks[4][-1].conv[-1]
#         bn4 = self.blocks[6][-1].conv[-1]
#         return [bn1, bn2, bn3, bn4]
#
#     def get_feat_modules(self):
#         feat_m = nn.ModuleList([])
#         feat_m.append(self.conv1)
#         feat_m.append(self.blocks)
#         return feat_m
#
#     def forward(self, x, is_feat=False, preact=False):
#
#         out = self.conv1(x)
#         f0 = out
#
#         out = self.blocks[0](out)
#         out = self.blocks[1](out)
#         f1 = out
#         out = self.blocks[2](out)
#         f2 = out
#         out = self.blocks[3](out)
#         out = self.blocks[4](out)
#         f3 = out
#         out = self.blocks[5](out)
#         out = self.blocks[6](out)
#         f4 = out
#
#         out = self.conv2(out)
#         f4 = out
#         if not self.remove_avg:
#             out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         f5 = out
#         out = self.classifier(out)
#
#         if is_feat:
#             return [f0, f1, f2, f3, f4, f5], out
#         else:
#             return out
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
# def mobilenetv2_T_w(T, W, feature_dim=100):
#     model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
#     return model
#
#
# def mobile_half(num_classes):
#     return mobilenetv2_T_w(6, 0.5, num_classes)
#
#
# if __name__ == '__main__':
#     x = torch.randn(2, 3, 32, 32)
#
#     net = mobile_half(100)
#
#     feats, logit = net(x, is_feat=True, preact=True)
#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)
#
#     for m in net.get_bn_before_relu():
#         if isinstance(m, nn.BatchNorm2d):
#             print('pass')
#         else:
#             print('warning')
#



import sys
import torch
from torch import nn
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional, Tuple

cifar10_pretrained_weight_urls = {
    'mobilenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_5-ca14ced9.pt',
    'mobilenetv2_x0_75': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_75-a53c314e.pt',
    'mobilenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_0-fe6a5b48.pt',
    'mobilenetv2_x1_4': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt',
}

cifar100_pretrained_weight_urls = {
    'mobilenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_5-9f915757.pt',
    'mobilenetv2_x0_75': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_75-d7891e60.pt',
    'mobilenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_0-1311f9ff.pt',
    'mobilenetv2_x1_4': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_4-8a269f5e.pt',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10/100
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))


        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=1, norm_layer=norm_layer)]  # NOTE: change stride 2 -> 1 for CIFAR10/100


        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel


        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)


        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, is_feat=True, preact=False, feat_s=None, feat_t=None) -> Tensor:
        # # This exists since TorchScript doesn't support inheritance, so the superclass method
        # # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # x = self.features(x)
        # # Cannot use "squeeze" as batch-size can be 1
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x

        feat = []  # 用于存储中间特征

        # 如果提供了 feat_s，则直接将其用于特征提取
        if feat_s is not None:
            new_feat = feat_s
            layer_num = len(self.features)

            # 遍历 self.features 的每一层
            for i, layer in enumerate(self.features):
                # 如果是最后一层，则直接应用到 new_feat 上
                if i == (layer_num - 1):
                    x = layer(new_feat)
                else:
                    # 否则跳过该层
                    pass

            # 展开特征并通过分类层
            #print(x.shape)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


        # 如果没有提供 feat_s，则按顺序提取特征
        for i, layer in enumerate(self.features):
            x = layer(x)

            # 如果需要在 preact 阶段存储特征，且不是最后一层
            if preact and i < len(self.features) - 1:
                feat.append(x)

            # 如果 is_feat 为真，存储所有层的特征
            elif is_feat:
                feat.append(x)


        # 最后应用自适应平均池化、展平以及分类层
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # 如果 is_feat 为真，则返回特征列表和最终输出
        if is_feat:
            return feat, x
        else:
            return x

    def forward(self, x: Tensor, is_feat=True, preact=False, feat_s=None, feat_t=None) -> Tensor:
        return self._forward_impl(x, is_feat=is_feat, preact=preact, feat_s=feat_s, feat_t=feat_t)


def mobilenetv2_x0_5(num_classes=100, **kwargs):
    return MobileNetV2(num_classes=num_classes, width_mult=0.5, **kwargs)

def mobilenetv2_x0_75(num_classes=100, **kwargs):
    return MobileNetV2(num_classes=num_classes, width_mult=0.75, **kwargs)

def mobilenetv2_x1_0(num_classes=100, **kwargs):
    return MobileNetV2(num_classes=num_classes, width_mult=1.0, **kwargs)

def mobilenetv2_x1_4(num_classes=100, **kwargs):
    return MobileNetV2(num_classes=num_classes, width_mult=1.4, **kwargs)





