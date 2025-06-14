from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet50, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn

#from .mobilenetv2 import mobile_half

from .mobilenetv2 import mobilenetv2_x0_5, mobilenetv2_x0_75, mobilenetv2_x1_0, mobilenetv2_x1_4

from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2

from .repvgg import repvgg_a0, repvgg_a1, repvgg_a2


model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet50': resnet50,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,

    #'MobileNetV2': mobile_half,

    'mobilenetv2_x0_5':mobilenetv2_x0_5,
    'mobilenetv2_x0_75':mobilenetv2_x0_75,
    'mobilenetv2_x1_0':mobilenetv2_x1_0,
    'mobilenetv2_x1_4':mobilenetv2_x1_4,

    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,

    'repvgg_a0': repvgg_a0,
    'repvgg_a1': repvgg_a1,
    'repvgg_a2': repvgg_a2
}
