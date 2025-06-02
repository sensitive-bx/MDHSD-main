import os
import sys
import torch
import logging
from models import model_dict

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def get_assistant_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')

    model_t = get_teacher_name(model_path)

    #如果教师模型是repvgg_a2
    #model_t = "repvgg_a2"

    #如果教师模型是mobilenetv2_x1_4
    #model_t = "mobilenetv2_x1_4"

    model = model_dict[model_t](num_classes=n_cls)

    if model_t == 'resnet110':
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        model.load_state_dict(torch.load(model_path))

    print('==> done')
    return model


def load_assistant(model_path, n_cls):
    print('==> loading assistant model')

    model_a = get_assistant_name(model_path)

    # 如果助理模型是repvgg_a2
    #model_a = "repvgg_a1"

    # 如果助理模型是mobilenetv2_x1_0
    # model_a = "mobilenetv2_x1_0"

    model = model_dict[model_a](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path))
    print('==> done')
    return model

def init_logging(log_root, models_root):
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
    handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)
