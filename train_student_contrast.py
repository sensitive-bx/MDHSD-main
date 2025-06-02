"""
the general training framework
"""

from __future__ import print_function

import os


import math



import argparse
import time
import logging

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import model_dict
from models.util import ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser
from distiller_zoo.AIN import transfer_conv, statm_loss
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from helper.util import adjust_learning_rate
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss, CRDLoss

from helper.loops_contrast import train_distill as train_bl
from helper.loops_contrast import train_ssldistill as train_ssl
from helper.loops_contrast import train_ssldistill2 as train_ssl2

from helper.loops_contrast import validate
from helper.pretrain import init
from utils.utils_contrast import get_teacher_name, load_teacher, init_logging

from models.temp_global import Global_T


from distiller_zoo.MLLD import MLDLoss
from distiller_zoo.FAM import FAMLoss
from distiller_zoo.NKD import NKDLoss
from distiller_zoo.KD import DistillKL_CTKD

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--v2', action='store_true', help='seperate batch or not')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # labeled dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='dataset')

    # select unlabeled dataset
    parser.add_argument('--ood', type=str, default='tin',
                        choices=['tin', 'places', 'cifar100'])  # 选择没标签的数据集，默认为tin，可自己改成places

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8x4')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # CTKD distillation
    parser.add_argument('--have_mlp', type=int, default=0)
    parser.add_argument('--mlp_name', type=str, default='global')
    parser.add_argument('--t_start', type=float, default=1)
    parser.add_argument('--t_end', type=float, default=20)
    parser.add_argument('--cosine_decay', type=int, default=1)
    parser.add_argument('--decay_max', type=float, default=0)
    parser.add_argument('--decay_min', type=float, default=0)
    parser.add_argument('--decay_loops', type=float, default=0)

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'mobilenetv2_x0_5']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './Results/kdssl_%s/student_model' % str(opt.v2)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.model_s == 'repvgg_a0':
        opt.model_t = "repvgg_a2"
    else:
        opt.model_t = get_teacher_name(opt.path_t)



    # 如果学生模型是mobilenetv2_x0_5，教师模型是如果学生模型是mobilenetv2_x1_4，则将上面注释掉
    # opt.model_t = 'mobilenetv2_x1_4'
    # opt.model_s == 'mobilenetv2_x0_5'

    opt.model_name = 'O:{}S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}e:{}_{}'.format(opt.ood, opt.model_s, opt.model_t, opt.dataset,
                                                                        opt.distill, opt.gamma, opt.alpha, opt.beta,
                                                                        str(os.environ['CONDA_DEFAULT_ENV']),
                                                                        str(time.ctime()))
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)
    print(opt.save_folder)
    log_root = logging.getLogger()
    init_logging(log_root, opt.save_folder)

    return opt


class CosineDecay(object):
    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class LinearDecay(object):
    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1

        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (-value)

        return value


def main():
    best_acc = 0

    opt = parse_option()

    logger = SummaryWriter(opt.save_folder)

    # dataloader
    if opt.dataset == 'cifar100':
        n_cls = 100
        if opt.distill == 'crd':
            model_t = get_teacher_name(opt.path_t)
            train_loader, _, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                           num_workers=opt.num_workers,
                                                                           is_sample=True,
                                                                           is_instance=False,
                                                                           k=opt.nce_k,
                                                                           mode=opt.mode,
                                                                           ood=opt.ood,
                                                                           model=model_t)
        else:
            train_loader, utrain_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                                       num_workers=opt.num_workers,
                                                                                       is_instance=True,
                                                                                       is_sample=False,
                                                                                       ood=opt.ood)

    elif opt.dataset == 'cifar10':
        n_cls = 10
        if opt.distill == 'crd':
            model_t = get_teacher_name(opt.path_t)
            train_loader, _, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                          num_workers=opt.num_workers,
                                                                          is_sample=True,
                                                                          is_instance=False,
                                                                          k=opt.nce_k,
                                                                          mode=opt.mode,
                                                                          ood=opt.ood,
                                                                          model=model_t)
        else:
            train_loader, utrain_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                                      num_workers=opt.num_workers,
                                                                                      is_instance=True,
                                                                                      is_sample=False,
                                                                                      ood=opt.ood)

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)

    mlp = None

    if opt.have_mlp:
        if opt.mlp_name == 'global':
            print("opt.mlp_name == 'global'")
            mlp = Global_T()
        else:
            print('mlp name wrong')

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()

    #区分是KD还是CTKD
    if not opt.have_mlp:
        criterion_div = DistillKL(opt.kd_T)
    else:
        criterion_div = DistillKL_CTKD()

    if opt.distill == 'kd':
        if not opt.have_mlp:
            criterion_kd = DistillKL(opt.kd_T)
        else:
            criterion_kd = DistillKL_CTKD()

    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif 'srd' in opt.distill:
        opt.s_dim = feat_s[-2].shape[1]
        opt.t_dim = feat_t[-2].shape[1]
        connector = transfer_conv(opt.s_dim, opt.t_dim)
        module_list.append(connector)
        # add this because connector need to to updated
        trainable_list.append(connector)
        criterion_kd = statm_loss()
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        #init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        #init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass

    elif opt.distill == 'mlld':
        criterion_kd = MLDLoss(model_s, model_t)

    elif opt.distill == 'fam':
        # Feature dimensions from sample forward pass for FAM
        s_channels = [f.shape[1] for f in feat_s[1:-1]]  # Skip first and last layers
        t_channels = [f.shape[1] for f in feat_t[1:-1]]  # Skip first and last layers
        criterion_kd = FAMLoss(s_channels, t_channels)
        # Add FAM modules to trainable list
        trainable_list.append(criterion_kd)

    elif opt.distill == 'nkd':
        criterion_kd = NKDLoss(gamma=opt.gamma, temp=opt.kd_T)

    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    if opt.cosine_decay:
        gradient_decay = CosineDecay(max_value=opt.decay_max, min_value=opt.decay_min, num_loops=opt.decay_loops)
    else:
        gradient_decay = LinearDecay(max_value=opt.decay_max, min_value=opt.decay_min, num_loops=opt.decay_loops)

    decay_value = 1

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        if opt.have_mlp:
            decay_value = gradient_decay.get_value(epoch)

        time1 = time.time()
        if opt.distill == 'crd':
            #print("train_acc, train_loss = train_bl(epoch, train_loader, module_list, criterion_list, optimizer, opt)")
            train_acc, train_loss = train_bl(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        else:
            if opt.v2:
                # in the seperate batch

                train_acc, train_loss = train_ssl2(epoch, train_loader, utrain_loader, module_list, criterion_list,
                                                   optimizer, opt)
            else:
                # in the same batch
                train_acc, train_loss = train_ssl(epoch, train_loader, utrain_loader, module_list, mlp, decay_value,
                                                  criterion_list, optimizer, opt)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.add_scalar('train/train_loss', train_loss, epoch)
        logger.add_scalar('train/train_acc', train_acc, epoch)
        logger.add_scalar('test/test_acc', test_acc, epoch)
        logger.add_scalar('test/test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_top5 = tect_acc_top5
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)
        msg = "Epoch %d test_acc %.3f, test_top5 %.3f, best_acc %.3f, best_top5 %.3f" % (
            epoch, test_acc, tect_acc_top5, best_acc, best_acc_top5)
        logging.info(msg)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'accuracy': test_acc
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()







