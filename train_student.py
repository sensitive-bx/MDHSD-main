
from __future__ import print_function

import os
import os

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
from dataset.cifar100 import get_cifar100_dataloaders
from helper.util import adjust_learning_rate
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss, CRDLoss
from distiller_zoo.KD import DistillKL_logit_stand
from distiller_zoo.dist_kd import DIST


from helper.loops import train_distill as train_bl
from helper.loops import train_ssldistill as train_ssl
from helper.loops import train_ssldistill2 as train_ssl2

from helper.loops import validate
from helper.pretrain import init
from utils.utils import get_teacher_name, get_assistant_name, load_teacher, load_assistant, init_logging

from torchsummary import summary

from dataset.cifar10 import get_cifar10_dataloaders




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
    parser.add_argument('--ood', type=str, default='tin', choices=['tin', 'cifar100'])

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model
    parser.add_argument('--model_s', type=str, default='resnet32')


    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--path_a', type=str, default=None, help='assistant model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd')


    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    parser.add_argument('-c', type=float, default=1, help='weight balance for logit standardization KD distillation')
    parser.add_argument('-d', type=float, default=1, help='weight balance for DIST distillation')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')


    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'mobilenetv2_x0_5']:
        opt.learning_rate = 0.01

    opt.model_path = './Results/kdssl_%s/student_model' % str(opt.v2)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    if opt.model_s == 'repvgg_a0':
        opt.model_t = "repvgg_a2"
        opt.model_a = "repvgg_a1"
    else:
        opt.model_t = get_teacher_name(opt.path_t)
        opt.model_a = get_assistant_name(opt.path_a)



    # 如果学生模型是mobilenetv2_x0_5，助理模型是如果学生模型是mobilenetv2_x1_0，教师模型是如果学生模型是mobilenetv2_x1_4，则将上面注释掉
    # opt.model_t = 'mobilenetv2_x1_4'
    # opt.model_a = 'mobilenetv2_x1_0'
    # opt.model_s == 'mobilenetv2_x0_5'



    opt.model_name = 'O:{}S:{}_A:{}_T:{}_{}_{}_r:{}_a:{}_b:{}e:{}_{}'.format(
        opt.ood, opt.model_s, opt.model_a, opt.model_t, opt.dataset,
        opt.distill, opt.gamma, opt.alpha, opt.beta,
        str(os.environ['CONDA_DEFAULT_ENV']), str(time.ctime())
    )
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)
    print(opt.save_folder)
    log_root = logging.getLogger()
    init_logging(log_root, opt.save_folder)

    return opt



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

    # Load teacher model
    model_t = load_teacher(opt.path_t, n_cls)
    if torch.cuda.is_available():
        model_t = model_t.cuda()


    # Validate teacher model accuracy before training assistant model
    teacher_acc, _, _ = validate(val_loader, model_t, nn.CrossEntropyLoss(), opt)
    print('Teacher accuracy: ', teacher_acc)

    # 使用torchsummary打印教师模型参数
    #summary(model_t, (3, 32, 32))

    # Initialize assistant models
    #model_a = model_dict[opt.model_a](num_classes=n_cls)

    model_a = load_assistant(opt.path_a, n_cls)
    if torch.cuda.is_available():
        model_a = model_a.cuda()

    # 使用torchsummary打印助理模型参数
    #summary(model_a, (3, 32, 32))

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_a.eval()
    feat_t, _ = model_t(data.cuda(), is_feat=True)
    feat_a, _ = model_a(data.cuda(), is_feat=True)

    # Training assistant model
    module_list = nn.ModuleList([model_a])
    trainable_list = nn.ModuleList([model_a])

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_div_logit_stand = DistillKL_logit_stand(opt.kd_T)
    criterion_dist = DIST(m=1.0, n=1.0, T=opt.kd_T)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_a = ConvReg(feat_a[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_a)
        trainable_list.append(regress_a)
    elif opt.distill == 'crd':
        opt.s_dim = feat_a[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif 'srd' in opt.distill:
        opt.s_dim = feat_a[-2].shape[1]
        opt.t_dim = feat_t[-2].shape[1]
        connector = transfer_conv(opt.s_dim, opt.t_dim)
        module_list.append(connector)
        # add this because connector need to updated
        trainable_list.append(connector)
        criterion_kd = statm_loss()
    elif opt.distill == 'srd_kd':
        print("opt.distill == 'srd_kd'")
        # 设置 KD 和 SRD 的参数
        opt.s_dim = feat_a[-2].shape[1]
        opt.t_dim = feat_t[-2].shape[1]
        # 初始化连接器（用于 SRD 部分）
        connector = transfer_conv(opt.s_dim, opt.t_dim)
        module_list.append(connector)
        trainable_list.append(connector)
        # 初始化 KD 和 SRD 的损失函数
        criterion_kd = statm_loss()  # SRD 的损失函数
        criterion_div_logit_stand = DistillKL_logit_stand(opt.kd_T)  # 添加logit_stand的KD的损失函数
    elif opt.distill == 'srd_kd_dist':
        print("opt.distill == 'srd_kd_dist'")
        # 设置 KD 和 SRD 的参数
        opt.s_dim = feat_a[-2].shape[1]
        opt.t_dim = feat_t[-2].shape[1]
        # 初始化连接器（用于 SRD 部分）
        connector = transfer_conv(opt.s_dim, opt.t_dim)
        module_list.append(connector)
        trainable_list.append(connector)
        # 初始化 dist 和 SRD 的损失函数
        criterion_kd = statm_loss()  # SRD 的损失函数
        criterion_div_logit_stand = DistillKL_logit_stand(opt.kd_T)  # 添加logit_stand的KD的损失函数
        criterion_dist = DIST(m=1.0, n=1.0, T=opt.kd_T)
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
        embed_s = LinearEmbed(feat_a[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_a[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_a[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([connector] + model_a.get_feat_modules())
        criterion_kd = ABLoss(len(feat_a[1:-1]))
        #init(model_a, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_a[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        init_trainable_list = nn.ModuleList([paraphraser])
        criterion_init = nn.MSELoss()
        init(model_a, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_a[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList(model_a.get_feat_modules())
        #init(model_a, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([
        criterion_cls,  # classification loss
        criterion_div,  # KL divergence loss, original knowledge distillation
        criterion_kd,   # other knowledge distillation loss
        criterion_div_logit_stand,
        criterion_dist
    ])

    # optimizer
    optimizer_a = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    temp = 0
    # Training assistant model
    print("==> Training assistant model...")
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer_a)

        time1 = time.time()
        if opt.distill == 'crd':
            train_acc, train_loss = train_bl(epoch, train_loader, module_list, criterion_list, optimizer_a, opt)
        else:
            if opt.v2:
                train_acc, train_loss = train_ssl2(epoch, train_loader, utrain_loader, module_list, criterion_list, optimizer_a, opt)
            else:
                train_acc, train_loss = train_ssl(epoch, train_loader, utrain_loader, module_list, criterion_list, optimizer_a, opt, temp)

        time2 = time.time()
        print('Assistant model epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_a, criterion_cls, opt)
        logger.add_scalar('train/train_loss_a', train_loss, epoch)
        logger.add_scalar('train/train_acc_a', train_acc, epoch)
        logger.add_scalar('test/test_acc_a', test_acc, epoch)
        logger.add_scalar('test/test_loss_a', test_loss, epoch)

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     best_acc_top5 = tect_acc_top5
        #     state = {
        #         'epoch': epoch,
        #         'model': model_a.state_dict(),
        #         'best_acc': best_acc,
        #     }
        #     save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_a))
        #     print('Saving the best assistant model!')
        #     torch.save(state, save_file)
        # msg = "Epoch %d test_acc %.3f, test_top5 %.3f, best_acc %.3f, best_top5 %.3f" % (
        #     epoch, test_acc, tect_acc_top5, best_acc, best_acc_top5)
        # logging.info(msg)



    # Now train the student model with the trained assistant model
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    if torch.cuda.is_available():
        model_s = model_s.cuda()

    # 使用torchsummary打印学生模型参数
    #summary(model_s, (3, 32, 32))

    data = torch.randn(2, 3, 32, 32)
    model_a.eval()
    model_s.eval()
    feat_a, _ = model_a(data.cuda(), is_feat=True)
    feat_s, _ = model_s(data.cuda(), is_feat=True)

    # # 将学生模型设为评估模式
    # model_s.eval()

    # Update the module list
    module_list = nn.ModuleList([model_s, model_a])
    trainable_list = nn.ModuleList([model_s])

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_div_logit_stand = DistillKL_logit_stand(opt.kd_T)
    criterion_dist = DIST(m=1.0, n=1.0, T=1.0)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'srd_kd_dist':
        #print("opt.distill == 'srd_kd_dist'")
        # 设置 KD 和 SRD 的参数
        opt.s_dim = feat_s[-2].shape[1]
        opt.t_dim = feat_a[-2].shape[1]
        # 初始化连接器（用于 SRD 部分）
        connector = transfer_conv(opt.s_dim, opt.t_dim)
        module_list.append(connector)
        trainable_list.append(connector)
        # 初始化 dist 和 SRD 的损失函数
        criterion_kd = statm_loss()  # SRD 的损失函数
        criterion_div_logit_stand = DistillKL_logit_stand(opt.kd_T)  # 添加logit_stand的KD的损失函数
        criterion_dist = DIST(m=1.0, n=1.0, T=1.0)
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_a[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_a[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        init_trainable_list = nn.ModuleList([paraphraser])
        criterion_init = nn.MSELoss()
        init(model_s, model_a, init_trainable_list, criterion_init, train_loader, logger, opt)
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
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
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_a[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        trainable_list.append(criterion_kd)

    else:
        raise NotImplementedError(opt.distill)


    criterion_list = nn.ModuleList([
        criterion_cls,  # classification loss
        criterion_div,  # KL divergence loss, original knowledge distillation
        criterion_kd,  # other knowledge distillation loss
        criterion_div_logit_stand,
        criterion_dist
    ])


    # optimizer
    optimizer_s = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher and assistant after optimizer to avoid weight_decay
    module_list.append(model_t)
    module_list.append(model_a)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # Training student model
    print("==> Training student model...")
    temp = 1
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer_s)

        time1 = time.time()
        if opt.distill == 'crd':
            train_acc, train_loss = train_bl(epoch, train_loader, module_list, criterion_list, optimizer_s, opt)
        else:
            if opt.v2:
                train_acc, train_loss = train_ssl2(epoch, train_loader, utrain_loader, module_list, criterion_list, optimizer_s, opt)
            else:
                train_acc, train_loss = train_ssl(epoch, train_loader, utrain_loader, module_list, criterion_list, optimizer_s, opt, temp)

        time2 = time.time()
        print('Student model epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
        logger.add_scalar('train/train_loss', train_loss, epoch)
        logger.add_scalar('train/train_acc', train_acc, epoch)
        logger.add_scalar('test/test_acc', test_acc, epoch)
        logger.add_scalar('test/test_loss', test_loss, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_top5 = tect_acc_top5
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('Saving the best student model!')
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



