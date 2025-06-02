import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def kd_loss(logits_s, logits_t, T):
    log_s = F.log_softmax(logits_s / T, dim=1)
    p_t = F.softmax(logits_t / T, dim=1)
    return F.kl_div(log_s, p_t, reduction='batchmean') * T * T


def cc_loss(logits_s, logits_t, T):
    ps = F.softmax(logits_s / T, dim=1)
    pt = F.softmax(logits_t / T, dim=1)
    sm_s = torch.mm(ps.t(), ps)
    sm_t = torch.mm(pt.t(), pt)
    return F.mse_loss(sm_s, sm_t)


def bc_loss(logits_s, logits_t, T):
    ps = F.softmax(logits_s / T, dim=1)
    pt = F.softmax(logits_t / T, dim=1)
    bm_s = torch.mm(ps, ps.t())
    bm_t = torch.mm(pt, pt.t())
    return F.mse_loss(bm_s, bm_t)


class MLDLoss(nn.Module):
    def __init__(self, student, teacher):
        super(MLDLoss, self).__init__()
        # 直接定义超参数（避免 cfg）：
        self.temperatures = [2.0, 3.0, 5.0, 6.0]
        self.kd_weight = 0.1
        self.cc_weight = 0.1
        self.bc_weight = 0.1
        self.ce_weight = 0.1

        self.student = student  # 学生模型
        self.teacher = teacher  # 教师模型

    def forward(self, image_weak, image_strong, target, **kwargs):
        feat_s_weak, logit_s_weak = self.student(image_weak, is_feat=True)
        feat_s_strong, logit_s_strong = self.student(image_strong, is_feat=True)

        with torch.no_grad():
            feat_t_weak, logit_t_weak = self.teacher(image_weak, is_feat=True)
            feat_t_strong, logit_t_strong = self.teacher(image_strong, is_feat=True)

        loss_ce = self.ce_weight * (
                F.cross_entropy(logit_s_weak, target) +
                F.cross_entropy(logit_s_strong, target)
        )

        # 用弱图像得到 teacher 置信度和伪标签
        pred_t_weak = F.softmax(logit_t_weak, dim=1)
        confidence, pseudo_labels = pred_t_weak.max(dim=1)
        conf_mask = confidence <= torch.quantile(confidence, 0.5)

        class_conf = pred_t_weak.sum(dim=0)
        class_mask = class_conf <= torch.quantile(class_conf, 0.5)

        # 多温度蒸馏损失
        loss_kd = 0.0
        for T in self.temperatures:
            loss_kd += kd_loss(logit_s_weak, logit_t_weak, T) * conf_mask.float().mean()
            loss_kd += kd_loss(logit_s_strong, logit_t_strong, T)

        loss_cc = 0.0
        for T in self.temperatures:
            loss_cc += cc_loss(logit_s_weak, logit_t_weak, T) * class_mask.float().mean()

        loss_bc = 0.0
        for T in self.temperatures:
            loss_bc += bc_loss(logit_s_strong, logit_t_strong, T)


        losses = {
            "loss_ce": loss_ce,
            'loss_kd': self.kd_weight * loss_kd,
            'loss_cc': self.cc_weight * loss_cc,
            'loss_bc': self.bc_weight * loss_bc,
        }
        return losses
