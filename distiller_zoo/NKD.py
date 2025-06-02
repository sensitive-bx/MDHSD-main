import torch
import torch.nn as nn
import torch.nn.functional as F


class NKDLoss(nn.Module):
    """
    归一化知识蒸馏损失(Normalized Knowledge Distillation Loss)

    参考论文: "From Knowledge Distillation to Self-Knowledge Distillation:
    A Unified Approach with Normalized Loss and Customized Soft Labels"
    """

    def __init__(self, gamma=1.0, temp=1.0):
        super(NKDLoss, self).__init__()
        self.gamma = gamma  # 非目标类损失的权重系数
        self.temp = temp  # 温度参数

    def forward(self, logit_s, logit_t, target=None):
        # 应用温度缩放
        scaled_logit_s = logit_s / self.temp
        scaled_logit_t = logit_t / self.temp

        # 将logits转换为概率
        s_probs = F.softmax(scaled_logit_s, dim=1)
        t_probs = F.softmax(scaled_logit_t, dim=1)

        if target is not None:
            # 获取批次大小和类别数量
            batch_size, num_classes = logit_s.size()

            # 创建目标类掩码
            mask = torch.zeros_like(s_probs).scatter_(1, target.unsqueeze(1), 1)

            # 获取目标类概率
            s_target = torch.sum(s_probs * mask, dim=1)
            t_target = torch.sum(t_probs * mask, dim=1)

            # 创建非目标类掩码
            non_target_mask = 1 - mask

            # 获取非目标类概率
            s_non_target = s_probs * non_target_mask
            t_non_target = t_probs * non_target_mask

            # 归一化非目标类概率
            s_sum = torch.sum(s_non_target, dim=1, keepdim=True)
            t_sum = torch.sum(t_non_target, dim=1, keepdim=True)

            # 避免除零问题
            s_sum = torch.clamp(s_sum, min=1e-8)
            t_sum = torch.clamp(t_sum, min=1e-8)

            # 归一化操作 N(S_i) = S_i / (1-S_t)
            norm_s = s_non_target / s_sum
            norm_t = t_non_target / t_sum

            # 计算目标类损失
            target_loss = -torch.mean(t_target * torch.log(torch.clamp(s_target, min=1e-8)))

            # 计算归一化后的非目标类损失
            non_target_loss = -torch.mean(torch.sum(
                norm_t * torch.log(torch.clamp(norm_s, min=1e-8)),
                dim=1
            ))

            # 总损失 = 目标类损失 + gamma * temp^2 * 非目标类损失
            loss = target_loss + self.gamma * (self.temp ** 2) * non_target_loss

        else:
            # 当目标未知时，对所有类应用NKD（回退到常规KL散度）
            loss = F.kl_div(
                F.log_softmax(scaled_logit_s, dim=1),
                F.softmax(scaled_logit_t, dim=1),
                reduction='batchmean'
            ) * (self.temp ** 2)

        return loss