import torch.nn as nn

# 定义余弦相似度函数
def cosine_similarity(a, b, eps=1e-8):
    # 计算两个张量 a 和 b 的余弦相似度
    # 余弦相似度 = a和b的点积 / (a的模长 * b的模长 + 一个小的eps值防止除零)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

# 定义皮尔逊相关系数函数
def pearson_correlation(a, b, eps=1e-8):
    # 计算两个张量 a 和 b 的皮尔逊相关系数
    # 通过计算减去均值后的余弦相似度实现
    # a.mean(1).unsqueeze(1) 和 b.mean(1).unsqueeze(1) 用于在每个样本维度上减去均值
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)

# 定义类间关系损失函数
def inter_class_relation(y_s, y_t):
    # 计算学生模型和教师模型输出的类间关系损失
    # 1 - 皮尔逊相关系数的平均值
    return 1 - pearson_correlation(y_s, y_t).mean()

# 定义类内关系损失函数
def intra_class_relation(y_s, y_t):
    # 计算类内关系损失，采用类间关系损失的转置版本
    # 通过交换行和列，计算每类的特征间关系
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

# 定义 DIST 蒸馏类
class DIST(nn.Module):
    def __init__(self, m=1.0, n=1.0, T=1.0):
        super(DIST, self).__init__()
        self.m = m  # 控制类间损失权重
        self.n = n  # 控制类内损失权重
        self.T = T

    def forward(self, z_s, z_t):
        # 前向传播方法，计算 DIST 损失
        # z_s: 学生模型的输出
        # z_t: 教师模型的输出

        # 使用温度缩放计算学生模型和教师模型的 softmax 输出
        y_s = (z_s / self.T).softmax(dim=1)
        y_t = (z_t / self.T).softmax(dim=1)
        # 计算类间关系损失，乘以温度平方项
        inter_loss = self.T**2 * inter_class_relation(y_s, y_t)
        # 计算类内关系损失，乘以温度平方项
        intra_loss = self.T**2 * intra_class_relation(y_s, y_t)
        # 结合类间和类内损失，计算最终的蒸馏损失
        kd_loss = self.m * inter_loss + self.n * intra_loss
        return kd_loss
